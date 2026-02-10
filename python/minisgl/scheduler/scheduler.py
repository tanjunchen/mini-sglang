from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
import torch.nn.functional as F
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import init_logger
from transformers import AutoTokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    load_indices: torch.Tensor
    write_indices: torch.Tensor


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"

"""
Scheduler 它的核心价值是“决策与编排”，本质是一层控制器，把上层的请求流转化为GPU每一轮的执行指令

本轮推理执行prefill（前缀填充）还是decode（增量解码）；
为当前批次请求分配多少KV页（KV pages）；
请求结束后，如何将其KV数据转化为可复用的前缀缓存；

Scheduler 的核心运行依赖主循环，mini-sglang默认采用overlap_loop，
这也是它吞吐性能出色的关键——实现了“当前轮GPU前向计算”与“上一轮CPU处理/回包/资源释放”的流水线重叠，最大化硬件利用率。


第四类是 Scheduler Workers（每个 GPU 一个），这是核心调度器，管理推理任务的调度和执行。
每个 GPU 对应一个 Scheduler Worker（TP Rank），负责管理 Engine、KV 缓存和资源分配。
实现代码位于 scheduler.py。
"""
class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        # 负责模型前向传播、注意力后端、CUDA图优化，以及最关键的物理KV缓存页和页表（page_table，KV页的索引入口）
        self.engine = Engine(config)
        # Initialize the I/O mixin
        # 装配控制面通信：继承SchedulerIOMixin，通过super().__init__(config, self.engine.tp_cpu_group)
        # 封装通信逻辑，包括rank0节点对外收发消息、多TP节点间的广播同步。
        super().__init__(config, self.engine.tp_cpu_group)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        # 两个 CUDA 流
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        # 初始化调度状态：整合TableManager、CacheManager、PrefillManager/DecodeManager三大模块。
        # 其中，表结构（token_pool/page_table）承载GPU侧请求状态，CacheManager负责KV页的分配、复用与驱逐，
        # Prefill/Decode管理器则负责将请求打包成批次（batch）。
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(self.device, self.engine.num_pages, config.cache_type)
        self.decode_manager = DecodeManager()
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        self.tp_info = config.tp_info
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.page_table = self.engine.page_table
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        self.dummy_write_2d_pos = (self.engine.dummy_req.table_idx, 1, 2)  # 0 for load, 1 for write

    def _process_last_data(
        self, last_data: ForwardData | None, ongoing_data: ForwardData | None
    ) -> None:
        if last_data is None:
            return
        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []

        for i, req in enumerate(batch.reqs):
            if req in self.finished_reqs or isinstance(req, ChunkedReq):
                continue

            next_token_id = next_tokens_cpu[i]
            req.append_host(next_token_id.unsqueeze(0))
            next_token = int(next_token_id.item())
            finished = not req.can_decode()
            if not req.sampling_params.ignore_eos:
                finished |= next_token == self.eos_token_id
            reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

            # free resources if the req is finished and not ongoing
            if finished:
                self.finished_reqs.add(req)
                self.decode_manager.remove_req(req)
                logger.debug_rank0("Request %s is finished", req)

        # free resources for finished but not ongoing reqs
        ongoing_reqs = ongoing_data[0].batch.reqs if ongoing_data else []
        for req in self.finished_reqs.difference(ongoing_reqs):
            self.table_manager.free(req.table_idx)
            self.cache_manager.free_and_cache_finished_req(
                req.cache_handle,
                req.input_ids[: req.cached_len],
                self.page_table[req.table_idx, : req.cached_len],
            )

        # keep only ongoing reqs in the finished set
        self.finished_reqs.intersection_update(ongoing_reqs)
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    """
    批次准备过程包括分配 KV 缓存页、准备元数据等操作
    """
    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        # 分配 KV 缓存页
        needed_size = sum(r.extend_len for r in batch.reqs)
        batch.out_loc = self.cache_manager.allocate(needed_size)
        
        # NOTE: Pad the batch if needed
        if padding_size := self.engine.graph_runner.pad_batch(batch):
            batch.out_loc = F.pad(batch.out_loc, (0, padding_size), value=self.engine.dummy_page)
        
        # NOTE: prepare 2d indices for token ids loading and writing
        # 准备 token IDs 加载/写入索引
        load_indices = self._make_2d_indices(
            [(r.table_idx, r.cached_len, r.device_len) for r in batch.padded_reqs]
        )
        write_indices = self._make_2d_indices(
            [
                (
                    (r.table_idx, r.device_len, r.device_len + 1)
                    if r.can_decode()  # NOTE: for chunked req, write to dummy pos
                    else self.dummy_write_2d_pos
                )
                for r in batch.reqs
            ]
        )
        assert all(r.device_len < self.engine.max_seq_len for r in batch.reqs)
        # NOTE: write out_loc to page_table before `prepare_metadata`
        # 写入页表
        self.page_table.view(-1)[load_indices] = batch.out_loc
        
        # 准备注意力元数据
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            load_indices=load_indices,
            write_indices=write_indices,
        )

    """
    而本轮执行prefill还是decode的策略，代码实现非常简洁，核心是“prefill优先”：
    """
    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        # 优先调度 Prefill，然后调度 Decode
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        return self._prepare_batch(batch) if batch else None

    def _make_2d_indices(self, ranges: List[Tuple[int, int, int]]) -> torch.Tensor:
        """
        Return the 1D indices for the given 2D table and ranges.

        Example: The underlying indices of a 2D table (3, 4) are:
            [[ 0,  1,  2,  3],
             [ 4,  5,  6,  7],
             [ 8,  9, 10, 11]]
        For ranges [(0, 1, 3), (2, 0, 2)], the returned indices are [1, 2, 8, 9].

        Args:
            ranges (List[Tuple[int, int, int]]): A list of tuples (entry, begin, end),
                where `entry` is the row index in the 2D table, and `begin` and `end`
                specify the range of column indices to include.
        Returns:
            torch.Tensor: A 1D tensor of indices.
        """
        STRIDE = self.token_pool.stride(0)
        needed_size = sum(end - begin for _, begin, end in ranges)
        indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
        offset = 0
        for entry, begin, end in ranges:
            length = end - begin
            offset += length
            torch.arange(
                begin + entry * STRIDE,
                end + entry * STRIDE,
                dtype=torch.int32,
                out=indices_host[offset - length : offset],
            )
        return indices_host.to(self.device, non_blocking=True)

    def _load_token_ids(self, input: ForwardInput) -> None:
        input.batch.input_ids = self.token_pool.view(-1)[input.load_indices]

    def _write_token_ids(self, input: ForwardInput, output: ForwardOutput) -> None:
        self.token_pool.view(-1)[input.write_indices] = output.next_tokens_gpu

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        self._load_token_ids(forward_input)
        batch, sample_args = forward_input.batch, forward_input.sample_args
        if ENV.OVERLAP_EXTRA_SYNC:  # NOTE: https://github.com/sgl-project/mini-sglang/issues/58
            self.stream.synchronize()
        forward_output = self.engine.forward_batch(batch, sample_args)
        self._write_token_ids(forward_input, forward_output)
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()


    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """重叠调度循环：CPU 调度与 GPU 计算并行"""
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        blocking = not (
            last_data  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        # 先通过receive_msg接收新请求，这是持续批处理（Continuous batching）的基础
        for msg in self.receive_msg(blocking=blocking):
            # 接收新消息（非阻塞）
            self._process_one_msg(msg)

        # 调用_schedule_next_batch组装下一轮执行的batch，调度下一个批次
        forward_input = self._schedule_next_batch()
        ongoing_data = None
        # 在 Engine 流中执行推理（GPU 计算）
        if forward_input is not None:
            # 一旦batch就绪，就在Engine的流上提交_forward任务，GPU开始执行本轮计算
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        # GPU运算的同时，CPU立刻回头执行_process_last_data，处理上一轮的输出token——判断请求是否结束、
        # 向上游回包，同时回收已结束请求的资源，并将其KV数据回填缓存。
        # 并行处理上一批次的结果（CPU 处理）
        self._process_last_data(last_data, ongoing_data)
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data, None)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        """调度器主循环"""
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                # 普通模式：顺序执行
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                # 重叠调度模式：并行执行
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()
