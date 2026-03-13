from __future__ import annotations

from datetime import timedelta
from typing import Dict, NamedTuple, Tuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, set_global_ctx
from minisgl.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from minisgl.kvcache import create_kvcache
from minisgl.layers import set_rope_device
from minisgl.models import create_model, load_hf_weight
from minisgl.utils import divide_even, init_logger, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)

# GPU → CPU 异步拷贝：next_tokens_gpu 在 GPU 上，next_tokens_cpu 在 CPU 上。
# 异步拷贝避免阻塞 CPU。
# 事件同步：copy_done_event 用于等待异步拷贝完成。
class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor # GPU 上的 next token IDs
    next_tokens_cpu: torch.Tensor # 同步到 CPU 的版本
    copy_done_event: torch.cuda.Event # 异步拷贝完成事件


def create_page_table(shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.int32, device=device)


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32


class Engine:
    def __init__(self, config: EngineConfig):
        """初始化推理引擎"""
        self.model_config = config.model_config
        # 设置张量并行信息
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)

        assert not torch.cuda.is_initialized()
        # 初始化 CUDA 设备和流
        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype

        # 初始化分布式通信（NCCL/gloo）
        self.tp_cpu_group = self._init_communication(config)

        # 同步所有 TP rank 的内存状态。
        # 该方法首先同步 CUDA 设备并清空缓存，获取当前空闲内存。
        # 然后使用 all_reduce 获取所有 rank 的最小和最大内存。
        # 如果发现最大和最小内存差异超过 2GB，则抛出异常，提示内存在 TP ranks 之间不平衡。
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # load model and determine number of pages
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_path, config.model_config)
        
        # 加载模型权重
        self.model.load_state_dict(self._load_weight_state_dict(config))
        
        # 确定 KV 缓存页数（根据可用内存计算）
        self.num_pages = self.dummy_page = self._determine_num_pages(init_free_memory, config)


        self.kv_cache = create_kvcache(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            device=self.device,
            dtype=self.dtype,
        )

        # NOTE: make page table 128 aligned (32 * sizeof(int32) == 128 bytes)
        self.max_seq_len = _align_up_32(min(config.max_seq_len, self.num_pages))

        self.page_table = create_page_table(  # + 1 for dummy request
            (config.max_running_req + 1, self.max_seq_len),
            device=self.device,
        )

        self.attn_backend = create_attention_backend(
            config.attention_backend,
            config.model_config,
            self.kv_cache,
            self.page_table,
        )

        self.ctx = Context(page_size=1, attn_backend=self.attn_backend)
        set_global_ctx(self.ctx)


        self.sampler = Sampler(self.device, self.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # cuda graph related
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )

        self.page_table[self.dummy_req.table_idx].fill_(self.dummy_page)


        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=self.max_seq_len,
            vocab_size=self.model_config.vocab_size,
            dummy_req=self.dummy_req,
        )

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        if config.tp_info.size == 1 or config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        if config.use_dummy_weight:
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            return {
                k: v.to(self.dtype)
                for k, v in load_hf_weight(config.model_path, self.device).items()
            }

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        new_free_memory = self._sync_get_memory()[1]
        cache_per_page = (
            2  # key + value
            * self.model_config.head_dim
            * divide_even(self.model_config.num_kv_heads, config.tp_info.size)
            * config.page_size
            * self.dtype.itemsize
            * self.model_config.num_layers
        )
        num_pages = config.num_page_override
        if num_pages is None:
            model_memory = old_free_memory - new_free_memory
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            num_pages = available_memory // cache_per_page

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-tokens"
        real_kv_size = num_pages * cache_per_page
        logger.info(f"Allocating {num_pages} pages for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        """获取所有 TP rank 的最小和最大可用内存"""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        free_memory = get_free_memory(self.device)
        
        # 使用 all_reduce 获取所有 rank 的内存信息
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())

        # 检查内存平衡
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    """执行批次的前向传播"""
    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream
        with self.ctx.forward_batch(batch):
            # 判断是否可以使用 CUDA Graph
            if self.graph_runner.can_use_cuda_graph(batch):
                # 使用 CUDA Graph 重放（更快）
                logits = self.graph_runner.replay(batch)
            else:
                # 正常前向传播
                logits = self.model.forward()

        for req in batch.reqs:
            req.complete_one()

        # 采样得到下一个 token（注意切片和类型转换）
        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        # 异步拷贝到 CPU
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        # 创建同步事件并记录到当前流
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)


        """
        CUDA Graph 条件判断：can_use_cuda_graph(batch) 检查是否为 Decode 阶段且 batch size 在预捕获范围内
        Graph 重放 vs 正常推理：Decode 阶段使用 replay() 消除 Python 开销，Prefill 阶段正常执行 model.forward()
        状态更新：每次推理后调用 req.complete_one() 更新请求的 cached_len 和 device_len
        类型转换：采样结果转换为 int32 类型以节省内存
        """

        # 更新状态、采样、拷贝
        # Engine.forward_batch() 返回的不是简单的 logits，而是一个结构化的结果：
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()
