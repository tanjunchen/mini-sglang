from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Dict, List

import torch
from minisgl.core import Batch, Req, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import init_logger
from tqdm import tqdm

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend
    from minisgl.models import BaseLLMModel

logger = init_logger(__name__)

# 自动确定图形批次大小
def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    return [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))


def mem_GB(size: int) -> str:
    return f"{size / (1024**3):.2f} GiB"


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]

"""
GraphRunner 是 CUDA Graph 运行器，初始化时接收 stream、device、model、attn_backend、cuda_graph_bs 等参数。
它为不同的 batch size 预捕获图，维护 graphs（batch_size -> CUDAGraph）、static_inputs（batch_size -> 静态输入）
和 static_outputs（batch_size -> 静态输出）字典。初始化时会捕获所有 batch size 的图。
"""
class GraphRunner:
    # 实际构造函数参数更多，包括 stream, device, model, attn_backend 等
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.stream = stream
        self.device = device
        """
        为什么需要 CUDA Graph？ 
        CUDA Kernel 启动存在隐藏成本。每次调用 CUDA kernel（如 matmul、attention）都需要 CPU 发出指令，每个 kernel 启动有 5-20μs 的 CPU 开销。
        Transformer 模型的一次 forward pass 有 100+ 个 kernel 调用，总 CPU 开销达到 100 × 10μs = 1ms。

        问题在 Decode 阶段尤为突出。
        每次只生成 1 个 token，计算量很小（可能只需 2-3ms），CPU 启动开销占总时间的 25-50%。
        小 batch size 时问题更严重（batch=1 时，计算可能只需 1ms，开销占 50%+）。

        为什么不能用 Overlap Scheduling 解决？ 
        Overlap Scheduling 只能隐藏”调度”开销，无法隐藏”kernel 启动”开销。
        Kernel 启动发生在 GPU 计算过程中，必须串行执行。

        CUDA Graph 通过捕获一系列 CUDA 操作并将其作为图（graph）重放，
        减少 CPU-GPU 同步开销，显著提升 Decode 阶段的性能。
        """
        self.graph_map = self._capture_graphs(max_seq_len, vocab_size, model)

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        
        graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            logger.info_rank0("CUDA graph is disabled.")
            return graph_map

        self.logits = torch.empty(
            (self.max_graph_bs, vocab_size),
            dtype=torch.float32,
            device=self.device,
        )
        """捕获不同 batch_size 的计算图"""
        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            self.attn_backend.prepare_for_capture(batch)
            with get_global_ctx().forward_batch(batch):
                self.logits[:bs] = model.forward()
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.logits[:bs] = model.forward()
            if pool is None:
                pool = graph.pool()
            graph_map[bs] = graph

        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")
        return graph_map

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        """判断是否可以使用 CUDA Graph来捕获和重播计算图"""
        return batch.is_decode and batch.size <= self.max_graph_bs

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        """重放计算图"""
        g.replay()
        return self.logits[: batch.size]

    # 填充批次至图形大小
    def pad_batch(self, batch: Batch) -> int:
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)
        return batch.padded_size - batch.size

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    # 销毁图形资源 
    def destroy_cuda_graphs(self) -> None:
        del self.graph_map
        gc.collect()
