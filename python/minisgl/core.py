from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal

import torch

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from minisgl.kvcache import BaseCacheHandle

"""
SamplingParams 类表示采样参数，包含 max_tokens（最大生成 token 数）、
temperature（采样温度，0.0表示贪心，大于0.0表示随机）、
top_p（nucleus 采样参数）、
top_k（top-k 采样参数）、
ignore_eos（是否忽略 EOS token）
以及停止条件（stop 停止词列表、
stop_token_ids 停止 token IDs 列表）。
"""
@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0

"""
Req 类表示一个推理请求，包含基本信息（uid 唯一标识符、input_ids 输入 token IDs 在 CPU 上、sampling_params 采样参数）、
资源信息（table_idx 页表索引、cache_handle KV 缓存句柄）和状态信息（cached_len 已缓存的长度、output_len 需要生成的长度）。
它提供了多个属性方法，包括 device_len（设备上的 token 数量，等于已缓存加已处理）、extend_len（需要扩展的长度，即本次前向传播处理的 token 数）
和 remain_len（剩余需要生成的长度）。
还提供了 complete_one 方法标记一个 token 已完成，以及 append_host 方法追加新生成的 token。
"""
@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor  # cpu tensor
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, next_token: torch.Tensor) -> None:
        self.input_ids = torch.cat([self.input_ids, next_token])

    def can_decode(self) -> bool:
        return self.remain_len > 0

    def __repr__(self) -> str:
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )

"""
Batch 类表示一组请求的批次，包含 reqs（请求列表）、input_ids（批次的 token IDs 拼接后）、
out_loc（输出 KV cache 位置）和 attn_metadata（注意力元数据）。
它提供了多个属性方法，包括 size（批次大小，即实际请求数）、padded_reqs（包含 padding 的请求列表，用于 CUDA Graph）和 
has_prefill（是否包含 Prefill 请求）。
"""
@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    # these fields should be set by scheduler
    input_ids: torch.Tensor = field(init=False)
    out_loc: torch.Tensor = field(init=False)
    padded_reqs: List[Req] = field(init=False)  # may contain some dummy reqs for padding
    # this field should be set by attention backend
    attn_metadata: BaseAttnMetadata = field(init=False)

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)

"""
Context 类表示全局推理上下文，包含资源信息（page_size 页大小、kv_cache KV 缓存池、attn_backend 注意力后端、page_table 页表）
和当前批次状态（batch 当前批次）。
它提供了 forward_batch 上下文管理器方法设置当前批次。
"""
@dataclass
class Context:
    page_size: int
    attn_backend: BaseAttnBackend
    _batch: Batch | None = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
