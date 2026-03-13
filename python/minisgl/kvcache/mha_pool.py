from __future__ import annotations

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import divide_even

from .base import BaseKVCache, KVCacheLayout

"""
KV 缓存系统负责存储和管理所有请求的 Key-Value 缓存，支持高效的缓存复用。
"""
class MHAKVCache(BaseKVCache):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.
    """

    """多头注意力 KV 缓存池"""
    def __init__(
        self,
        num_kv_heads: int,  # KV 头总数（模型定义）
        num_layers: int, # 层数
        head_dim: int, # 每头维度
        num_pages: int,  # 页面数
        dtype: torch.dtype, # 数据类型（如 torch.float16）
        kv_layout: KVCacheLayout, # 存储布局
        device: torch.device,  # 设备（GPU）
    ):
        # 在张量并行环境下，每个 GPU 只负责一部分 KV 头
        tp_info = get_tp_info()
        
        # 在张量并行（Tensor Parallelism）环境下，KV 头会被均匀分配到多个 GPU 上。
        # 每个 GPU 只负责存储和处理 local_kv_heads = num_kv_heads / tp_size 个 KV 头。
        # get_tp_info() 获取当前进程的张量并行信息，divide_even 确保均匀分配。
        local_kv_heads = divide_even(num_kv_heads, tp_info.size)
        
        match kv_layout:
            # 为每一层创建 KV 缓存
            case KVCacheLayout.PageFirst:
                kv_buffer = torch.empty(
                    (2, num_pages, num_layers, local_kv_heads, head_dim),
                    device=device,
                    dtype=dtype,
                ).permute(0, 2, 1, 3, 4)
            case KVCacheLayout.LayerFirst:
                kv_buffer = torch.empty(
                    (2, num_layers, num_pages, local_kv_heads, head_dim),
                    device=device,
                    dtype=dtype,
                )
            case _:
                raise ValueError(f"Unsupported kv_layout: {kv_layout}")
        self._kv_buffer = kv_buffer.view(2, num_layers, num_pages, 1, local_kv_heads, head_dim)
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        self._storage_shape = (num_pages, local_kv_heads, head_dim)

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        from minisgl.kernel import store_cache

        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
