from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.kvcache import BaseCacheHandle, create_cache_manager

if TYPE_CHECKING:
    from .utils import PendingReq

"""
CacheManager与RadixCacheManager的组合设计，核心是拆分“KV缓存物理资源管理”与“前缀复用逻辑策略”。
对Scheduler来说，它只需要关心“是否复用前缀、需要多少页、结束后是否缓存”，无需直接操作Radix树的遍历、分裂、LRU驱逐等细节。

CacheManager作为统一入口，向上层暴露三个核心接口：match_req（前缀匹配）、allocate（页分配）、
free_and_cache_finished_req（资源释放与缓存回填）；
向下则将前缀复用的具体逻辑委托给RadixCacheManager，自身仅维护一个朴素的物理空闲页池_free_slots。
"""
class CacheManager:
    def __init__(self, device: torch.device, num_pages: int, type: str):
        # TODO: support page_size > 1
        self._free_slots = torch.arange(num_pages, dtype=torch.int32, device=device)
        self.device = device
        self.manager = create_cache_manager(device=device, type=type)
        self.num_pages = num_pages

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self._free_slots = torch.cat([self._free_slots, indices])

    """
    接收请求的token序列，调用RadixCacheManager的match_prefix方法，返回命中句柄（含命中长度）和对应物理页索引。
    上层只需将页索引写入page_table，就能直接复用已有KV数据。
    """
    def match_req(self, req: PendingReq):
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self.manager.match_prefix(req.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        return self.manager.size_info.evictable_size + len(self._free_slots)

    def lock(self, handle: BaseCacheHandle) -> None:
        self.manager.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.manager.lock_handle(handle, unlock=True)

    """
    页分配（allocate） ：为新增token分配物理页。
    优先从_free_slots取空闲页，不足时触发RadixCacheManager的驱逐逻辑（按LRU和引用计数规则），回收足够页数后再分配。
    """
    def allocate(self, needed_len: int) -> torch.Tensor:
        if needed_len <= (free_len := len(self._free_slots)):
            allocated = self._free_slots[:needed_len]
            self._free_slots = self._free_slots[needed_len:]
            return allocated

        # NOTE: len(evicted) + free_len >= needed_len
        evicted = self.manager.evict(needed_len - free_len)
        merged = torch.cat([self._free_slots, evicted])
        assert len(merged) >= needed_len, "Eviction did not free enough space."

        allocated = merged[:needed_len]
        self._free_slots = merged[needed_len:]
        return allocated

    """
    资源释放与缓存回填（free_and_cache_finished_req） ：请求结束后，并非直接释放所有KV页，
    而是将其前缀写入Radix树，转化为可复用资源。再回收重叠部分的页到空闲池，解锁旧句柄允许后续驱逐。
    """
    def free_and_cache_finished_req(
        self,
        old_handle: BaseCacheHandle,
        input_ids: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        in_cache_len = self.manager.insert_prefix(input_ids, indices)
        self._free(indices[old_handle.cached_len : in_cache_len])
        self.unlock(old_handle)

    def check_integrity(self) -> None:
        self.manager.check_integrity()
        if len(self._free_slots) + self.manager.size_info.total_size != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_slots({len(self._free_slots)}) +"
                f" total_size({self.manager.size_info.total_size}) != num_pages({self.num_pages})"
            )
