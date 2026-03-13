from typing import Tuple

import torch

from .base import BaseOP

"""
# RMS Normalization
"""
class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        self.rmsnorm(x, self.weight, self.eps, out=x)

"""
普通实现:   x' = LayerNorm(x + residual)
            = Normalize(x + residual)
            需要 2 次 kernel 调用（add + norm）
融合实现:   fused_add_rmsnorm(x, residual)
            = Normalize(x + residual)
            只需要 1 次 kernel 调用
性能提升:   减少内存带宽压力，提高吞吐
"""
class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        # 动态导入 flashinfer 内核
        from flashinfer import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size) # 空张量，后续加载权重
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """融合版 RMSNorm，支持残差连接"""
        if residual is None:
            # 标准 RMSNorm
            return self.rmsnorm(x, self.weight, self.eps), x
        # 融合 add + RMSNorm（一次内核调用）
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
