from __future__ import annotations

from typing import TYPE_CHECKING

from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVMerged,
    LinearRowParallel,
    RMSNorm,
    silu_and_mul,
)
from minisgl.models import ModelConfig
from minisgl.utils import nvtx_annotate

if TYPE_CHECKING:
    import torch


class GatedMLP(BaseOP):
    def __init__(self, config: ModelConfig):
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            has_bias=False,
        )

        match config.hidden_act:
            case "silu":
                self.act_fn = silu_and_mul
            case act_fn:
                raise ValueError(f"Unsupported activation function: {act_fn}")

        self.down_proj = LinearRowParallel(
            config.intermediate_size,
            config.hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)
        del x
        y = self.act_fn(gate_up)
        del gate_up
        return self.down_proj.forward(y)


"""
融合 QKV 投影：使用 LinearQKVMerged 一次性计算 Q、K、V，减少内存访问
注意力层封装：AttentionLayer 封装 RoPE 和注意力后端调用
全局上下文：通过 get_global_ctx() 获取位置信息，简化接口
"""

class RopeAttn(BaseOP):
    """初始化带有旋转位置编码的注意力层(RoPE Attention)。

    初始化注意力层的各个组件，包括QKV投影、归一化层(可选)和注意力计算层。

    Args:
        config: 模型配置对象，包含隐藏层大小、头数等参数
        layer_id: 当前层的ID标识，用于区分不同层的配置
        has_attn_bias: 是否在注意力计算中使用偏置项，默认为False
        has_qk_norm: 是否对查询(Q)和键(K)进行归一化处理，默认为False
    """
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        has_attn_bias: bool = False,
        has_qk_norm: bool = False,    # Qwen3 等模型需要设置为 True
    ):
        head_dim = config.head_dim
        # 融合 QKV 投影（提高效率）
        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=has_attn_bias,
        )
        # 可选的 Q/K 归一化
        self.has_qk_norm = has_qk_norm
        if has_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        # 注意力层封装
        self.attn = AttentionLayer(
            layer_id=layer_id,
            head_dim=head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_config=config.rotary_config,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )

        # 输出投影
        self.o_proj = LinearOProj(
            head_dim * config.num_qo_heads,
            config.hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播（使用全局上下文获取位置信息）"""
        qkv = self.qkv_proj.forward(x)
        del x # 内存优化：立即释放输入张量
        o = self.attn.forward(qkv) # 内部处理 RoPE 和注意力计算
        return self.o_proj.forward(o)


__all__ = ["GatedMLP", "RopeAttn"]
