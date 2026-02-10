from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


"""
所有的消息基类（BaseTokenizerMsg, BaseBackendMsg, BaseFrontendMsg）都实现了 encoder 和 decoder 方法。
由于 Python 的 multiprocessing.Queue 或 ZMQ 传输的是字节流，对象必须被序列化。
mini-sglang 没有使用默认的 pickle（虽然通用但有时不安全或效率低），而是实现了一套基于类型内省的 serialize_type。
这套机制特别针对 torch.Tensor 做了优化，确保 UserMsg 中的 Tensor 能在进程间正确、高效地重建。
"""

@dataclass
class BaseBackendMsg:
    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    data: List[BaseBackendMsg]
# 这是一个典型的组合模式 (Composite Pattern) 。它允许系统将多个 UserMsg 打包成一个大包发送。
# 对于 Scheduler 来说，一次性接收 100 个请求比接收 100 次单个请求要高效得多，这减少了 Python 解释器的开销和 ZMQ 的系统调用次数。

@dataclass
class ExitMsg(BaseBackendMsg):
    pass

# 这一层定义了 Tokenizer 与 Scheduler (Backend) 之间的交互格式。
# 关键词：Tensor
@dataclass
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams

# 数据质变：对比 TokenizeMsg，这里的 text 消失了，取而代之的是 input_ids。这标志着数据从“人类可读”转变为“机器可算”。

# CPU Tensor：注释明确指出这是 CPU 1D int32 tensor。
# 这是一个关键的性能细节——ZMQ 在进程间传输 GPU Tensor 非常复杂且低效（涉及 CUDA IPC）。
# 因此，mini-sglang 选择在 CPU 端完成传输，Scheduler 收到后再统一搬运到 GPU (Host-to-Device)。
