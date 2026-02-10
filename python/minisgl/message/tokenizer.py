from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseTokenizerMsg:
    @staticmethod
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTokenizerMsg:
        return deserialize_type(globals(), json)

@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    data: List[BaseTokenizerMsg]


# DetokenizeMsg(回程) ：这是推理结果的中间态。
@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    uid: int
    next_token: int  # 只有一个整数 ID
    finished: bool

# TokenizeMsg(去程) ：这是整个推理链路的起点，也是最“重”的消息。
@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    uid: int
    text: str | List[Dict[str, str]]  # 原始文本或对话历史
    sampling_params: SamplingParams   # 采样参数


@dataclass
class AbortMsg(BaseTokenizerMsg):
    uid: int
