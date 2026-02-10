from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .utils import deserialize_type, serialize_type


@dataclass
class BaseFrontendMsg:
    @staticmethod
    def encoder(msg: BaseFrontendMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseFrontendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchFrontendMsg(BaseFrontendMsg):
    data: List[BaseFrontendMsg]


@dataclass
class UserReply(BaseFrontendMsg):
    uid: int
    incremental_output: str
    finished: bool

# 这一层定义了 Tokenizer 处理完结果后，发回给 API Server 的格式。
# 增量输出：incremental_output 字段表明这是流式生成的一部分。它通常只包含 1 个 token 对应的字符（例如 "lo" 或 " world"）。
# API Server 收到这个对象后，会直接将其内容写入 HTTP SSE (Server-Sent Events) 流，推送到用户的浏览器。
