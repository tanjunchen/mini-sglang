# Mini-SGLang 编码规范

## 类型注解规范

### 强制类型注解

所有公共 API、函数参数和返回值都必须有类型注解。

```python
# ✅ 正确
def forward_batch(
    self, 
    batch: Batch, 
    args: BatchSamplingArgs
) -> ForwardOutput:
    ...

# ❌ 错误
def forward_batch(self, batch, args):
    ...
```

### 类型导入

使用 `from __future__ import annotations` 启用延迟注解解析，支持循环引用。

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minisgl.core import Batch
    from minisgl.engine import ForwardOutput
```

### 类型别名

对于复杂类型，使用 `TypeAlias` 定义别名提高可读性。

```python
from typing import TypeAlias

ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"
```

## 代码风格

### 基本规范

遵循项目配置的 Black 和 Ruff 规则：

```toml
[tool.black]
line-length = 100
target-version = ['py310']

[tool.ruff]
target-version = "py310"
line-length = 100
```

### 导入顺序

1. 标准库导入
2. 第三方库导入
3. 本地模块导入
4. `if TYPE_CHECKING:` 导入（避免循环依赖）

```python
from __future__ import annotations

# 标准库
import asyncio
from dataclasses import dataclass

# 第三方库
import torch
from transformers import AutoTokenizer

# 本地模块
from minisgl.core import Batch, Req

if TYPE_CHECKING:
    from minisgl.engine import ForwardOutput
```

### 命名约定

| 类型 | 约定 | 示例 |
|------|------|------|
| 类名 | PascalCase | `AttentionLayer`, `Batch` |
| 函数/方法 | snake_case | `forward_batch`, `schedule_next_batch` |
| 常量 | UPPER_SNAKE_CASE | `MINISGL_ENV_PREFIX` |
| 私有成员 | _leading_underscore | `_parent`, `_internal` |
| 类型变量 | 单个大写字母 | `T`, `K`, `V` |

## 数据类规范

### 使用 `@dataclass`

优先使用 `@dataclass` 定义数据容器类。

```python
@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    max_tokens: int = 1024
    ignore_eos: bool = False
```

### 不可变数据类

对于不变的数据结构，使用 `@dataclass(frozen=True)`。

```python
@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    cached_len: int
```

### 属性方法

对于计算属性，使用 `@property` 装饰器。

```python
@dataclass(eq=False)
class Req:
    cached_len: int
    device_len: int
    max_device_len: int

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len
```

## 抽象基类规范

### 使用 `ABC` 和 `abstractmethod`

定义接口时使用抽象基类。

```python
from abc import ABC, abstractmethod

class BaseAttnBackend(ABC):
    @abstractmethod
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        layer_id: int, 
        batch: Batch
    ) -> torch.Tensor: ...
```

### 类型提示

抽象方法的参数和返回值必须有类型提示。

## 异常处理规范

### 自定义异常

定义清晰的异常类继承标准异常。

```python
class RequestAllFinished(Exception):
    pass

class RuntimeError(Exception):
    pass
```

### 日志记录

使用结构化日志记录重要事件。

```python
from minisgl.utils import init_logger

logger = init_logger(__name__)

def forward_batch(self, batch: Batch, args: BatchSamplingArgs):
    logger.debug_rank0("Processing batch with %d requests", batch.size)
    
    try:
        result = self._forward(batch, args)
    except Exception as e:
        logger.error_rank0("Failed to process batch: %s", e)
        raise
```

### 错误消息

错误消息应包含足够的上下文信息。

```python
# ✅ 正确
raise RuntimeError(
    f"Cannot evict {size}, only {self.evictable_size} is evictable"
)

# ❌ 错误
raise RuntimeError("Cannot evict")
```

## 资源管理规范

### 上下文管理器

对于需要清理资源的对象，使用上下文管理器。

```python
@contextmanager
def forward_batch(self, batch: Batch):
    assert self._batch is None, "Nested forward_batch is not allowed"
    try:
        self._batch = batch
        yield
    finally:
        self._batch = None
```

### CUDA 流管理

正确管理 CUDA 流和同步。

```python
# 创建流
self.stream = torch.cuda.Stream(device=self.device)
torch.cuda.set_stream(self.stream)

# 流间同步
self.engine.stream.wait_stream(self.stream)

# 同步等待
self.stream.synchronize()
torch.cuda.synchronize(self.device)
```

### 内存管理

及时释放不需要的张量。

```python
# 清空缓存
torch.cuda.empty_cache()

# 重置峰值内存统计
torch.cuda.reset_peak_memory_stats(self.device)

# 删除不再需要的张量
del large_tensor
```

## 并发和异步规范

### 进程间通信

使用 ZeroMQ 进行进程间通信。

```python
from minisgl.utils import ZmqAsyncPushQueue, ZmqAsyncPullQueue

# 发送队列
send_queue = ZmqAsyncPushQueue(
    config.zmq_tokenizer_addr,
    create=True,
    encoder=BaseTokenizerMsg.encoder,
)

# 接收队列
recv_queue = ZmqAsyncPullQueue(
    config.zmq_backend_addr,
    create=False,
    decoder=BaseBackendMsg.decoder,
)

# 使用
await send_queue.put(msg)
msg = await recv_queue.get()
```

### 异步编程

使用 `asyncio` 进行异步操作。

```python
async def stream_generate(self, uid: int):
    async for ack in self.wait_for_ack(uid):
        yield f"data: {ack.incremental_output}\n".encode()
        if ack.finished:
            break
    yield "data: [DONE]\n".encode()
```

### 事件同步

使用 `asyncio.Event` 进行异步通知。

```python
self.event_map[uid] = asyncio.Event()

# 等待
await self.event_map[uid].wait()

# 触发
self.event_map[uid].set()
```

## 分布式计算规范

### 分布式初始化

正确初始化分布式环境。

```python
torch.distributed.init_process_group(
    backend="nccl",
    rank=config.tp_info.rank,
    world_size=config.tp_info.size,
    timeout=timedelta(seconds=config.distributed_timeout),
    init_method=config.distributed_addr,
)
```

### 分布式通信

使用统一的通信接口。

```python
from minisgl.distributed import DistributedCommunicator

comm = DistributedCommunicator()

# All-Reduce
output = comm.all_reduce(input_tensor)

# All-Gather
output = comm.all_gather(input_tensor)
```

### 主从同步

主节点（rank 0）负责特殊逻辑。

```python
if self.tp_info.is_primary():
    # 只在主节点执行
    ack_queue.put("Scheduler is ready")
    logger.info("Primary rank ready")

# 所有 rank 同步
self.sync_all_ranks()
```

## 性能优化规范

### CUDA Graph

对 decode 阶段使用 CUDA Graph 优化。

```python
if self.graph_runner.can_use_cuda_graph(batch):
    logits = self.graph_runner.replay(batch)
else:
    logits = self.model.forward()
```

### 异步拷贝

使用非阻塞拷贝隐藏传输延迟。

```python
# GPU → CPU 异步拷贝
next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)

# 记录事件
copy_done_event = torch.cuda.Event()
copy_done_event.record(self.stream)

# 等待完成
copy_done_event.synchronize()
```

### 内存对齐

确保张量内存对齐以提高访问效率。

```python
# 32 字节对齐（256 bit）
def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32

self.max_seq_len = _align_up_32(max_seq_len)
```

### 批处理

尽可能使用批处理提高 GPU 利用率。

```python
# 批量发送消息
class BatchBackendMsg(BaseBackendMsg):
    data: List[BaseBackendMsg]

# 批量处理请求
results = await asyncio.gather(*tasks)
```

## 测试规范

### 测试文件命名

测试文件以 `test_` 开头，位于 `tests/` 目录下。

```
tests/
├── core/
│   └── test_scheduler.py
├── kernel/
│   └── test_tensor.py
└── misc/
    └── test_serialize.py
```

### 测试函数命名

测试函数以 `test_` 开头，使用描述性名称。

```python
def test_forward_batch_with_decode(self):
    ...

def test_scheduler_overflow(self):
    ...
```

### 使用 `torch.inference_mode()`

推理测试使用 `inference_mode` 禁用梯度计算。

```python
@torch.inference_mode()
def test_model_forward(self):
    ...
```

### 使用 pytest

使用 pytest 框架和 fixtures。

```python
import pytest

@pytest.fixture
def scheduler():
    config = SchedulerConfig(...)
    return Scheduler(config)

def test_scheduler_prefill(scheduler):
    ...
```

## 文档规范

### 文档字符串

公共 API 必须有文档字符串。

```python
def forward_batch(
    self, 
    batch: Batch, 
    args: BatchSamplingArgs
) -> ForwardOutput:
    """
    Execute forward pass for a batch.
    
    Args:
        batch: The batch of requests to process.
        args: Sampling arguments for the batch.
    
    Returns:
        ForwardOutput containing next tokens and copy event.
    """
    ...
```

### 内联注释

对于复杂逻辑，添加简洁的注释。

```python
# 先通过 receive_msg 接收新请求
for msg in self.receive_msg(blocking=blocking):
    self._process_one_msg(msg)

# 调度下一个批次
forward_input = self._schedule_next_batch()
```

### 避免

- 注释显而易见的代码
- 注释被注释掉的代码（删除它）
- 过度的注释

## 配置管理规范

### 环境变量

使用 `EnvVar` 类管理环境变量。

```python
class EnvClassSingleton:
    SHELL_MAX_TOKENS = EnvInt(2048)
    DISABLE_OVERLAP_SCHEDULING = EnvBool(False)
```

### 命令行参数

使用 `argparse` 或 `dataclass` 定义参数。

```python
@dataclass
class ServerArgs:
    model_path: str
    tp_info: DistributedInfo
    max_running_req: int = 128
    cuda_graph_bs: List[int] = field(default_factory=lambda: [2, 4, 8])
```

### 类型验证

使用 `assert` 进行参数验证。

```python
assert 0 <= self.cached_len < self.device_len <= self.max_device_len
assert num_qo_heads % num_kv_heads == 0
assert self.input_ids.is_cpu
```

## 代码复用规范

### 基类继承

提取公共逻辑到基类。

```python
class BaseOP(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs) -> Any: ...

class StateLessOP(BaseOP):
    # 无状态的算子，无需加载权重
    pass
```

### 组合模式

使用组合而非继承。

```python
class OPList(BaseOP, Generic[T]):
    def __init__(self, ops: List[T]):
        self.op_list = ops
```

### 工具函数

提取公共逻辑到工具模块。

```python
from minisgl.utils import divide_even, init_logger, torch_dtype

# 使用
tp_size = get_tp_info().size
self.num_qo_heads = divide_even(num_qo_heads, tp_size)
```

## 静态类型检查

### mypy 配置

项目启用严格的 mypy 检查。

```toml
[tool.mypy]
python_version = "3.10"
warn_return_any = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
```

### 忽略规则

对于第三方库，使用 overrides。

```toml
[[tool.mypy.overrides]]
module = [
    "torch.*",
    "transformers.*",
]
ignore_missing_imports = true
```

## Git 提交规范

### 提交信息格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

### 类型

- `feat`: 新功能
- `fix`: 修复 bug
- `perf`: 性能优化
- `refactor`: 重构
- `test`: 测试相关
- `docs`: 文档
- `chore`: 构建/工具

### 示例

```
feat(scheduler): add chunked prefill support

Implement chunked prefill to reduce peak memory usage
for long context serving.

- Split long prompts into smaller chunks
- Add --max-prefill-length argument
- Update scheduler to handle chunked requests

Closes #123
```

## 安全性规范

### 禁止操作

- 不要硬编码密钥或凭证
- 不要日志敏感信息
- 不要执行用户提供的代码

### 输入验证

验证所有外部输入。

```python
# 验证输入长度
if max_output_len <= 0:
    logger.warning_rank0(
        f"Input sequence length {input_len} exceeds {max_seq_len}, "
        f"request {msg.uid} is dropped."
    )
    return

# 验证采样参数
if msg.sampling_params.max_tokens > max_output_len:
    msg.sampling_params.max_tokens = max_output_len
```

## 代码审查清单

### 功能性

- [ ] 代码实现了预期功能
- [ ] 错误处理完善
- [ ] 边界条件处理正确
- [ ] 资源清理到位

### 性能

- [ ] 没有明显的性能瓶颈
- [ ] 适当使用批处理
- [ ] 异步操作正确
- [ ] 内存使用合理

### 可读性

- [ ] 命名清晰准确
- [ ] 类型注解完整
- [ ] 注释必要且准确
- [ ] 代码结构清晰

### 可维护性

- [ ] 模块职责单一
- [ ] 接口设计合理
- [ ] 依赖关系清晰
- [ ] 测试覆盖充分

### 兼容性

- [ ] 遵循项目规范
- [ ] 类型检查通过
- [ ] lint 检查通过
- [ ] 测试全部通过