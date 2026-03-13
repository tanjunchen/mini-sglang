# Mini-SGLang 架构设计

## 系统架构概览

Mini-SGLang 采用**多进程分布式架构**，各进程协同工作实现高效的 LLM 推理。

### 核心组件

```
┌─────────────────────────────────────────────────────────────┐
│                        User Request                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    API Server (FastAPI)                     │
│  - 接收 HTTP 请求                                           │
│  - 提供 OpenAI 兼容 API                                     │
│  - 管理请求生命周期和流式响应                                │
└────────────────────────┬────────────────────────────────────┘
                         │ ZeroMQ
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Tokenizer Workers                         │
│  - 将文本转换为 tokens                                       │
│  - 支持多个 tokenizer 进程以提高吞吐量                       │
└────────────────────────┬────────────────────────────────────┘
                         │ ZeroMQ
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Scheduler Worker (Rank 0)                  │
│  - 主调度器，接收新请求                                     │
│  - 广播请求到其他 ranks                                     │
│  - 收集输出 token                                            │
└─────┬───────────────────────┬───────────────────────────────┘
      │                       │
      │ NCCL                  │ NCCL
      ▼                       ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Scheduler Rank 1 │  │ Scheduler Rank 2 │  │ ... Scheduler N  │
│ - 执行推理        │  │ - 执行推理        │  │ - 执行推理         │
│ - 管理本地资源    │  │ - 管理本地资源    │  │ - 管理本地资源     │
└──────────────────┘  └──────────────────┘  └──────────────────┘
      │                       │
      └───────────┬───────────┘
                  │ NCCL (收集输出)
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   Detokenizer Worker                         │
│  - 将 tokens 转换回文本                                      │
│  - 发送结果回 API Server                                    │
└────────────────────────┬────────────────────────────────────┘
                         │ ZeroMQ
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Server                               │
│  - 流式返回结果给用户                                        │
└─────────────────────────────────────────────────────────────┘
```

### 数据流

1. **用户** → **API Server**: HTTP 请求（提示词 + 采样参数）
2. **API Server** → **Tokenizer**: ZeroMQ 消息（文本）
3. **Tokenizer** → **Scheduler (Rank 0)**: ZeroMQ 消息（input_ids + 采样参数）
4. **Scheduler (Rank 0)** → **所有 Schedulers**: NCCL 广播
5. **所有 Schedulers** → **本地 Engine**: 执行前向传播
6. **Scheduler (Rank 0)** → **Detokenizer**: ZeroMQ 消息（输出 token）
7. **Detokenizer** → **API Server**: ZeroMQ 消息（解码后的文本）
8. **API Server** → **用户**: 流式 HTTP 响应

## 模块架构

### 1. 核心数据结构 (`minisgl.core`)

```python
# 采样参数
SamplingParams
  - temperature: float        # 采样温度
  - top_k: int                 # top-k 采样
  - top_p: float               # nucleus 采样
  - max_tokens: int            # 最大生成 token 数
  - ignore_eos: bool           # 忽略 EOS token

# 单个推理请求
Req
  - uid: int                   # 唯一标识符
  - input_ids: Tensor          # 输入 token IDs (CPU)
  - sampling_params: SamplingParams
  - cached_len: int            # 已缓存的长度
  - device_len: int            # 设备上的 token 数
  - table_idx: int             # 页表索引
  - cache_handle: BaseCacheHandle  # KV 缓存句柄

# 批次请求
Batch
  - reqs: List[Req]            # 请求列表
  - phase: "prefill" | "decode"  # 当前阶段
  - input_ids: Tensor          # 拼接后的 token IDs
  - out_loc: Tensor            # 输出 KV cache 位置
  - attn_metadata: BaseAttnMetadata  # 注意力元数据

# 全局上下文
Context
  - page_size: int
  - attn_backend: BaseAttnBackend
  - batch: Batch               # 当前批次
```

### 2. 调度器 (`minisgl.scheduler`)

**核心职责**: 决策与编排，将请求流转化为 GPU 每轮执行指令

**关键决策**:
- 本轮执行 prefill 还是 decode
- 为批次分配多少 KV 页
- 请求结束后的 KV 缓存复用策略

**核心循环**:
- `overlap_loop`: 重叠调度模式（默认）
  - CPU 处理上一轮结果 + GPU 执行本轮计算
- `normal_loop`: 顺序执行模式

**管理器组件**:
- `TableManager`: 管理 token 池和页表
- `CacheManager`: 管理 KV 页分配、复用和驱逐
- `PrefillManager`: 管理 prefill 阶段的请求批次
- `DecodeManager`: 管理 decode 阶段的请求批次

### 3. 推理引擎 (`minisgl.engine`)

**Engine**: 单进程上的 TP worker

**核心职责**:
- 加载和管理模型
- 管理 KV 缓存和页表
- 执行批次前向传播
- CUDA Graph 优化

**关键流程**:
```python
def forward_batch(batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
    # 1. 检查是否可以使用 CUDA Graph
    if graph_runner.can_use_cuda_graph(batch):
        logits = graph_runner.replay(batch)
    else:
        logits = model.forward()
    
    # 2. 更新请求状态
    for req in batch.reqs:
        req.complete_one()
    
    # 3. 采样下一个 token
    next_tokens_gpu = sampler.sample(logits, args)
    
    # 4. 异步拷贝到 CPU
    next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
    copy_done_event.record()
    
    return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)
```

### 4. 注意力后端 (`minisgl.attention`)

**接口设计**:
```python
class BaseAttnBackend(ABC):
    @abstractmethod
    def forward(q, k, v, layer_id, batch) -> Tensor: ...
    
    @abstractmethod
    def prepare_metadata(batch) -> None: ...
    
    @abstractmethod
    def init_capture_graph(max_seq_len, bs_list) -> None: ...
```

**支持的实现**:
- **FlashAttention Backend**: 高性能前向注意力
- **FlashInfer Backend**: 优化的 decode 注意力
- **HybridBackend**: 混合使用不同后端（prefill 用 FA，decode 用 FI）

### 5. KV 缓存管理 (`minisgl.kvcache`)

**接口设计**:
```python
class BaseCacheManager(ABC):
    @abstractmethod
    def match_prefix(input_ids) -> Tuple[Handle, Tensor]: ...
    
    @abstractmethod
    def insert_prefix(input_ids, indices) -> int: ...
    
    @abstractmethod
    def evict(size) -> Tensor: ...
    
    @abstractmethod
    def lock_handle(handle, unlock=False) -> None: ...
```

**实现策略**:
- **RadixCacheManager**: 压缩 Trie 结构，高效匹配共享前缀
  - 每个节点存储一段 token 序列
  - LRU 驱逐策略
  - 支持节点分裂和合并
- **NaiveCacheManager**: 简单的线性缓存管理（用于对比）

**数据结构**:
```python
RadixTreeNode
  - children: Dict[int, RadixTreeNode]  # 子节点
  - _key: Tensor                         # 存储的 token 序列
  - _value: Tensor                       # 对应的 KV 页索引
  - ref_count: int                       # 引用计数
  - timestamp: int                       # LRU 时间戳
```

### 6. 模型层 (`minisgl.layers`)

**基础类**:
```python
class BaseOP(ABC):
    @abstractmethod
    def forward(*args, **kwargs) -> Any: ...
    
    def state_dict(prefix="", result=None) -> Dict: ...
    
    def load_state_dict(state_dict, prefix="", _internal=False) -> None: ...

class StateLessOP(BaseOP):
    # 无状态的算子（如 RoPE），无需加载权重

class OPList(BaseOP, Generic[T]):
    # 算子列表，用于管理多层结构
```

**主要组件**:
- **AttentionLayer**: 注意力计算（Q/K/V 投影、RoPE、注意力）
- **LinearQKVMerged**: 合并的 QKV 线性层（支持 Tensor Parallelism）
- **RMSNorm**: RMS 归一化
- **RotaryEmbedding**: 旋转位置编码
- **VocabParallelEmbedding**: 词汇表并行的 embedding 层

### 7. 消息传递 (`minisgl.message`)

**设计原则**:
- 所有消息支持自动序列化/反序列化
- 基于 `dataclass` 和类型内省
- 特别优化 `torch.Tensor` 传输

**消息类型**:
```python
# Backend ← Tokenizer
class BaseBackendMsg: ...

class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: Tensor      # CPU 1D int32 tensor
    sampling_params: SamplingParams

class BatchBackendMsg(BaseBackendMsg):
    data: List[BaseBackendMsg]  # 批量消息

# Tokenizer ← Backend
class BaseFrontendMsg: ...

class DetokenizeMsg(BaseFrontendMsg):
    uid: int
    next_token: int
    finished: bool

# Frontend → Tokenizer
class BaseTokenizerMsg: ...

class TokenizeMsg(BaseTokenizerMsg):
    uid: int
    text: str | List[dict]   # 支持聊天格式
    sampling_params: SamplingParams
```

**序列化机制**:
- 使用 `serialize_type` / `deserialize_type`
- 自动检测 `torch.Tensor` 并使用 `msgpack` 序列化
- 其他类型使用标准 JSON 序列化

### 8. 分布式通信 (`minisgl.distributed`)

**接口设计**:
```python
class DistributedImpl(ABC):
    @abstractmethod
    def all_reduce(x: Tensor) -> Tensor: ...
    
    @abstractmethod
    def all_gather(x: Tensor) -> Tensor: ...
```

**实现策略**:
- **TorchDistributedImpl**: 使用 PyTorch 原生的 `torch.distributed`
  - 后端: NCCL（GPU 间）或 Gloo（CPU 间）
- **PyNCCLDistributedImpl**: 使用自定义的 PyNCCL 绑定
  - 更低延迟，支持异构内存

**通信模式**:
- **All-Reduce**: 聚合所有 TP rank 的结果（如线性层输出）
- **All-Gather**: 收集所有 rank 的数据（如 embedding）

### 9. CUDA 内核 (`minisgl.kernel`)

**自定义内核**:
- `indexing`: KV cache 索引操作
- `store_cache`: 存储 KV 到 cache
- `radix`: 快速前缀匹配
- `tensor`: 张量操作
- `pynccl`: 基于 PyTorch FFI 的 NCCL 绑定

**编译方式**:
- 使用 TVM FFI 进行 Python 绑定
- 支持 JIT 编译
- CUDA 源码位于 `kernel/csrc/`

## 性能优化技术

### 1. 重叠调度 (Overlap Scheduling)

**原理**: CPU 调度开销与 GPU 计算并行

```python
def overlap_loop(last_data):
    # CPU: 处理上一轮结果
    _process_last_data(last_data)
    
    # GPU: 执行本轮计算（并行）
    with engine_stream_ctx:
        forward_input = _schedule_next_batch()
        if forward_input:
            ongoing_data = (forward_input, _forward(forward_input))
    
    return ongoing_data
```

**效果**: 隐藏 CPU 调度延迟，提升吞吐量

### 2. CUDA Graph

**原理**: 捕获并重放 CUDA 图，消除 Python 开销

**条件**:
- Decode 阶段
- Batch size 在预捕获范围内
- 固定的计算图结构

**实现**:
```python
if graph_runner.can_use_cuda_graph(batch):
    logits = graph_runner.replay(batch)
else:
    logits = model.forward()
```

### 3. Radix Cache

**原理**: 使用压缩 Trie 存储和重用 KV 缓存

**优势**:
- 跨请求共享前缀
- 减少重复计算
- 提升长序列场景性能

**数据结构**: 压缩 Trie（每个节点存储一段 token 序列）

### 4. Chunked Prefill

**原理**: 将长提示拆分为小块处理

**优势**:
- 降低峰值内存使用
- 避免 OOM 错误
- 支持更长上下文

**配置**: `--max-prefill-length n`

### 5. Tensor Parallelism

**原理**: 将模型张量分割到多个 GPU

**应用**:
- **QKV 投影**: 沿头维度分割
- **Output 投影**: 沿头维度分割后 All-Reduce
- **FFN**: 沿隐藏层维度分割
- **Embedding**: 词汇表并行

**通信**:
- Linear 层输出: All-Reduce
- Embedding: All-Gather

## 错误处理和资源管理

### 请求生命周期

```
创建 → Prefill → Decode → 完成 → 资源释放
  ↓
失败/超时 → 中止 → 资源释放
```

### 资源管理策略

1. **KV 页分配**:
   - 根据 `extend_len` 动态分配
   - 使用 `CacheManager` 统一管理
   - 请求完成后回收或缓存

2. **页表管理**:
   - 使用 `TableManager` 维护 token 池和页表
   - 实时更新页表映射关系
   - 支持快速索引查找

3. **内存平衡检查**:
   - 同步所有 TP rank 的可用内存
   - 检测内存不平衡（>2GB 报错）
   - 动态调整 KV 缓存大小

4. **异常处理**:
   - 键盘中断优雅退出
   - 子进程异常通知主进程
   - 资源泄漏预防

## 配置系统

### 环境变量 (`minisgl.env`)

```python
# Shell
MINISGL_SHELL_MAX_TOKENS=2048
MINISGL_SHELL_TEMPERATURE=0.6

# 后端运行时
MINISGL_DISABLE_OVERLAP_SCHEDULING=false
MINISGL_PYNCCL_MAX_BUFFER_SIZE=1G
MINISGL_OVERLAP_EXTRA_SYNC=false
```

### 命令行参数 (`minisgl.server.args`)

主要参数:
- `--model`: 模型路径
- `--tp`: Tensor Parallelism 度数
- `--max-running-req`: 最大并发请求数
- `--cache`: 缓存策略（radix/naive）
- `--attn`: 注意力后端
- `--cuda-graph-max-bs`: CUDA Graph 最大 batch size
- `--max-prefill-length`: Prefill 最大长度

## 扩展性设计

### 添加新模型

1. 继承 `BaseLLMModel`
2. 实现模型层（使用 `minisgl.layers`）
3. 配置 `RotaryConfig` 和模型参数
4. 实现 `create_model` 工厂函数

### 添加新注意力后端

1. 继承 `BaseAttnBackend`
2. 实现 `forward`, `prepare_metadata` 等方法
3. 可选: 实现 CUDA Graph 捕获接口
4. 注册到 `create_attention_backend`

### 添加新 KV 缓存策略

1. 继承 `BaseCacheManager`
2. 实现 `match_prefix`, `insert_prefix`, `evict` 等方法
3. 实现锁机制（`lock_handle`）
4. 注册到 `create_kvcache`