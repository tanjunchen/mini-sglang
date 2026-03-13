# Mini-SGLang 项目概览

## 项目定位

Mini-SGLang 是一个**轻量级但高性能**的大语言模型推理框架，是 SGLang 的精简实现。整个代码库约 5000 行 Python 代码，旨在提供一个既能作为高效推理引擎，又能作为研究和开发者参考的透明实现。

## 核心目标

1. **高性能**: 通过 Radix Cache、Chunked Prefill、Overlap Scheduling、Tensor Parallelism 等先进优化技术，实现最先进的吞吐量和延迟
2. **轻量级且可读性强**: 清洁、模块化、完全类型注解的代码库，易于理解和修改
3. **教育价值**: 作为 LLM 推理系统的透明参考实现

## 技术栈

### 核心依赖
- **PyTorch**: 深度学习框架
- **FlashAttention / FlashInfer**: 高性能注意力计算内核
- **sgl_kernel**: JIT 编译的 CUDA 内核
- **transformers**: Hugging Face 模型支持
- **ZeroMQ**: 进程间通信
- **FastAPI**: Web API 服务器
- **NCCL / PyNCCL**: 分布式 GPU 通信

### 平台支持
- **仅支持 Linux** (x86_64 和 aarch64)
- 需要 NVIDIA CUDA Toolkit
- Windows 用户需要通过 WSL2

## 代码规模

- Python 代码: ~5000 行
- CUDA 代码: 自定义内核（index, store, radix, tensor, pynccl）
- 测试代码: 核心模块测试 + 内核测试

## 项目结构

```
mini-sglang/
├── python/minisgl/          # 主代码包
│   ├── attention/           # 注意力后端实现
│   ├── benchmark/           # 性能测试工具
│   ├── distributed/         # 分布式通信
│   ├── engine/              # 推理引擎
│   ├── kernel/              # CUDA 内核
│   ├── kvcache/             # KV 缓存管理
│   ├── layers/              # 模型层实现
│   ├── llm/                 # 高级 LLM 接口
│   ├── message/             # 进程间消息定义
│   ├── models/              # 模型实现（Llama, Qwen3）
│   ├── scheduler/           # 调度器
│   ├── server/              # 服务器启动和 API
│   ├── tokenizer/           # Tokenizer/分词器
│   └── utils/               # 工具函数
├── tests/                   # 测试代码
├── benchmark/               # 基准测试脚本
└── docs/                    # 文档
```

## 核心特性

1. **Radix Cache**: 跨请求重用 KV 缓存的共享前缀
2. **Chunked Prefill**: 将长提示拆分为小块，降低峰值内存
3. **Overlap Scheduling**: 重叠 CPU 调度开销与 GPU 计算
4. **Tensor Parallelism**: 跨多个 GPU 扩展推理
5. **CUDA Graph**: 最小化 CPU 启动开销
6. **OpenAI 兼容 API**: 标准的 `/v1/chat/completions` 端点
7. **多种注意力后端**: FlashAttention, FlashInfer 等

## 支持的模型

- **Llama-3 系列**
- **Qwen-3 系列**

## 系统架构概览

Mini-SGLang 采用多进程分布式架构：

1. **API Server**: 前端入口，提供 OpenAI 兼容 API
2. **Tokenizer Workers**: 将文本转换为 tokens
3. **Detokenizer Worker**: 将 tokens 转换回文本
4. **Scheduler Workers**: 核心调度器（每个 GPU 一个），管理计算和资源分配

组件间使用 ZeroMQ 传递控制消息，使用 NCCL 传输张量数据。

## 性能特点

- **持续批处理**: 高效利用 GPU
- **流水线重叠**: CPU 调度与 GPU 计算并行
- **内存优化**: Radix Cache 实现前缀复用
- **低延迟**: CUDA Graph 消除 Python 开销

## 开发规范

- **类型注解**: 所有函数都有完整的类型提示
- **代码风格**: 遵循 Black (100 字符行长度) 和 Ruff
- **测试覆盖**: 核心模块有单元测试
- **文档完善**: 架构文档和特性说明

## 使用场景

1. **生产部署**: 高吞吐量、低延迟的 LLM 服务
2. **研究学习**: 理解现代 LLM 推理系统
3. **二次开发**: 基于精简代码定制优化
4. **教学演示**: 清晰展示推理系统设计