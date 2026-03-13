# Mini-SGLang 开发工作流

## 环境设置

### 1. 克隆项目

```bash
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang
```

### 2. 创建虚拟环境

推荐使用 `uv` 进行快速安装：

```bash
# Python 3.10+ 推荐
uv venv --python=3.12
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
# 安装项目及开发依赖
uv pip install -e ".[dev]"

# 或使用 pip
pip install -e ".[dev]"
```

### 4. 验证安装

```bash
# 检查 Python 版本
python --version  # 应该 >= 3.10

# 检查 CUDA
nvidia-smi

# 运行测试
pytest tests/ -v
```

## 开发流程

### 1. 创建功能分支

```bash
git checkout -b feature/your-feature-name
# 或
git checkout -b fix/issue-number-description
```

### 2. 代码开发

#### 添加新功能

1. **设计阶段**
   - 阅读相关架构文档
   - 理解现有设计模式
   - 规划接口和数据结构

2. **实现阶段**
   - 先写测试用例
   - 编写实现代码
   - 确保类型注解完整

3. **验证阶段**
   - 运行单元测试
   - 运行集成测试
   - 性能基准测试

#### 修复 Bug

1. **复现问题**
   - 创建最小复现脚本
   - 添加失败的测试用例

2. **定位问题**
   - 使用调试工具
   - 分析日志输出
   - 理解代码执行路径

3. **修复问题**
   - 应用最小化修复
   - 更新测试用例
   - 验证修复效果

### 3. 代码质量检查

#### 格式化代码

```bash
# Black 格式化
black python/ tests/

# 检查格式化
black --check python/ tests/
```

#### Lint 检查

```bash
# Ruff lint
ruff check python/ tests/

# 自动修复
ruff check --fix python/ tests/
```

#### 类型检查

```bash
# MyPy 类型检查
mypy python/

# 检查特定模块
mypy python/minisgl/scheduler/
```

### 4. 运行测试

#### 单元测试

```bash
# 运行所有测试
pytest

# 运行特定模块
pytest tests/core/test_scheduler.py

# 运行特定测试
pytest tests/core/test_scheduler.py::test_forward_batch

# 显示详细输出
pytest -v

# 显示 print 输出
pytest -s
```

#### 覆盖率测试

```bash
# 生成覆盖率报告
pytest --cov=minisgl --cov-report=html

# 查看报告
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

#### 性能测试

```bash
# 运行离线基准测试
python benchmark/offline/bench.py --model Qwen/Qwen3-0.6B

# 运行在线基准测试
python benchmark/online/bench_qwen.py --model Qwen/Qwen3-32B
```

### 5. 本地验证

#### 启动服务器

```bash
# 单 GPU
python -m minisgl --model "Qwen/Qwen3-0.6B"

# 多 GPU (TP=4)
python -m minisgl --model "Qwen/Qwen3-32B" --tp 4

# 使用 dummy weights
python -m minisgl --model "meta-llama/Llama-3.1-70B-Instruct" --use-dummy-weight
```

#### 交互式 Shell

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

#### API 测试

```bash
# 使用 curl
curl http://localhost:8000/v1/models

# 使用 OpenAI 客户端
python -c "
import openai
client = openai.OpenAI(base_url='http://localhost:8000/v1', api_key='empty')
resp = client.chat.completions.create(
    model='Qwen/Qwen3-0.6B',
    messages=[{'role': 'user', 'content': 'Hello!'}],
    max_tokens=100
)
print(resp.choices[0].message.content)
"
```

### 6. 提交更改

#### 查看更改

```bash
# 查看状态
git status

# 查看差异
git diff

# 暂存更改
git add <files>

# 或暂存所有
git add -A
```

#### 提交信息

遵循规范格式：

```
<type>(<scope>): <subject>

<body>

<footer>
```

示例：

```
feat(scheduler): add support for dynamic batching

Implement dynamic batching to improve GPU utilization
for variable-length requests.

- Add DynamicBatchManager class
- Modify scheduler to use dynamic batching
- Add configuration option --dynamic-batching
- Update tests for new batching behavior

Closes #456
```

提交：

```bash
git commit -m "feat(scheduler): add support for dynamic batching"
```

### 7. 推送和创建 PR

```bash
# 推送到远程
git push origin feature/your-feature-name

# 或首次推送
git push -u origin feature/your-feature-name
```

然后在 GitHub 上创建 Pull Request。

## 常见开发任务

### 添加新模型

1. **创建模型文件**

```bash
touch python/minisgl/models/your_model.py
```

2. **实现模型类**

```python
from minisgl.models.base import BaseLLMModel
from minisgl.layers import *

class YourModel(BaseLLMModel):
    def __init__(self, config: YourModelConfig):
        super().__init__()
        # 实现模型层
        self.layers = OPList([
            TransformerLayer(config) 
            for _ in range(config.num_layers)
        ])
    
    def forward(self) -> torch.Tensor:
        # 实现前向传播
        ...
```

3. **注册模型工厂**

```python
# python/minisgl/models/__init__.py
def create_model(model_path: str, config: ModelConfig) -> BaseLLMModel:
    if config.model_type == "llama":
        return LlamaModel(config)
    elif config.model_type == "your_model":
        return YourModel(config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
```

4. **添加测试**

```python
# tests/models/test_your_model.py
import pytest
import torch
from minisgl.models import YourModel, YourModelConfig

def test_your_model_forward():
    config = YourModelConfig(...)
    model = YourModel(config)
    output = model.forward()
    assert output.shape == (1, vocab_size)
```

### 添加新注意力后端

1. **创建后端文件**

```bash
touch python/minisgl/attention/your_backend.py
```

2. **实现后端类**

```python
from minisgl.attention.base import BaseAttnBackend, BaseAttnMetadata

class YourAttnBackend(BaseAttnBackend):
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        layer_id: int, 
        batch: Batch
    ) -> torch.Tensor:
        # 实现注意力计算
        ...
    
    def prepare_metadata(self, batch: Batch) -> None:
        # 准备元数据
        ...
    
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        # 初始化 CUDA Graph 捕获
        ...
```

3. **注册后端**

```python
# python/minisgl/attention/__init__.py
def create_attention_backend(
    backend_name: str,
    config: ModelConfig,
    kv_cache: BaseKVCache,
    page_table: torch.Tensor
) -> BaseAttnBackend:
    if backend_name == "fa":
        return FlashAttnBackend(...)
    elif backend_name == "your_backend":
        return YourAttnBackend(...)
    ...
```

### 添加新的 KV 缓存策略

1. **创建管理器文件**

```bash
touch python/minisgl/kvcache/your_manager.py
```

2. **实现管理器类**

```python
from minisgl.kvcache.base import BaseCacheManager, BaseCacheHandle

class YourCacheManager(BaseCacheManager):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        # 初始化数据结构
    
    def match_prefix(
        self, 
        input_ids: torch.Tensor
    ) -> Tuple[BaseCacheHandle, torch.Tensor]:
        # 实现前缀匹配
        ...
    
    def insert_prefix(
        self, 
        input_ids: torch.Tensor, 
        indices: torch.Tensor
    ) -> int:
        # 插入前缀
        ...
    
    def evict(self, size: int) -> torch.Tensor:
        # 驱逐缓存
        ...
```

3. **注册管理器**

```python
# python/minisgl/kvcache/__init__.py
def create_kvcache(
    model_config: ModelConfig,
    num_pages: int,
    device: torch.device,
    dtype: torch.dtype
) -> BaseKVCache:
    if cache_type == "radix":
        manager = RadixCacheManager(device)
    elif cache_type == "your_strategy":
        manager = YourCacheManager(device)
    ...
```

### 调试技巧

#### 启用详细日志

```bash
# 设置环境变量
export MINISGL_LOG_LEVEL=DEBUG

# 或在代码中
from minisgl.utils import init_logger
logger = init_logger(__name__, "DEBUG")
```

#### 使用断点

```python
# 在代码中添加断点
import pdb; pdb.set_trace()

# 或使用 ipdb（需要安装）
import ipdb; ipdb.set_trace()
```

#### GPU 调试

```bash
# 启用 CUDA 同步以定位错误
export CUDA_LAUNCH_BLOCKING=1

# 检查内存使用
nvidia-smi

# Python 中检查
import torch
print(torch.cuda.memory_summary())
```

#### 性能分析

```bash
# 使用 cProfile
python -m cProfile -o profile.stats -m minisgl --model ...

# 分析结果
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"
```

## 常见问题解决

### CUDA 相关问题

#### CUDA 版本不匹配

```bash
# 检查驱动版本
nvidia-smi

# 检查 PyTorch CUDA 版本
python -c "import torch; print(torch.version.cuda)"

# 重新安装匹配的 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### OOM (Out of Memory)

```bash
# 减小 batch size
python -m minisgl --model ... --max-running-req 64

# 减小 max_seq_len
python -m minisgl --model ... --max-seq-len 4096

# 使用更小的模型
python -m minisgl --model "Qwen/Qwen3-0.6B"
```

### 构建问题

#### 内核编译失败

```bash
# 清理构建缓存
rm -rf ~/.cache/tvm/

# 重新安装
pip install --force-reinstall apache-tvm-ffi

# 检查 CUDA Toolkit
nvcc --version
```

#### 依赖冲突

```bash
# 创建新环境
uv venv --python=3.12
source .venv/bin/activate

# 重新安装
uv pip install -e ".[dev]"
```

### 运行时问题

#### 端口占用

```bash
# 检查端口占用
lsof -i :8000

# 杀死进程
kill -9 <PID>

# 或使用不同端口
python -m minisgl --model ... --port 8001
```

#### 分布式通信问题

```bash
# 检查 NCCL
python -c "import torch; torch.distributed.is_nccl_available()"

# 设置 NCCL 调试
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

## 性能优化指南

### 定位瓶颈

1. **使用性能分析工具**

```python
import time
import torch

start = time.time()
# 你的代码
elapsed = time.time() - start
print(f"Time: {elapsed:.4f}s")
```

2. **使用 PyTorch Profiler**

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    with_stack=True,
) as prof:
    # 你的代码
    
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

3. **检查 GPU 利用率**

```bash
# 监控 GPU
watch -n 1 nvidia-smi

# 检查 SM 利用率
nvidia-smi dmon -s u
```

### 优化策略

1. **减少内存拷贝**
   - 使用 `non_blocking=True` 进行异步拷贝
   - 避免不必要的数据转换

2. **提高 GPU 利用率**
   - 增加批处理大小
   - 使用持续批处理
   - 重叠 CPU 和 GPU 操作

3. **优化内核**
   - 使用优化的注意力后端
   - 启用 CUDA Graph
   - 使用融合内核

4. **内存优化**
   - 使用 Radix Cache
   - 合理设置 KV 缓存大小
   - 及时释放不需要的张量

## 发布流程

### 版本管理

1. **更新版本号**

```toml
# pyproject.toml
[project]
version = "0.2.0"
```

2. **更新 CHANGELOG**

```markdown
# Changelog

## [0.2.0] - 2024-01-15

### Added
- Support for dynamic batching
- New attention backend

### Fixed
- Memory leak in scheduler

### Changed
- Improved Radix Cache performance
```

3. **创建标签**

```bash
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0
```

### 发布包

```bash
# 构建发布包
python -m build

# 发布到 PyPI
twine upload dist/*
```

## 贡献指南

### 报告问题

1. 搜索现有 issues
2. 创建新 issue，包含：
   - 清晰的标题
   - 详细的问题描述
   - 复现步骤
   - 预期行为
   - 环境信息

### 提交 PR

1. 确保代码通过所有检查
2. 添加适当的测试
3. 更新文档
4. 描述清楚变更内容
5. 关联相关 issues

### 代码审查

1. 尊重审查意见
2. 及时响应评论
3. 保持代码风格一致
4. 确保测试通过

## 资源链接

### 官方文档
- [SGLang 主项目](https://github.com/sgl-project/sglang)
- [Mini-SGLang 文档](./README.md)
- [架构设计](./structures.md)
- [特性说明](./features.md)

### 相关工具
- [PyTorch 文档](https://pytorch.org/docs/stable/)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)

### 社区
- [GitHub Issues](https://github.com/sgl-project/mini-sglang/issues)
- [GitHub Discussions](https://github.com/sgl-project/mini-sglang/discussions)

## 最佳实践总结

### DO (应该做的)

- ✅ 始终编写测试
- ✅ 使用类型注解
- ✅ 保持代码简洁
- ✅ 遵循项目规范
- ✅ 更新相关文档
- ✅ 进行代码审查
- ✅ 性能测试

### DON'T (不应该做的)

- ❌ 硬编码配置
- ❌ 忽略类型检查
- ❌ 提交测试失败的代码
- ❌ 过度优化
- ❌ 忽略错误处理
- ❌ 破坏向后兼容性

### 持续改进

- 定期重构代码
- 关注性能指标
- 倾听用户反馈
- 学习新技术
- 参与社区讨论