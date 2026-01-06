# DeepEP 测试与性能分析指南

本指南详细介绍如何对DeepEP进行测试和性能优化分析。

## 一、DeepEP 的三种主要场景

### 1.1 场景分类

| 场景 | 测试文件 | 应用场景 | 通信方式 | 关键特点 |
|------|---------|---------|---------|---------|
| **高吞吐量-单节点** | `tests/test_intranode.py` | 训练/预填充(单节点) | NVLink | 带宽：153-158 GB/s |
| **高吞吐量-多节点** | `tests/test_internode.py` | 训练/预填充(多节点) | NVLink + RDMA | 带宽：43-58 GB/s |
| **低延迟-推理解码** | `tests/test_low_latency.py` | 在线推理(解码阶段) | 纯RDMA | 延迟：77-369 us |

### 1.2 场景详细说明

#### 场景A：高吞吐量单节点通信 (Intranode)
**适用场景**：单机多卡训练，预填充阶段
**典型配置**：
```python
num_tokens = 4096          # 批次大小
hidden = 7168              # 隐藏层维度
num_topk = 8               # Top-K专家数
num_experts = 256          # 总专家数
num_processes = 8          # GPU数量(通过NVLink连接)
```

**测试命令**：
```bash
python tests/test_intranode.py \
    --num-processes 8 \
    --num-tokens 4096 \
    --hidden 7168 \
    --num-topk 8 \
    --num-experts 256
```

#### 场景B：高吞吐量多节点通信 (Internode)
**适用场景**：多机多卡训练，大规模预填充
**典型配置**：
```python
num_tokens = 4096
hidden = 7168
num_topk_groups = 4        # 分组Top-K（对应DeepSeek-V3的group-limited gating）
num_topk = 8
num_experts = 256
num_processes = 16+        # 跨节点GPU数量
```

**测试命令**：
```bash
# 基础测试
python tests/test_internode.py \
    --num-processes 16 \
    --num-topk-groups 4

# 压力测试模式（用于稳定性验证）
python tests/test_internode.py \
    --num-processes 16 \
    --pressure-test-mode 1

# 测试与低延迟内核的兼容性
python tests/test_internode.py \
    --test-ll-compatibility
```

#### 场景C：低延迟推理解码 (Low Latency)
**适用场景**：在线推理服务，解码阶段
**典型配置**：
```python
num_tokens = 128           # 推理批次（较小）
hidden = 7168
num_experts = 256
num_topk = 8               # 动态调整
```

**测试命令**：
```bash
# 基础低延迟测试
python tests/test_low_latency.py

# 缩小测试（模拟rank失败场景）
python tests/test_low_latency.py --shrink-test

# 使用LogFMT精度格式
python tests/test_low_latency.py --use-logfmt
```

---

## 二、性能分析工具与方法

### 2.1 内置性能测试工具

DeepEP提供了两种主要的性能分析工具，位于`tests/utils.py`：

#### 工具1: `bench()` - 基础GPU性能测试
**位置**: `tests/utils.py:100-125`

**功能**：
- L2缓存预热（256MB数据）
- 可配置预热次数（默认50次）
- 使用CUDA事件精确计时
- 返回平均/最小/最大执行时间

**使用示例**：
```python
from utils import bench

# 测试dispatch操作
avg_time, min_time, max_time = bench(
    lambda: buffer.dispatch(x, topk_idx, config),
    num_warmups=50,
    num_tests=50
)

# 计算带宽
data_size_gb = total_bytes / 1e9
bandwidth_gbps = data_size_gb / avg_time

print(f'平均时间: {avg_time * 1e6:.2f} us')
print(f'带宽: {bandwidth_gbps:.2f} GB/s')
```

#### 工具2: `bench_kineto()` - 细粒度内核分析
**位置**: `tests/utils.py:173-238`

**功能**：
- 使用PyTorch Profiler追踪CUDA活动
- 解析特定内核的执行时间
- 支持导出Chrome trace文件
- 可用于通信-计算重叠分析

**使用示例**：
```python
from utils import bench_kineto

# 分析单个内核
kernel_time = bench_kineto(
    lambda: buffer.dispatch(x, topk_idx, config),
    kernel_names='dispatch',
    num_tests=30,
    suppress_kineto_output=True
)

# 分析多个内核（例如dispatch中的notify和主体）
notify_time, main_time = bench_kineto(
    lambda: buffer.combine(x, handle, topk_weights, config),
    kernel_names=('notify', 'combine'),
    num_tests=30
)

# 导出trace文件用于可视化
bench_kineto(
    lambda: buffer.dispatch(x, topk_idx, config),
    kernel_names='dispatch',
    trace_path='./trace_dispatch.json'
)
# 使用chrome://tracing打开trace_dispatch.json查看
```

### 2.2 各模块耗时分析方法

#### 方法1：布局计算模块 (Layout)
**位置**: `tests/test_intranode.py:62-70`

```python
# 测试get_dispatch_layout的性能
layout_time = bench(
    lambda: buffer.get_dispatch_layout(topk_idx, num_experts)
)[0]

print(f'布局计算耗时: {layout_time * 1000:.3f} ms')
```

**优化目标**：
- 对于4096 tokens，目标 < 1ms
- 可通过优化topk_idx数据结构减少计算

#### 方法2：Dispatch模块
**Dispatch包含的子操作**：
1. 数据准备 (FP8转换)
2. 布局计算
3. NVLink发送
4. RDMA发送

**分阶段测试示例**（来自`test_intranode.py:194-220`）：
```python
# 1. 总体性能
total_time = bench(lambda: buffer.dispatch(
    x=x_fp8,
    topk_idx=topk_idx,
    num_experts=num_experts,
    config=config
))[0]

# 2. 使用Kineto分析细粒度
dispatch_time, notify_time = bench_kineto(
    lambda: buffer.dispatch(...),
    kernel_names=('dispatch_kernel', 'notify_kernel'),
    num_tests=30,
    suppress_kineto_output=True
)

# 3. 计算各部分占比
total_bytes = num_tokens * hidden * 2  # BF16
nvlink_bandwidth = total_bytes / 1e9 / total_time
print(f'总耗时: {total_time * 1e6:.2f} us')
print(f'Notify开销: {notify_time * 1e6:.2f} us ({notify_time/total_time*100:.1f}%)')
print(f'主体开销: {dispatch_time * 1e6:.2f} us ({dispatch_time/total_time*100:.1f}%)')
print(f'NVLink带宽: {nvlink_bandwidth:.2f} GB/s')
```

#### 方法3：Combine模块
**Combine包含的子操作**：
1. 等待dispatch完成
2. 接收数据
3. 权重聚合
4. 结果写回

**测试示例**（来自`test_intranode.py:240-260`）：
```python
# 测试combine性能
handle = buffer.dispatch(...)  # 先dispatch

combine_time = bench(lambda: buffer.combine(
    x=x,
    handle=handle,
    topk_weights=topk_weights,
    config=config
))[0]

# 细粒度分析
notify_time, combine_kernel_time = bench_kineto(
    lambda: buffer.combine(...),
    kernel_names=('notify', 'combine'),
    suppress_kineto_output=True
)

print(f'Combine总耗时: {combine_time * 1e6:.2f} us')
print(f'  - Notify: {notify_time * 1e6:.0f} us')
print(f'  - Kernel: {combine_kernel_time * 1e6:.0f} us')
```

#### 方法4：低延迟模块专项分析
**接收Hook机制测试**（用于通信-计算重叠）：
```python
# 使用接收Hook实现零SM占用的数据接收
recv_hook = buffer.low_latency_dispatch(
    x=x_fp8,
    topk_idx=topk_idx,
    use_fp8=True,
    return_recv_hook=True  # 返回Hook而不是直接等待
)

# 在计算时重叠通信
computation_result = some_heavy_computation()

# 等待接收完成
recv_hook.wait()
```

### 2.3 配置参数调优

DeepEP的性能高度依赖于Config参数，以下是调优方法：

#### Config结构体参数
```python
class Config:
    num_sms: int                           # 使用的SM数量
    num_max_nvl_chunked_send_tokens: int   # NVLink发送chunk大小
    num_max_nvl_chunked_recv_tokens: int   # NVLink接收chunk大小
    num_max_rdma_chunked_send_tokens: int  # RDMA发送chunk大小
    num_max_rdma_chunked_recv_tokens: int  # RDMA接收chunk大小
```

#### 自动调优脚本示例
**Dispatch调优** (来自`test_intranode.py:194-220`):
```python
import deep_ep

num_sms = 24  # H800通常为24或32
best_bandwidth = 0
best_config = None

# 遍历NVLink chunk大小
for nvl_chunk_size in range(4, 33, 2):
    config = deep_ep.Config(
        num_sms=num_sms,
        num_max_nvl_chunked_send_tokens=nvl_chunk_size,
        num_max_nvl_chunked_recv_tokens=256,  # 固定接收buffer
        num_max_rdma_chunked_send_tokens=0,
        num_max_rdma_chunked_recv_tokens=0
    )

    # 测试性能
    time = bench(lambda: buffer.dispatch(
        x, topk_idx, num_experts, config
    ))[0]

    bandwidth = data_bytes / 1e9 / time
    print(f'NVL chunk {nvl_chunk_size}: {bandwidth:.2f} GB/s, {time*1e6:.2f} us')

    if bandwidth > best_bandwidth:
        best_bandwidth = bandwidth
        best_config = config

print(f'\n最优配置: chunk={best_config.num_max_nvl_chunked_send_tokens}, '
      f'带宽={best_bandwidth:.2f} GB/s')
```

**多节点RDMA调优** (来自`test_internode.py:242-264`):
```python
# 2D网格搜索
for nvl_chunk in range(4, 45, 4):
    for rdma_chunk in range(4, 33, 4):
        config = deep_ep.Config(
            num_sms=num_sms,
            num_max_nvl_chunked_send_tokens=nvl_chunk,
            num_max_nvl_chunked_recv_tokens=288,
            num_max_rdma_chunked_send_tokens=rdma_chunk,
            num_max_rdma_chunked_recv_tokens=128
        )

        # 使用Kineto避免CPU启动开销
        notify_time, dispatch_time = bench_kineto(
            lambda: buffer.dispatch(...),
            kernel_names=('notify', 'dispatch'),
            suppress_kineto_output=True
        )

        total_time = notify_time + dispatch_time
        bandwidth = data_bytes / 1e9 / total_time

        print(f'NVL={nvl_chunk}, RDMA={rdma_chunk}: '
              f'{bandwidth:.2f} GB/s ({total_time*1e6:.0f} us)')
```

### 2.4 数据精度对比测试

**FP8 vs BF16性能对比**：
```python
from utils import per_token_cast_to_fp8, per_token_cast_back

# BF16基准
x_bf16 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
time_bf16 = bench(lambda: buffer.dispatch(x_bf16, topk_idx, config))[0]

# FP8测试
x_fp8, x_scales = per_token_cast_to_fp8(x_bf16)
time_fp8 = bench(lambda: buffer.dispatch((x_fp8, x_scales), topk_idx, config))[0]

print(f'BF16耗时: {time_bf16 * 1e6:.2f} us')
print(f'FP8耗时:  {time_fp8 * 1e6:.2f} us')
print(f'加速比:   {time_bf16 / time_fp8:.2f}x')
```

---

## 三、完整的性能分析工作流

### 3.1 步骤1：环境准备
```bash
# 1. 编译DeepEP
NVSHMEM_DIR=/path/to/nvshmem python setup.py build
ln -s build/lib.linux-x86_64-cpython-38/deep_ep_cpp.*.so

# 2. 检查NVLink拓扑
nvidia-smi topo -m

# 3. 检查RDMA网卡
ibstat
ibv_devinfo
```

### 3.2 步骤2：基准测试
```bash
# 单节点基准
python tests/test_intranode.py --num-processes 8

# 多节点基准（需要修改tests/utils.py中的init_dist）
python tests/test_internode.py --num-processes 16

# 低延迟基准
python tests/test_low_latency.py
```

### 3.3 步骤3：性能剖析

创建自定义性能分析脚本 `analyze_performance.py`:

```python
#!/usr/bin/env python3
import torch
import torch.distributed as dist
import deep_ep
from tests.utils import init_dist, bench, bench_kineto
import argparse
import json

def profile_dispatch(buffer, x, topk_idx, config, num_experts):
    """分析dispatch各个阶段的耗时"""
    results = {}

    # 1. 布局计算
    layout_time = bench(
        lambda: buffer.get_dispatch_layout(topk_idx, num_experts),
        num_tests=30
    )[0]
    results['layout_ms'] = layout_time * 1000

    # 2. 总体dispatch
    total_time = bench(
        lambda: buffer.dispatch(x, topk_idx, num_experts, config),
        num_tests=30
    )[0]
    results['total_us'] = total_time * 1e6

    # 3. 细粒度内核分析
    try:
        kernel_times = bench_kineto(
            lambda: buffer.dispatch(x, topk_idx, num_experts, config),
            kernel_names='dispatch',
            num_tests=20,
            suppress_kineto_output=True,
            trace_path='dispatch_trace.json'
        )
        results['kernel_us'] = kernel_times * 1e6
    except Exception as e:
        print(f'Kineto分析失败: {e}')

    # 4. 计算带宽
    num_tokens, hidden = x.shape if not isinstance(x, tuple) else x[0].shape
    bytes_per_element = 1 if isinstance(x, tuple) else 2  # FP8 vs BF16
    total_bytes = num_tokens * hidden * bytes_per_element
    results['bandwidth_gbps'] = (total_bytes / 1e9) / total_time

    return results

def profile_combine(buffer, x, handle, topk_weights, config):
    """分析combine各个阶段的耗时"""
    results = {}

    # 总体combine
    total_time = bench(
        lambda: buffer.combine(x, handle, topk_weights, config),
        num_tests=30
    )[0]
    results['total_us'] = total_time * 1e6

    # 细粒度分析
    try:
        notify_time, combine_time = bench_kineto(
            lambda: buffer.combine(x, handle, topk_weights, config),
            kernel_names=('notify', 'combine'),
            num_tests=20,
            suppress_kineto_output=True
        )
        results['notify_us'] = notify_time * 1e6
        results['combine_kernel_us'] = combine_time * 1e6
        results['notify_overhead_pct'] = (notify_time / total_time) * 100
    except Exception as e:
        print(f'Kineto分析失败: {e}')

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-tokens', type=int, default=4096)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--output', type=str, default='performance_report.json')
    args = parser.parse_args()

    # 初始化分布式
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank, world_size, group = init_dist(local_rank, args.num_processes)

    # 创建测试数据
    x = torch.randn((args.num_tokens, args.hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((args.num_tokens, 256), dtype=torch.float32, device='cuda')
    topk_idx = torch.topk(scores, 8, dim=-1)[1].to(deep_ep.topk_idx_t)

    # 创建buffer
    buffer = deep_ep.Buffer(group, num_nvl_bytes=1<<30, num_rdma_bytes=1<<28)
    config = buffer.get_dispatch_config(world_size)

    # 性能分析
    report = {
        'config': {
            'num_processes': args.num_processes,
            'num_tokens': args.num_tokens,
            'hidden': args.hidden,
        },
        'dispatch': profile_dispatch(buffer, x, topk_idx, config, 256),
    }

    # 保存报告
    if rank == 0:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f'\n性能报告已保存到 {args.output}')
        print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
```

运行性能分析：
```bash
python analyze_performance.py --num-processes 8 --output perf_report.json
```

### 3.4 步骤4：可视化分析

查看Chrome trace:
1. 运行带有`trace_path`参数的`bench_kineto`
2. 在Chrome浏览器打开 `chrome://tracing`
3. 加载生成的JSON文件
4. 分析GPU kernel执行时序

### 3.5 步骤5：优化迭代

根据性能报告识别瓶颈：

| 瓶颈现象 | 可能原因 | 优化方向 |
|---------|---------|---------|
| NVLink带宽 < 100 GB/s | chunk大小不合适 | 调整`num_max_nvl_chunked_*_tokens` |
| RDMA带宽 < 30 GB/s | 网络配置问题 | 检查IB配置、virtual lane设置 |
| Notify开销 > 20% | 同步开销大 | 考虑使用异步模式、调整SM数量 |
| Layout计算 > 2ms | 数据结构低效 | 优化topk_idx生成算法 |

---

## 四、常见性能优化技巧

### 4.1 SM数量调整
```python
# 减少SM占用以避免与计算kernel冲突
deep_ep.Buffer.set_num_sms(16)  # 默认24

# 测试不同SM数量
for num_sms in [8, 16, 24, 32]:
    deep_ep.Buffer.set_num_sms(num_sms)
    time = bench(lambda: buffer.dispatch(...))
    print(f'SM={num_sms}: {time*1e6:.2f} us')
```

### 4.2 异步模式
```python
# 同步模式（默认）
buffer.dispatch(x, topk_idx, config, async_finish=False)

# 异步模式（允许CPU-GPU重叠）
handle = buffer.dispatch(x, topk_idx, config, async_finish=True)
# ... 可以在这里做其他工作 ...
buffer.combine(x, handle, topk_weights, config)  # 等待完成
```

### 4.3 数据预处理优化
```python
# 预先计算并缓存layout
num_tokens_per_rank, _, _, is_token_in_rank, _ = \
    buffer.get_dispatch_layout(topk_idx, num_experts)

# 后续dispatch重用布局信息（避免重复计算）
buffer.dispatch(
    x,
    num_tokens_per_rank=num_tokens_per_rank,
    is_token_in_rank=is_token_in_rank,
    config=config
)
```

### 4.4 网络层优化
```bash
# 设置NVSHMEM环境变量
export NVSHMEM_SYMMETRIC_SIZE=1073741824  # 1GB对称堆
export NVSHMEM_IB_SL=2                     # Virtual Lane 2
export NVSHMEM_DISABLE_CUDA_VMM=1          # 禁用虚拟内存管理（某些情况下更快）

# InfiniBand性能调优
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
```

---

## 五、故障排查与验证

### 5.1 数据正确性验证
```python
from utils import calc_diff, hash_tensor

# 余弦相似度检查
diff = calc_diff(output, expected_output)
assert diff < 1e-3, f"输出差异过大: {diff}"

# Hash一致性检查（多节点）
local_hash = hash_tensor(output)
all_hashes = [torch.tensor(0, dtype=torch.int, device='cuda') for _ in range(world_size)]
dist.all_gather(all_hashes, torch.tensor(local_hash, dtype=torch.int, device='cuda'))
assert len(set([h.item() for h in all_hashes])) == 1, "不同rank输出不一致"
```

### 5.2 压力测试
```bash
# 运行1000次迭代的稳定性测试
python tests/test_internode.py --pressure-test-mode 2
```

### 5.3 缩小测试（模拟节点故障）
```bash
# 测试rank动态失败场景
python tests/test_low_latency.py --shrink-test
```

---

## 六、性能报告模板

记录性能测试结果时，建议包含以下信息：

```markdown
## 性能测试报告

### 测试环境
- GPU型号: H800 / A100
- GPU数量: 8 / 16 / 32
- NVLink带宽: 900 GB/s (NVLink 4.0)
- RDMA网卡: CX7 400 Gb/s
- CUDA版本: 12.3
- PyTorch版本: 2.1.0
- NVSHMEM版本: 2.11.0

### 测试配置
- num_tokens: 4096
- hidden: 7168
- num_topk: 8
- num_experts: 256
- 数据格式: FP8 / BF16

### 性能数据

#### Dispatch性能
- 总耗时: XXX us
- 带宽: XXX GB/s
- 布局计算: XXX ms
- Notify开销: XXX us (XX%)
- 主内核: XXX us

#### Combine性能
- 总耗时: XXX us
- 带宽: XXX GB/s
- Notify开销: XXX us (XX%)
- 聚合内核: XXX us

### 优化建议
1. ...
2. ...
```

---

## 七、参考资源

- **主README**: `/home/user/DeepEP/README.md` - 快速开始和使用示例
- **测试工具**: `/home/user/DeepEP/tests/utils.py` - 性能测试函数库
- **配置定义**: `/home/user/DeepEP/csrc/config.hpp` - Config结构体说明
- **NVSHMEM安装**: `/home/user/DeepEP/third-party/README.md` - 依赖安装指南

---

## 八、快速参考命令

```bash
# 编译（开发模式）
NVSHMEM_DIR=/path/to/nvshmem python setup.py build

# 单节点测试
python tests/test_intranode.py --num-processes 8

# 多节点测试
MASTER_ADDR=node1 MASTER_PORT=8361 WORLD_SIZE=2 RANK=0 \
python tests/test_internode.py --num-processes 16

# 低延迟测试
python tests/test_low_latency.py --shrink-test

# 导出性能trace
python -c "
from tests.utils import bench_kineto
import deep_ep
# ... setup code ...
bench_kineto(
    lambda: buffer.dispatch(...),
    'dispatch',
    trace_path='trace.json'
)
"

# 查看trace
google-chrome chrome://tracing
```
