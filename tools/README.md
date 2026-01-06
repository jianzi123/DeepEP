# DeepEP性能分析工具

## 快速开始

### 1. 编译DeepEP
```bash
cd /home/user/DeepEP
NVSHMEM_DIR=/path/to/nvshmem python setup.py build
ln -s build/lib.linux-x86_64-cpython-*/deep_ep_cpp.*.so
```

### 2. 运行性能分析

#### 单节点分析（8 GPU）
```bash
python tools/analyze_performance.py \
    --mode intranode \
    --num-processes 8 \
    --num-tokens 4096 \
    --hidden 7168 \
    --output perf_intranode.json
```

#### 多节点分析（需要配置多机环境）
```bash
# 在节点0上
MASTER_ADDR=node0 MASTER_PORT=8361 WORLD_SIZE=2 RANK=0 \
python tools/analyze_performance.py \
    --mode internode \
    --num-processes 16

# 在节点1上
MASTER_ADDR=node0 MASTER_PORT=8361 WORLD_SIZE=2 RANK=1 \
python tools/analyze_performance.py \
    --mode internode \
    --num-processes 16
```

#### 配置参数自动调优
```bash
python tools/analyze_performance.py \
    --mode intranode \
    --num-processes 8 \
    --tune-config \
    --output tuning_results.json
```

#### 导出Chrome trace用于可视化
```bash
python tools/analyze_performance.py \
    --export-trace \
    --trace-dir ./traces \
    --analyze-dispatch \
    --analyze-combine

# 然后在Chrome浏览器中打开 chrome://tracing
# 加载生成的JSON文件进行可视化分析
```

#### 测试FP8性能
```bash
python tools/analyze_performance.py \
    --test-fp8 \
    --output perf_fp8_comparison.json
```

## 参数说明

### 基础参数
- `--mode`: 测试模式 (`intranode`, `internode`, `low-latency`)
- `--num-processes`: GPU进程数
- `--num-tokens`: Token批次大小
- `--hidden`: 隐藏层维度
- `--num-experts`: 专家总数
- `--num-topk`: Top-K专家数

### 分析选项
- `--analyze-layout`: 分析布局计算性能
- `--analyze-dispatch`: 分析Dispatch性能
- `--analyze-combine`: 分析Combine性能
- `--tune-config`: 自动调优配置参数
- `--test-fp8`: 测试FP8精度性能

### 输出选项
- `--export-trace`: 导出Chrome trace文件
- `--trace-dir`: Trace文件保存目录
- `--output`: 性能报告JSON输出文件

## 性能报告示例

运行后会生成JSON格式的性能报告：

```json
{
  "metadata": {
    "mode": "intranode",
    "num_processes": 8,
    "num_tokens": 4096,
    "hidden": 7168,
    "gpu_name": "NVIDIA H800"
  },
  "layout": {
    "avg_ms": 0.523,
    "min_ms": 0.512,
    "max_ms": 0.548
  },
  "dispatch_bf16": {
    "total_avg_us": 385.23,
    "bandwidth_gbps": 152.4
  },
  "combine_bf16": {
    "total_avg_us": 371.18,
    "bandwidth_gbps": 158.2,
    "notify_us": 45.2,
    "notify_overhead_pct": 12.2
  }
}
```

## 查看详细指南

完整的测试和性能分析指南请参考：
- [TESTING_AND_BENCHMARKING_GUIDE.md](../TESTING_AND_BENCHMARKING_GUIDE.md)

## 故障排查

### 问题1: "No module named 'deep_ep'"
**解决**: 确保已编译并创建符号链接
```bash
NVSHMEM_DIR=/path/to/nvshmem python setup.py build
ln -s build/lib.*/deep_ep_cpp.*.so
```

### 问题2: "NCCL initialization failed"
**解决**: 检查分布式环境变量
```bash
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8361
export WORLD_SIZE=1
export RANK=0
```

### 问题3: Kineto分析失败
**解决**: 确保PyTorch版本 >= 2.1，且CUDA可用
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 进一步优化

根据性能报告结果，可以：

1. **调整chunk大小**: 使用`--tune-config`自动寻找最优配置
2. **调整SM数量**: 在代码中设置`Buffer.set_num_sms(N)`
3. **网络优化**: 配置NVSHMEM环境变量和InfiniBand参数
4. **精度优化**: 对比BF16和FP8性能，选择合适的精度

详细优化方法见完整指南。
