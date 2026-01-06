#!/usr/bin/env python3
"""
DeepEP性能分析工具

用法:
    # 单节点分析
    python tools/analyze_performance.py --mode intranode --num-processes 8

    # 多节点分析（需要多机环境）
    python tools/analyze_performance.py --mode internode --num-processes 16

    # 低延迟分析
    python tools/analyze_performance.py --mode low-latency --num-processes 8

    # 导出详细trace
    python tools/analyze_performance.py --export-trace --trace-dir ./traces
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.distributed as dist

# 添加tests目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'tests'))

import deep_ep
from utils import init_dist, bench, bench_kineto, per_token_cast_to_fp8, calc_diff


class PerformanceAnalyzer:
    """DeepEP性能分析器"""

    def __init__(self, rank: int, world_size: int, group: dist.ProcessGroup,
                 num_tokens: int, hidden: int, num_experts: int, num_topk: int):
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.num_tokens = num_tokens
        self.hidden = hidden
        self.num_experts = num_experts
        self.num_topk = num_topk

        # 创建测试数据
        self.x_bf16 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        self.scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
        self.topk_idx = torch.topk(self.scores, num_topk, dim=-1, largest=True, sorted=False)[1]
        self.topk_idx = self.topk_idx.to(deep_ep.topk_idx_t)
        self.topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')

        # FP8数据（如果支持）
        self.x_fp8 = None
        if deep_ep.Buffer.is_sm90_compiled():
            fp8_data, fp8_scales = per_token_cast_to_fp8(self.x_bf16)
            self.x_fp8 = (fp8_data, fp8_scales.T.contiguous().T)

        self.buffer = None

    def log(self, message: str):
        """只在rank 0打印日志"""
        if self.rank == 0:
            print(message, flush=True)

    def create_buffer(self, num_nvl_bytes: int = 1<<30, num_rdma_bytes: int = 1<<28) -> deep_ep.Buffer:
        """创建通信buffer"""
        self.buffer = deep_ep.Buffer(self.group, num_nvl_bytes, num_rdma_bytes)
        return self.buffer

    def analyze_layout(self) -> Dict:
        """分析布局计算性能"""
        self.log("\n" + "="*60)
        self.log("分析模块: 布局计算 (Layout)")
        self.log("="*60)

        results = {}

        # 预热
        for _ in range(10):
            self.buffer.get_dispatch_layout(self.topk_idx, self.num_experts)

        # 性能测试
        avg_time, min_time, max_time = bench(
            lambda: self.buffer.get_dispatch_layout(self.topk_idx, self.num_experts),
            num_warmups=50,
            num_tests=100
        )

        results['avg_ms'] = avg_time * 1000
        results['min_ms'] = min_time * 1000
        results['max_ms'] = max_time * 1000

        self.log(f"布局计算耗时:")
        self.log(f"  平均: {avg_time * 1000:.3f} ms")
        self.log(f"  最小: {min_time * 1000:.3f} ms")
        self.log(f"  最大: {max_time * 1000:.3f} ms")

        return results

    def analyze_dispatch(self, config: deep_ep.Config, use_fp8: bool = False,
                         export_trace: bool = False, trace_path: Optional[str] = None) -> Dict:
        """分析Dispatch性能"""
        self.log("\n" + "="*60)
        self.log(f"分析模块: Dispatch ({'FP8' if use_fp8 else 'BF16'})")
        self.log("="*60)

        results = {'dtype': 'fp8' if use_fp8 else 'bf16'}
        x = self.x_fp8 if use_fp8 and self.x_fp8 is not None else self.x_bf16

        # 计算数据量
        bytes_per_element = 1 if use_fp8 else 2
        total_bytes = self.num_tokens * self.hidden * bytes_per_element
        results['total_bytes'] = total_bytes

        # 1. 总体性能
        avg_time, min_time, max_time = bench(
            lambda: self.buffer.dispatch(x, self.topk_idx, self.num_experts, config),
            num_warmups=50,
            num_tests=50
        )

        results['total_avg_us'] = avg_time * 1e6
        results['total_min_us'] = min_time * 1e6
        results['total_max_us'] = max_time * 1e6
        results['bandwidth_gbps'] = (total_bytes / 1e9) / avg_time

        self.log(f"Dispatch总体性能:")
        self.log(f"  平均时间: {avg_time * 1e6:.2f} us")
        self.log(f"  最小时间: {min_time * 1e6:.2f} us")
        self.log(f"  最大时间: {max_time * 1e6:.2f} us")
        self.log(f"  有效带宽: {results['bandwidth_gbps']:.2f} GB/s")

        # 2. Kineto细粒度分析
        try:
            kernel_time = bench_kineto(
                lambda: self.buffer.dispatch(x, self.topk_idx, self.num_experts, config),
                kernel_names='dispatch',
                num_tests=30,
                suppress_kineto_output=True,
                trace_path=trace_path if export_trace else None
            )
            results['kernel_us'] = kernel_time * 1e6
            self.log(f"\nKineto内核分析:")
            self.log(f"  Dispatch内核: {kernel_time * 1e6:.2f} us")

            if export_trace and trace_path:
                self.log(f"  Trace已导出: {trace_path}")
        except Exception as e:
            self.log(f"\nKineto分析失败: {e}")

        return results

    def analyze_combine(self, config: deep_ep.Config, use_fp8: bool = False,
                       export_trace: bool = False, trace_path: Optional[str] = None) -> Dict:
        """分析Combine性能"""
        self.log("\n" + "="*60)
        self.log(f"分析模块: Combine ({'FP8' if use_fp8 else 'BF16'})")
        self.log("="*60)

        results = {'dtype': 'fp8' if use_fp8 else 'bf16'}
        x = self.x_fp8 if use_fp8 and self.x_fp8 is not None else self.x_bf16

        # 先dispatch
        handle = self.buffer.dispatch(x, self.topk_idx, self.num_experts, config)

        # 1. 总体性能
        avg_time, min_time, max_time = bench(
            lambda: self.buffer.combine(
                x,
                self.buffer.dispatch(x, self.topk_idx, self.num_experts, config),
                self.topk_weights,
                config
            ),
            num_warmups=50,
            num_tests=50
        )

        results['total_avg_us'] = avg_time * 1e6
        results['total_min_us'] = min_time * 1e6
        results['total_max_us'] = max_time * 1e6

        bytes_per_element = 1 if use_fp8 else 2
        total_bytes = self.num_tokens * self.hidden * bytes_per_element
        results['bandwidth_gbps'] = (total_bytes / 1e9) / avg_time

        self.log(f"Combine总体性能:")
        self.log(f"  平均时间: {avg_time * 1e6:.2f} us")
        self.log(f"  最小时间: {min_time * 1e6:.2f} us")
        self.log(f"  最大时间: {max_time * 1e6:.2f} us")
        self.log(f"  有效带宽: {results['bandwidth_gbps']:.2f} GB/s")

        # 2. Kineto细粒度分析
        try:
            notify_time, combine_time = bench_kineto(
                lambda: self.buffer.combine(
                    x,
                    self.buffer.dispatch(x, self.topk_idx, self.num_experts, config),
                    self.topk_weights,
                    config
                ),
                kernel_names=('notify', 'combine'),
                num_tests=30,
                suppress_kineto_output=True,
                trace_path=trace_path if export_trace else None
            )
            results['notify_us'] = notify_time * 1e6
            results['combine_kernel_us'] = combine_time * 1e6
            results['notify_overhead_pct'] = (notify_time / avg_time) * 100

            self.log(f"\nKineto内核分析:")
            self.log(f"  Notify: {notify_time * 1e6:.2f} us ({results['notify_overhead_pct']:.1f}%)")
            self.log(f"  Combine内核: {combine_time * 1e6:.2f} us")

            if export_trace and trace_path:
                self.log(f"  Trace已导出: {trace_path}")
        except Exception as e:
            self.log(f"\nKineto分析失败: {e}")

        return results

    def tune_config(self, mode: str = 'dispatch') -> Tuple[deep_ep.Config, Dict]:
        """配置参数自动调优"""
        self.log("\n" + "="*60)
        self.log(f"配置调优: {mode.upper()}")
        self.log("="*60)

        num_sms = 24  # H800默认值
        best_bandwidth = 0
        best_config = None
        tuning_results = []

        if mode == 'dispatch':
            # Dispatch调优：主要调整NVLink chunk大小
            nvl_chunk_sizes = list(range(4, 33, 2)) + [0]  # 4-32步长2，加上0
            for nvl_chunk in nvl_chunk_sizes:
                config = deep_ep.Config(
                    num_sms=num_sms,
                    num_max_nvl_chunked_send_tokens=nvl_chunk,
                    num_max_nvl_chunked_recv_tokens=256,
                    num_max_rdma_chunked_send_tokens=0,
                    num_max_rdma_chunked_recv_tokens=0
                )

                try:
                    avg_time = bench(
                        lambda: self.buffer.dispatch(self.x_bf16, self.topk_idx, self.num_experts, config),
                        num_warmups=20,
                        num_tests=30
                    )[0]

                    bandwidth = (self.num_tokens * self.hidden * 2 / 1e9) / avg_time
                    tuning_results.append({
                        'nvl_chunk': nvl_chunk,
                        'time_us': avg_time * 1e6,
                        'bandwidth_gbps': bandwidth
                    })

                    self.log(f"NVL chunk {nvl_chunk:3d}: {bandwidth:6.2f} GB/s, {avg_time*1e6:7.2f} us")

                    if bandwidth > best_bandwidth:
                        best_bandwidth = bandwidth
                        best_config = config
                except Exception as e:
                    self.log(f"NVL chunk {nvl_chunk:3d}: 失败 - {e}")

        elif mode == 'combine':
            # Combine调优：chunk大小通常更小
            nvl_chunk_sizes = range(1, 17, 1)
            for nvl_chunk in nvl_chunk_sizes:
                config = deep_ep.Config(
                    num_sms=num_sms,
                    num_max_nvl_chunked_send_tokens=nvl_chunk,
                    num_max_nvl_chunked_recv_tokens=256,
                    num_max_rdma_chunked_send_tokens=0,
                    num_max_rdma_chunked_recv_tokens=0
                )

                try:
                    avg_time = bench(
                        lambda: self.buffer.combine(
                            self.x_bf16,
                            self.buffer.dispatch(self.x_bf16, self.topk_idx, self.num_experts, config),
                            self.topk_weights,
                            config
                        ),
                        num_warmups=20,
                        num_tests=30
                    )[0]

                    bandwidth = (self.num_tokens * self.hidden * 2 / 1e9) / avg_time
                    tuning_results.append({
                        'nvl_chunk': nvl_chunk,
                        'time_us': avg_time * 1e6,
                        'bandwidth_gbps': bandwidth
                    })

                    self.log(f"NVL chunk {nvl_chunk:3d}: {bandwidth:6.2f} GB/s, {avg_time*1e6:7.2f} us")

                    if bandwidth > best_bandwidth:
                        best_bandwidth = bandwidth
                        best_config = config
                except Exception as e:
                    self.log(f"NVL chunk {nvl_chunk:3d}: 失败 - {e}")

        self.log(f"\n最优配置:")
        self.log(f"  带宽: {best_bandwidth:.2f} GB/s")
        if best_config:
            self.log(f"  NVL send chunk: {best_config.num_max_nvl_chunked_send_tokens}")
            self.log(f"  NVL recv chunk: {best_config.num_max_nvl_chunked_recv_tokens}")

        return best_config, {'results': tuning_results, 'best_bandwidth': best_bandwidth}


def main():
    parser = argparse.ArgumentParser(description='DeepEP性能分析工具')

    # 基础参数
    parser.add_argument('--mode', type=str, default='intranode',
                       choices=['intranode', 'internode', 'low-latency'],
                       help='测试模式')
    parser.add_argument('--num-processes', type=int, default=8,
                       help='进程数（GPU数）')

    # 模型参数
    parser.add_argument('--num-tokens', type=int, default=4096,
                       help='Token数量')
    parser.add_argument('--hidden', type=int, default=7168,
                       help='隐藏层维度')
    parser.add_argument('--num-experts', type=int, default=256,
                       help='专家总数')
    parser.add_argument('--num-topk', type=int, default=8,
                       help='Top-K专家数')

    # 分析选项
    parser.add_argument('--analyze-layout', action='store_true',
                       help='分析布局计算性能')
    parser.add_argument('--analyze-dispatch', action='store_true',
                       help='分析Dispatch性能')
    parser.add_argument('--analyze-combine', action='store_true',
                       help='分析Combine性能')
    parser.add_argument('--tune-config', action='store_true',
                       help='自动调优配置参数')
    parser.add_argument('--test-fp8', action='store_true',
                       help='测试FP8性能')

    # 输出选项
    parser.add_argument('--export-trace', action='store_true',
                       help='导出Chrome trace文件')
    parser.add_argument('--trace-dir', type=str, default='./traces',
                       help='Trace文件保存目录')
    parser.add_argument('--output', type=str, default='performance_report.json',
                       help='性能报告输出文件')

    args = parser.parse_args()

    # 如果没有指定任何分析选项，默认全部分析
    if not any([args.analyze_layout, args.analyze_dispatch, args.analyze_combine, args.tune_config]):
        args.analyze_layout = True
        args.analyze_dispatch = True
        args.analyze_combine = True

    # 初始化分布式
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank, world_size, group = init_dist(local_rank, args.num_processes)

    # 创建trace目录
    if args.export_trace and rank == 0:
        Path(args.trace_dir).mkdir(parents=True, exist_ok=True)

    # 创建分析器
    analyzer = PerformanceAnalyzer(
        rank=rank,
        world_size=world_size,
        group=group,
        num_tokens=args.num_tokens,
        hidden=args.hidden,
        num_experts=args.num_experts,
        num_topk=args.num_topk
    )

    # 创建buffer
    analyzer.log(f"\n{'='*60}")
    analyzer.log(f"DeepEP性能分析工具")
    analyzer.log(f"{'='*60}")
    analyzer.log(f"模式: {args.mode}")
    analyzer.log(f"进程数: {args.num_processes}")
    analyzer.log(f"配置: tokens={args.num_tokens}, hidden={args.hidden}, "
                f"experts={args.num_experts}, topk={args.num_topk}")

    buffer = analyzer.create_buffer()

    # 获取或调优配置
    if args.tune_config:
        dispatch_config, dispatch_tuning = analyzer.tune_config('dispatch')
        combine_config, combine_tuning = analyzer.tune_config('combine')
    else:
        dispatch_config = buffer.get_dispatch_config(world_size)
        combine_config = buffer.get_combine_config(world_size)
        dispatch_tuning = None
        combine_tuning = None

    # 性能报告
    report = {
        'metadata': {
            'mode': args.mode,
            'num_processes': args.num_processes,
            'num_tokens': args.num_tokens,
            'hidden': args.hidden,
            'num_experts': args.num_experts,
            'num_topk': args.num_topk,
            'gpu_name': torch.cuda.get_device_name(0),
            'cuda_version': torch.version.cuda,
            'pytorch_version': torch.__version__,
        }
    }

    # 执行分析
    if args.analyze_layout:
        report['layout'] = analyzer.analyze_layout()

    if args.analyze_dispatch:
        trace_path = f"{args.trace_dir}/dispatch_bf16.json" if args.export_trace else None
        report['dispatch_bf16'] = analyzer.analyze_dispatch(
            dispatch_config, use_fp8=False, export_trace=args.export_trace, trace_path=trace_path
        )

        if args.test_fp8 and analyzer.x_fp8 is not None:
            trace_path = f"{args.trace_dir}/dispatch_fp8.json" if args.export_trace else None
            report['dispatch_fp8'] = analyzer.analyze_dispatch(
                dispatch_config, use_fp8=True, export_trace=args.export_trace, trace_path=trace_path
            )

    if args.analyze_combine:
        trace_path = f"{args.trace_dir}/combine_bf16.json" if args.export_trace else None
        report['combine_bf16'] = analyzer.analyze_combine(
            combine_config, use_fp8=False, export_trace=args.export_trace, trace_path=trace_path
        )

        if args.test_fp8 and analyzer.x_fp8 is not None:
            trace_path = f"{args.trace_dir}/combine_fp8.json" if args.export_trace else None
            report['combine_fp8'] = analyzer.analyze_combine(
                combine_config, use_fp8=True, export_trace=args.export_trace, trace_path=trace_path
            )

    if args.tune_config:
        report['tuning'] = {
            'dispatch': dispatch_tuning,
            'combine': combine_tuning
        }

    # 保存报告
    if rank == 0:
        analyzer.log(f"\n{'='*60}")
        analyzer.log("性能分析完成")
        analyzer.log(f"{'='*60}")

        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)

        analyzer.log(f"\n性能报告已保存到: {args.output}")

        # 打印摘要
        analyzer.log(f"\n性能摘要:")
        if 'layout' in report:
            analyzer.log(f"  布局计算: {report['layout']['avg_ms']:.3f} ms")
        if 'dispatch_bf16' in report:
            analyzer.log(f"  Dispatch (BF16): {report['dispatch_bf16']['total_avg_us']:.2f} us, "
                        f"{report['dispatch_bf16']['bandwidth_gbps']:.2f} GB/s")
        if 'dispatch_fp8' in report:
            analyzer.log(f"  Dispatch (FP8):  {report['dispatch_fp8']['total_avg_us']:.2f} us, "
                        f"{report['dispatch_fp8']['bandwidth_gbps']:.2f} GB/s")
        if 'combine_bf16' in report:
            analyzer.log(f"  Combine (BF16):  {report['combine_bf16']['total_avg_us']:.2f} us, "
                        f"{report['combine_bf16']['bandwidth_gbps']:.2f} GB/s")
        if 'combine_fp8' in report:
            analyzer.log(f"  Combine (FP8):   {report['combine_fp8']['total_avg_us']:.2f} us, "
                        f"{report['combine_fp8']['bandwidth_gbps']:.2f} GB/s")

    # 清理
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
