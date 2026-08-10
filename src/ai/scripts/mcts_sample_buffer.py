"""MCTS C++ 自对弈样本缓冲区脚本

使用 C++ MCTS 引擎让单个 cnn_v11 模型自对弈，持续产出训练样本并以
滑动窗口缓冲区（FIFO，最大 8 万条）管理落盘，供下游训练器消费。

所有对局在 C++ 层面用多线程并行运行，共享全局 MCTS 线程池和 GPU 推理引擎。
每步产生 8 条 D4 增强变体，样本格式与现有 mcts_self_play.py 完全一致。

用法：
    python src/ai/scripts/mcts_sample_buffer.py --games 1000
    python src/ai/scripts/mcts_sample_buffer.py --model cnn_v11.1 --games 1000
    python src/ai/scripts/mcts_sample_buffer.py --model cnn_v11.1 --games 500 --workers 4 --simulations 400

--model 为可选参数。不指定时自动识别 cnn_v11 的最高代数。

设计文档：design/49/main_docs/overview.md
"""

import argparse
import os
import signal
import sys
import time
from typing import List

# 确保项目根、src 和 cpp\python 在 sys.path 中
# 脚本路径: src/ai/scripts/mcts_sample_buffer.py，4 层 dirname 到项目根
_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_src_dir = os.path.join(_proj_root, 'src')
_cpp_py_dir = os.path.join(_proj_root, 'cpp', 'python')
for _p in (_proj_root, _src_dir, _cpp_py_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 将 torch\lib 加入 PATH（必须在导入 cpp_mcts 之前，否则无法加载 CUDA DLL）
import torch  # noqa: E402
_torch_lib_dir = os.path.join(os.path.dirname(os.path.dirname(torch.__file__)), 'lib')
if _torch_lib_dir not in os.environ.get('PATH', ''):
    os.environ['PATH'] = _torch_lib_dir + os.pathsep + os.environ.get('PATH', '')

from ai.scripts.mcts_config import MctsSelfPlayConfig
from cpp.python.cpp_sample_buffer_adapter import CppMctsSampleBufferAdapter
from config.mcts import MCTS_VIRTUAL_LOSS


# ---------- 中断处理 ----------


def _signal_handler(signum, frame):
    """SIGINT 信号处理器 — 直接终止进程"""
    print("\n[中断] 收到 Ctrl+C，立即终止进程")
    os._exit(0)


# ---------- 参数解析 ----------

def parse_args(argv: List[str] | None = None):
    """解析 CLI 参数

    Args:
        argv: 参数列表，None 时使用 sys.argv

    Returns:
        解析后的 argparse.Namespace
    """
    p = argparse.ArgumentParser(
        description="MCTS C++ 自对弈样本缓冲区脚本 — 产出训练样本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default=None,
                   help="模型 registry spec（如 cnn_v11.1）。不指定时自动识别 cnn_v11 最高代数")
    p.add_argument("--games", type=int, required=True,
                   help="自对弈总局数（≥1）")
    p.add_argument("--workers", type=int, default=0,
                   help="游戏工作线程数（0=每局一个线程，默认 0）")
    p.add_argument("--simulations", type=int, default=800,
                   help="每步 MCTS 模拟次数（默认 800）")
    p.add_argument("--c-puct", type=float, default=2.0,
                   help="PUCT 探索系数（默认 2.0）")
    p.add_argument("--temp-steps", type=int, default=8,
                   help="温度探索步数（默认 8）")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="温度参数（默认 1.0）")
    p.add_argument("--dirichlet-epsilon", type=float, default=0.25,
                   help="狄利克雷噪声比例（默认 0.25）")
    p.add_argument("--dirichlet-alpha", type=float, default=0.03,
                   help="狄利克雷分布参数（默认 0.03）")
    p.add_argument("--virtual-loss", type=float, default=None,
                   help="virtual loss 值（默认读取 MCTS_VIRTUAL_LOSS 配置）")
    p.add_argument("--buffer-max-size", type=int, default=80000,
                   help="缓冲区最大样本数（默认 80000）")
    p.add_argument("--chunk-games", type=int, default=10,
                   help="每 chunk 对局数（默认 10）")
    p.add_argument("--output-dir", type=str, default="sample_buffer",
                   help="缓冲区目录（默认 sample_buffer）")
    return p.parse_args(argv)


# ---------- 配置构建 ----------

def _build_config(args) -> MctsSelfPlayConfig:
    """从 CLI args 构建 MctsSelfPlayConfig

    Args:
        args: 解析后的 CLI 参数

    Returns:
        MctsSelfPlayConfig 实例
    """
    virtual_loss = args.virtual_loss if args.virtual_loss is not None else MCTS_VIRTUAL_LOSS

    return MctsSelfPlayConfig(
        simulations=args.simulations,
        c_puct=args.c_puct,
        temp_steps=args.temp_steps,
        temperature=args.temperature,
        dirichlet_epsilon=args.dirichlet_epsilon,
        dirichlet_alpha=args.dirichlet_alpha,
        virtual_loss=virtual_loss,
        use_prior=True,
    )


# ---------- 输出函数 ----------

def _print_statistics(adapter: CppMctsSampleBufferAdapter, elapsed: float) -> None:
    """打印对弈统计信息

    Args:
        adapter: 适配器实例
        elapsed: 总耗时（秒）
    """
    summary = adapter.get_buffer_summary()

    # 计算胜负统计
    try:
        results = adapter._runner.get_results()
        total = len(results)
        black_wins = sum(1 for r in results if r.winner == 1)
        white_wins = sum(1 for r in results if r.winner == 2)
        draws = total - black_wins - white_wins
    except Exception:
        total = black_wins = white_wins = draws = 0

    stats = adapter.get_sample_stats()

    print(f"\n=== 对弈进度统计 ===")
    print(f"  已完成: {stats['completed_games']}/{stats['total_games']}")
    if total > 0:
        print(f"  黑胜: {black_wins} ({black_wins / total * 100:.1f}%)")
        print(f"  白胜: {white_wins} ({white_wins / total * 100:.1f}%)")
        print(f"  平局: {draws} ({draws / total * 100:.1f}%)")
    print(f"  总样本: {stats['total_produced']} (已淘汰: {stats['evicted_samples']})")
    print(f"  当前缓冲区: {summary['total_samples']} 样本, "
          f"{summary['chunk_count']} chunk")

    print(f"\n[耗时] 总耗时 {elapsed:.1f}s")

    print(f"\n[完成] 缓冲区状态: {summary['total_samples']} 样本, "
          f"{summary['chunk_count']} chunk, 目录: {adapter._buffer._buffer_dir}")


# ---------- 主流程 ----------

def run_sample_buffer(args) -> None:
    """执行自对弈样本生成

    Args:
        args: 解析后的 CLI 参数
    """
    # 验证 games 参数
    if args.games < 1:
        print("[错误] --games 必须 ≥ 1", file=sys.stderr)
        sys.exit(1)

    # 自动识别 model（未指定时查找 cnn_v11 最高代数）
    if args.model is None:
        from ai.registry import get_highest_gen
        highest_gen = get_highest_gen("cnn_v11")
        args.model = f"cnn_v11.{highest_gen}"
        print(f"[自动识别] 未指定 --model，使用 cnn_v11 最高代数: {args.model}")

    # 构建配置
    config = _build_config(args)
    print(f"[超参数] simulations={config.simulations}, c_puct={config.c_puct}, "
          f"temperature={config.temperature}, temp_steps={config.temp_steps}, "
          f"dirichlet_epsilon={config.dirichlet_epsilon}, "
          f"virtual_loss={config.virtual_loss}")
    print(f"[模型] {args.model}")
    print(f"[模式] 自对弈（双方交替执方），总局数={args.games}")

    # 注册 SIGINT 信号处理器（直接终止）
    signal.signal(signal.SIGINT, _signal_handler)

    adapter = None
    try:
        # 创建适配器（内部完成模型路径解析和 C++ GameRunner 初始化）
        adapter = CppMctsSampleBufferAdapter(
            model_spec=args.model,
            num_games=args.games,
            config=config,
            num_workers=args.workers,
            buffer_dir=args.output_dir,
            buffer_max_size=args.buffer_max_size,
            chunk_games=args.chunk_games,
        )

        # 启动所有 C++ 游戏线程（非阻塞）
        t_start = time.time()
        adapter.start()

        # 轮询进度并实时打印
        last_print_time = 0.0

        while not adapter.is_finished():
            # 轮询进度并触发样本落盘
            adapter.poll_and_ingest()

            stats = adapter.get_sample_stats()
            now = time.time()
            # 每秒最多打印一次进度
            if now - last_print_time >= 1.0:
                print(f"[进度] 对局: {stats['completed_games']}/{stats['total_games']} | "
                      f"缓冲区: {stats['buffer_samples']} 样本, "
                      f"{stats['buffer_chunks']} chunk | "
                      f"淘汰: {stats['evicted_samples']}")
                last_print_time = now
            time.sleep(0.1)

        # 等待所有进行中的对局完成并落盘
        adapter.wait_completion()
        elapsed = time.time() - t_start

        # 打印最终统计
        _print_statistics(adapter, elapsed)

    except (FileNotFoundError, ValueError) as e:
        print(f"[错误] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[错误] 自对弈执行失败: {e}", file=sys.stderr)
        if adapter is not None:
            try:
                adapter.wait_completion()
            except Exception:
                pass
        raise


# ---------- 入口 ----------

if __name__ == "__main__":
    args = parse_args()
    run_sample_buffer(args)
