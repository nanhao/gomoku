"""MCTS C++ 双模型对战脚本

使用 C++ MCTS 引擎让两个 cnn_v11 模型并行对战，输出胜率统计。
所有对局在 C++ 层面用多线程并行运行，共享全局 MCTS 线程池和 GPU 推理引擎。

用法：
    python src/ai/scripts/mcts_battle_cpp.py --ai1 cnn_v11.1 --ai2 cnn_v11.2 --games 100
    python src/ai/scripts/mcts_battle_cpp.py --ai1 cnn_v11.1 --ai2 cnn_v11.2 --games 50 --simulations 400

设计文档：design/47/main_docs/overview.md
"""

import argparse
import os
import signal
import sys
import time
from typing import List

# 确保项目根、src 和 cpp\python 在 sys.path 中
# 脚本路径: src/ai/scripts/mcts_battle_cpp.py，4 层 dirname 到项目根
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
from cpp.python.cpp_battle_game_result import CppBattleGameResult
from cpp.python.cpp_mcts_battle_adapter import CppMctsBattleAdapter
from config.mcts import MCTS_VIRTUAL_LOSS


def _print_gpu_stats() -> None:
    """打印 GPU 推理线程运行时监控统计"""
    try:
        import cpp_mcts  # type: ignore[import-not-found]
        stats = cpp_mcts.get_gpu_stats()
        if stats.total_batches == 0:
            return  # 没有推理发生，跳过

        print(f"\n=== GPU 推理监控 ===")
        print(f"总批次数: {stats.total_batches}")
        print(f"总请求数: {stats.total_requests}")
        print(f"平均批次大小: {stats.avg_batch_size():.1f}")
        print(f"批次范围: [{stats.batch_size_min}, {stats.batch_size_max}]")
        print(f"队列峰值: {stats.queue_peak_size}")
        print(f"GPU 等待比例: {stats.wait_ratio() * 100:.1f}%  "
              f"({'GPU 利用率高' if stats.wait_ratio() < 0.3 else 'GPU 空等较多，瓶颈在 MCTS 生产端' if stats.wait_ratio() > 0.6 else '正常'})")
    except Exception:
        pass  # cpp_mcts 不可用时静默跳过


# ---------- 常量 ----------


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
        description="MCTS C++ 双模型对战脚本 — 输出双方胜率统计",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ai1", required=True,
                   help="模型1 registry spec（如 cnn_v11.1）")
    p.add_argument("--ai2", required=True,
                   help="模型2 registry spec（如 cnn_v11.2）")
    p.add_argument("--games", type=int, required=True,
                   help="对战局数（≥1）")
    p.add_argument("--workers", type=int, default=0,
                   help="游戏工作线程数（0=每局一个线程，>0=限定线程数，默认 0）")
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

def _print_per_game_results(results: List[CppBattleGameResult],
                            model1_name: str, model2_name: str) -> None:
    """打印每局对战结果

    Args:
        results: 对战结果列表
        model1_name: model1 显示名称
        model2_name: model2 显示名称
    """
    total = len(results)
    color_name = {1: "黑", 2: "白"}
    for r in results:
        if r.winner == 0:
            winner_desc = "平局"
        elif r.winner == 1:
            winner_desc = "黑方胜"
        else:
            winner_desc = "白方胜"
        print(f"[局 {r.game_index + 1}/{total}] "
              f"{model1_name}({color_name[r.model1_color]}) vs "
              f"{model2_name}({color_name[r.model2_color]}) → "
              f"{winner_desc} ({r.move_count}步)")


def _print_statistics(model1_name: str, model2_name: str,
                      results: List[CppBattleGameResult]) -> None:
    """打印对战统计信息

    Args:
        model1_name: model1 显示名称
        model2_name: model2 显示名称
        results: 对战结果列表
    """
    total = len(results)
    model1_wins = sum(1 for r in results if r.model1_won)
    model2_wins = sum(1 for r in results if r.model2_won)
    draws = total - model1_wins - model2_wins

    print(f"\n=== 对战结果 ===")
    print(f"总局数: {total}")
    print(f"{model1_name} 胜: {model1_wins} 局 ({model1_wins / total * 100:.1f}%)")
    print(f"{model2_name} 胜: {model2_wins} 局 ({model2_wins / total * 100:.1f}%)")
    print(f"平局: {draws} 局 ({draws / total * 100:.1f}%)")


# ---------- 主流程 ----------

def run_battle(args) -> None:
    """执行所有局对战

    Args:
        args: 解析后的 CLI 参数
    """
    # 验证 games 参数
    if args.games < 1:
        print("[错误] --games 必须 ≥ 1", file=sys.stderr)
        sys.exit(1)

    # 构建配置
    config = _build_config(args)
    print(f"[超参数] simulations={config.simulations}, c_puct={config.c_puct}, "
          f"temperature={config.temperature}, temp_steps={config.temp_steps}, "
          f"dirichlet_epsilon={config.dirichlet_epsilon}, "
          f"virtual_loss={config.virtual_loss}")

    # 模型名称直接使用 spec 字符串用于显示
    model1_display = args.ai1
    model2_display = args.ai2
    print(f"[模型] model1={model1_display}, model2={model2_display}")

    # 注册 SIGINT 信号处理器（直接终止，不等待）
    original_handler = signal.signal(signal.SIGINT, _signal_handler)

    adapter = None
    try:
        # 创建适配器（内部完成模型路径解析和 C++ GameRunner 初始化）
        adapter = CppMctsBattleAdapter(
            model1_spec=args.ai1,
            model2_spec=args.ai2,
            num_games=args.games,
            config=config,
            num_workers=args.workers,
        )

        # 启动所有 C++ 游戏线程（非阻塞）
        t_start = time.time()
        adapter.start()

        # 轮询进度并实时打印
        last_print_time = 0.0
        last_gpu_print = 0.0
        while not adapter.is_finished():
            completed, total = adapter.get_progress()
            now = time.time()
            # 每秒最多打印一次进度
            if now - last_print_time >= 1.0:
                print(f"[进度] {completed}/{total}")
                last_print_time = now
            # 每 5 秒打印一次 GPU 实时统计
            if now - last_gpu_print >= 5.0:
                _print_gpu_stats()
                last_gpu_print = now
            time.sleep(0.1)

        # 等待所有进行中的对局完成
        adapter.wait_completion()
        elapsed = time.time() - t_start

        # 获取结果
        results = adapter.get_results()

        # 打印每局结果和最终统计
        if results:
            _print_per_game_results(results, model1_display, model2_display)
            _print_statistics(model1_display, model2_display, results)

        print(f"\n[完成] 共 {len(results)} 局，耗时 {elapsed:.1f}s")

        # 打印 GPU 推理监控统计
        _print_gpu_stats()

    except (FileNotFoundError, ValueError) as e:
        print(f"[错误] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[错误] 对战执行失败: {e}", file=sys.stderr)
        if adapter is not None:
            try:
                adapter.wait_completion()
            except Exception:
                pass
        raise
    finally:
        signal.signal(signal.SIGINT, original_handler)


# ---------- 入口 ----------

if __name__ == "__main__":
    args = parse_args()
    run_battle(args)
