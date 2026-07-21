"""MCTS 双模型对战脚本

使用 MCTS 搜索让两个模型对打，输出胜率统计。

用法：
    python src/ai/scripts/mcts_battle.py --model1 cnn_v8.10 --model2 cnn_v8.11 --games 100
    python src/ai/scripts/mcts_battle.py --model1 cnn_v8.10 --model2 cnn_v8.11 --games 200 --workers 4

设计文档：design/42/main_docs/mcts_battle_design.md
"""

import argparse
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

# 确保项目根在 sys.path 中
_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import torch

from ai.scripts.mcts_battle_worker import MctsBattleWorker, BattleGameResult
from ai.scripts.mcts_config import MctsSelfPlayConfig
from config.mcts import MCTS_BATCH_SIZE, MCTS_VIRTUAL_LOSS


# ---------- 常量 ----------

# 中断标志位，由信号处理器设置
_interrupted = False


def _build_hyperparams(args) -> dict:
    """从 CLI args 构建超参数字典

    与 mcts_self_play._build_hyperparams 逻辑一致，消除重复。
    """
    return {
        "simulations": args.simulations,
        "c_puct": args.c_puct,
        "temp_steps": args.temp_steps,
        "temperature": args.temperature,
        "dirichlet_epsilon": args.dirichlet_epsilon,
        "dirichlet_alpha": args.dirichlet_alpha,
        "batch_size": args.batch_size if args.batch_size is not None else MCTS_BATCH_SIZE,
        "virtual_loss": args.virtual_loss if args.virtual_loss is not None else MCTS_VIRTUAL_LOSS,
        "use_prior": not args.no_prior,
    }


def _signal_handler(signum, frame):
    """SIGINT 信号处理器 — 设置中断标志"""
    global _interrupted
    _interrupted = True
    print("\n[中断] 收到 Ctrl+C，正在等待进行中的对局完成...")


def _distribute_indices(total: int, num_chunks: int) -> List[List[int]]:
    """将 [0, total) 的整数均匀分配到 num_chunks 个列表中

    与 mcts_self_play._distribute_indices 逻辑一致。

    Args:
        total: 总数
        num_chunks: 分块数

    Returns:
        列表的列表，每个子列表包含分配到的索引
    """
    if num_chunks <= 0:
        num_chunks = 1
    indices = list(range(total))
    chunks: List[List[int]] = [[] for _ in range(num_chunks)]
    for i, idx in enumerate(indices):
        chunks[i % num_chunks].append(idx)
    # 移除空 chunk（当 num_chunks > total 时）
    return [c for c in chunks if c]


# ---------- 参数解析 ----------

def parse_args():
    """解析 CLI 参数"""
    p = argparse.ArgumentParser(
        description="MCTS 双模型对战脚本 — 输出双方胜率统计",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model1", required=True,
                   help="模型1 注册名.代数，如 cnn_v8.10")
    p.add_argument("--model2", required=True,
                   help="模型2 注册名.代数，如 cnn_v8.11")
    p.add_argument("--games", type=int, required=True,
                   help="对战局数")
    p.add_argument("--simulations", type=int, default=800,
                   help="每步 MCTS 模拟次数（默认 800）")
    p.add_argument("--c-puct", type=float, default=2.0,
                   help="PUCT 探索系数（默认 2.0）")
    p.add_argument("--temp-steps", type=int, default=8,
                   help="温度探索步数，前 N 步 τ=1.0（默认 8）")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="温度参数（默认 1.0）")
    p.add_argument("--dirichlet-epsilon", type=float, default=0.25,
                   help="狄利克雷噪声比例（默认 0.25）")
    p.add_argument("--dirichlet-alpha", type=float, default=0.03,
                   help="狄利克雷分布参数（默认 0.03）")
    p.add_argument("--batch-size", type=int, default=None,
                   help="每批 simulation 数（默认读取 MCTS_BATCH_SIZE 配置）")
    p.add_argument("--virtual-loss", type=float, default=None,
                   help="virtual loss 值（默认读取 MCTS_VIRTUAL_LOSS 配置）")
    p.add_argument("--no-prior", action="store_true",
                   help="禁用模型先验概率，MCTS 仅依赖价值评估")
    p.add_argument("--device", default="cuda",
                   help="推理设备（默认 cuda）")
    p.add_argument("--workers", type=int, default=None,
                   help="并行 worker 进程数（默认 CPU 核数，1 为单进程串行）")
    return p.parse_args()


# ---------- 输出函数 ----------

def _print_statistics(model1_name: str, model2_name: str,
                      results: List[BattleGameResult]) -> None:
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


def _print_per_game_results(results: List[BattleGameResult],
                            model1_name: str, model2_name: str) -> None:
    """打印每局对战结果

    Args:
        results: 对战结果列表
        model1_name: model1 显示名称
        model2_name: model2 显示名称
    """
    total = len(results)
    for r in results:
        color_name = {1: "黑", 2: "白"}
        if r.winner == 0:
            winner_desc = "平局"
        elif r.winner == 1:
            winner_desc = "黑方胜"
        else:
            winner_desc = "白方胜"
        print(f"[局 {r.game_idx + 1}/{total}] "
              f"{model1_name}({color_name[r.model1_color]}) vs "
              f"{model2_name}({color_name[r.model2_color]}) → "
              f"{winner_desc} ({r.move_count}步)")


# ---------- 主流程 ----------

def run_battle(args):
    """执行所有局对战

    根据 --workers 参数选择单进程或多进程模式。
    """
    # 确定 worker 数
    workers = getattr(args, 'workers', 1)
    if workers is None:
        workers = os.cpu_count() or 1
    workers = max(1, workers)

    # 设备：多进程模式下固定使用 CPU
    if workers > 1:
        device = "cpu"
    else:
        device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[设备] {device}")
    print(f"[workers] {workers}")

    # 超参数汇总
    hyperparams = _build_hyperparams(args)
    print(f"[超参数] {hyperparams}")

    # 模型名称直接使用 spec 字符串用于显示
    model1_display = args.model1
    model2_display = args.model2
    print(f"[模型] model1={model1_display}, model2={model2_display}")

    # 生成本次运行的唯一标识
    run_id = f"{int(time.time() * 1000) % 1000000:06d}"

    # 注册 SIGINT 信号处理器
    global _interrupted
    _interrupted = False
    original_handler = signal.signal(signal.SIGINT, _signal_handler)

    try:
        all_results: List[BattleGameResult] = []

        if workers == 1:
            all_results = _run_single_process(
                args, hyperparams, run_id, device
            )
        else:
            all_results = _run_multi_process(
                args, hyperparams, run_id, workers
            )

        # 打印结果
        if all_results:
            _print_per_game_results(all_results, model1_display, model2_display)
            _print_statistics(model1_display, model2_display, all_results)

        print(f"\n[完成] 共 {len(all_results)} 局")

    finally:
        signal.signal(signal.SIGINT, original_handler)


def _run_single_process(args, hyperparams: dict, run_id: str, device) -> List[BattleGameResult]:
    """单进程模式：加载两个模型并串行执行所有对局"""
    config = MctsSelfPlayConfig(
        simulations=hyperparams["simulations"],
        c_puct=hyperparams["c_puct"],
        temp_steps=hyperparams["temp_steps"],
        temperature=hyperparams["temperature"],
        dirichlet_epsilon=hyperparams["dirichlet_epsilon"],
        dirichlet_alpha=hyperparams["dirichlet_alpha"],
        batch_size=hyperparams["batch_size"],
        virtual_loss=hyperparams["virtual_loss"],
        use_prior=hyperparams["use_prior"],
    )

    print(f"[加载模型1] {args.model1}")
    print(f"[加载模型2] {args.model2}")

    worker = MctsBattleWorker(
        model1_spec=args.model1,
        model2_spec=args.model2,
        device=str(device),
        config=config,
    )

    game_indices = list(range(args.games))
    if _interrupted:
        return []
    return worker.run_games(game_indices=game_indices, run_id=run_id)


def _run_multi_process(args, hyperparams: dict, run_id: str,
                       workers: int) -> List[BattleGameResult]:
    """多进程模式：通过 ProcessPoolExecutor 并行执行对局"""
    config = MctsSelfPlayConfig(
        simulations=hyperparams["simulations"],
        c_puct=hyperparams["c_puct"],
        temp_steps=hyperparams["temp_steps"],
        temperature=hyperparams["temperature"],
        dirichlet_epsilon=hyperparams["dirichlet_epsilon"],
        dirichlet_alpha=hyperparams["dirichlet_alpha"],
        batch_size=hyperparams["batch_size"],
        virtual_loss=hyperparams["virtual_loss"],
        use_prior=hyperparams["use_prior"],
    )

    # 分配 game_indices
    chunks = _distribute_indices(args.games, workers)
    print(f"[分配] {workers} 个 worker，各负责局数: {[len(c) for c in chunks]}")

    all_results: List[BattleGameResult] = []
    completed_games = 0
    t_total_start = time.time()

    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for chunk in chunks:
                future = executor.submit(
                    MctsBattleWorker.worker_entry,
                    model1_spec=args.model1,
                    model2_spec=args.model2,
                    device="cpu",
                    config=config,
                    game_indices=chunk,
                    run_id=run_id,
                )
                futures[future] = len(chunk)

            for future in as_completed(futures):
                chunk_size = futures[future]
                try:
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
                    completed_games += chunk_size
                    print(f"[进度] worker 完成 {chunk_size} 局，累计 {completed_games}/{args.games}")
                except Exception as e:
                    print(f"[错误] worker 执行失败: {e}")

    except KeyboardInterrupt:
        print("\n[中断] 等待进行中的 worker 完成...")

    elapsed_total = time.time() - t_total_start
    print(f"[耗时] 总耗时 {elapsed_total:.1f}s")

    return all_results


# ---------- 入口 ----------

if __name__ == "__main__":
    args = parse_args()
    run_battle(args)
