"""独立 CLI 脚本：执行 AI 对弈并生成训练样本

用法:
    python ai/create_samples.py --model cnn_v3 --mode winning_move --games 100 --output ai/train_data/win_samples.json
"""

import sys
import os
import json
import argparse
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ai.registry import get_model
from ai.engine import AIEngine
from ai.game_manager import AIGameManager
from ai.data_recorder import DataRecorder
from ai.sample_creator import SampleCreator
from ai.move_sampler import TempConfig, SUPPORTED_TEMP_SIDES, TEMP_SIDE_BOTH
from ai.game_filter import should_keep_game


def is_sample_game(result: dict) -> bool:
    """判定单局对弈结果是否为样本局。

    样本局需同时满足：
    1. winner ∈ {1, 2}（非平局、非未结束）
       — 此前置检查与 SampleCreator.create 的 winner 过滤对齐，
         确保计数与最终样本生成的一致性。
    2. should_keep_game(result) 返回 True

    此判定逻辑与 SampleCreator.create 的过滤逻辑保持一致，
    确保计数与最终样本生成的一致性。

    Args:
        result: 对局结果字典，含 winner 和 temp_side_this_game 键。
            {"winner": int, "temp_side_this_game": str|None}

    Returns:
        True → 该局为样本局；False → 非样本局
    """
    winner = result.get("winner", 0)
    if winner not in (1, 2):
        return False
    return should_keep_game(result)


def print_sample_progress(sample_count: int, target: int, total_games: int) -> None:
    """打印样本局收集进度。

    纯输出函数，不负责触发判定（触发判定在调用方主循环中）。

    前置条件: target > 0（由 CLI 参数校验保证）

    Args:
        sample_count: 当前已收集的样本局数
        target: 目标样本局数（必须 > 0）
        total_games: 已完成的原始对弈总数
    """
    assert target > 0, f"target 必须 > 0，实际为 {target}"
    pct = sample_count * 100 // target
    print(f"进度: 样本局 {sample_count}/{target} ({pct}%), 已对弈 {total_games} 局")


def main():
    """主函数：解析参数、加载模型、执行对弈、生成样本、写入输出"""
    parser = argparse.ArgumentParser(
        description='AI 对弈样本生成器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # winning_move 模式（默认）：仅取每局最后一步，启用 D4 增强
  python ai/create_samples.py --model cnn_v3.1 --output ai/train_data/win_samples.json
  python ai/create_samples.py --model cnn_v3.1 --mode winning_move --games 200 --output ai/train_data/win_samples.json

  # winner_path 模式：取获胜方每一步，强制不做 D4 增强
  python ai/create_samples.py --model cnn_v3.1 --mode winner_path --games 200 --output ai/train_data/winner_path_samples.json

  # 样本局驱动模式：持续对弈直到收集到目标数量的样本局
  python ai/create_samples.py --model cnn_v3.1 --sample_games 50 --output ai/train_data/win_samples.json

  # 其它模型
  python ai/create_samples.py --model fc_v1 --output ai/train_data/fc_samples.json
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='模型指定，如 cnn_v3.1、fc_v1'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='winning_move',
        choices=['winning_move', 'winner_path'],
        help='sample_creator 模式，默认 winning_move；winner_path 为获胜方整条路径（强制不做 D4 增强）'
    )

    parser.add_argument(
        '--games',
        type=int,
        default=None,
        help='对局数，默认 100（与 --sample_games 互斥）'
    )

    parser.add_argument(
        '--sample_games',
        type=int,
        default=None,
        help='目标样本局数，与 --games 互斥；脚本持续对弈直到样本局达标'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出文件路径（.json）'
    )

    parser.add_argument(
        '--no-augment',
        action='store_true',
        default=False,
        help='禁用数据增强（默认启用 D4 群 8 变换）'
    )

    # ---- 温度采样参数（默认启用，用于自对弈探索）----
    parser.add_argument(
        '--no-temp-sampling',
        action='store_true',
        default=False,
        help='禁用开局阶段温度采样，全程走 argmax'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='温度值 T（>0），越大越随机，默认 1.0'
    )
    parser.add_argument(
        '--temp-moves',
        type=int,
        default=12,
        help='前 N 个全局步启用温度采样，之后走 argmax，默认 12'
    )
    parser.add_argument(
        '--temp-side',
        type=str,
        default=TEMP_SIDE_BOTH,
        choices=list(SUPPORTED_TEMP_SIDES),
        help='启用温度采样的一方，默认 both'
    )
    parser.add_argument(
        '--temp-seed',
        type=int,
        default=None,
        help='温度采样随机种子（用于复现）'
    )

    args = parser.parse_args()

    # ---- 互斥校验: --games 与 --sample_games ----
    if args.games is not None and args.sample_games is not None:
        print("错误: --games 和 --sample_games 不能同时指定", file=sys.stderr)
        sys.exit(1)

    # --sample_games 值校验
    if args.sample_games is not None and args.sample_games <= 0:
        print("错误: --sample_games 的值必须大于0", file=sys.stderr)
        sys.exit(1)

    # --games 值校验
    if args.games is not None and args.games <= 0:
        print("错误: --games 的值必须大于0", file=sys.stderr)
        sys.exit(1)

    # 两者都不指定时，默认 --games 100
    if args.games is None and args.sample_games is None:
        args.games = 100

    # 确定运行模式
    use_sample_mode = (args.sample_games is not None)

    # ---- 构造 TempConfig ----
    if args.no_temp_sampling:
        # 若用户显式传了温度相关参数，提示被忽略
        user_set_temp = (
            args.temperature != 1.0
            or args.temp_moves != 12
            or args.temp_side != TEMP_SIDE_BOTH
            or args.temp_seed is not None
        )
        if user_set_temp:
            warnings.warn(
                "已启用 --no-temp-sampling，以下参数被忽略："
                "--temperature / --temp-moves / --temp-side / --temp-seed",
                stacklevel=2,
            )
        temp_config = None
    else:
        temp_config = TempConfig(
            temperature=args.temperature,
            temp_moves=args.temp_moves,
            temp_side=args.temp_side,
            top_k=None,
            seed=args.temp_seed,
        )

    # 加载模型
    print(f"加载模型: {args.model}")
    model = get_model(args.model)

    # 创建 AIEngine
    engine = AIEngine(model)

    # 创建 DataRecorder
    recorder = DataRecorder(ai_name=f"{model.full_name}_battle")

    # 执行对弈
    print(f"开始 {args.model} 自我对弈...")
    print(f"参数:")
    if use_sample_mode:
        print(f"  目标样本局数 (sample_games): {args.sample_games}")
    else:
        print(f"  对局数 (games): {args.games}")
    print(f"  模式 (mode):    {args.mode}")

    if use_sample_mode:
        # ---- --sample_games 模式：逐局对弈，样本局达标即停 ----
        target = args.sample_games
        results = []
        sample_count = 0
        total_games = 0
        last_pct = 0

        try:
            while sample_count < target:
                total_games += 1
                result = AIGameManager.play_single_game(
                    engine, engine, recorder, total_games,
                    name_prefix="ai_vs_ai", temp_config=temp_config,
                )
                result_dict = {
                    "game": total_games,
                    "winner": result.winner,
                    "temp_side_this_game": result.temp_side_this_game,
                }
                results.append(result_dict)

                if is_sample_game(result_dict):
                    sample_count += 1
                    pct = sample_count * 100 // target
                    # 百分比跨越 10% 整数倍，或达标时打印进度
                    if pct >= last_pct + 10 or sample_count == target:
                        print_sample_progress(sample_count, target, total_games)
                        last_pct = (pct // 10) * 10

        except KeyboardInterrupt:
            pct = sample_count * 100 // target if target > 0 else 0
            print(f"\n[中断] 检测到 Ctrl+C，正在整理已有数据...")
            print(f"[中断] 已收集样本局 {sample_count}/{target} ({pct}%), "
                  f"已完成原始对弈 {total_games} 局")
            if total_games == 0:
                print("[中断] 未完成任何对弈，退出")
                sys.exit(0)
    else:
        # ---- --games 模式：原有行为不变 ----
        def on_progress(current, total):
            pct = current * 100 // total
            print(f"\r进度: {current}/{total} ({pct}%)", end="", flush=True)

        results = AIGameManager.start_ai_vs_ai(
            engine, engine, args.games, recorder, on_progress=on_progress,
            temp_config=temp_config,
        )
        print()  # 换行，避免后续输出黏在进度行上

    # 打印对局统计
    black_wins = sum(1 for r in results if r["winner"] == 1)
    white_wins = sum(1 for r in results if r["winner"] == 2)
    draws = sum(1 for r in results if r["winner"] == 3)
    print(f"\n对局统计: 黑胜 {black_wins}, 白胜 {white_wins}, 平局 {draws}")

    # 获取轮次目录
    round_dir = recorder.get_round_dir()

    # 生成样本（传入 record_filter 实现对局级过滤）
    creator = SampleCreator(mode=args.mode, augment=not args.no_augment)
    samples = creator.create_from_round_dir(round_dir, record_filter=should_keep_game)

    # 确保输出目录存在
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 写入输出文件
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False)

    print(f"\n样本生成完成:")
    print(f"  对局数:     {len(results)}")
    # --sample_games 模式额外输出样本局数
    if use_sample_mode:
        actual_sample_count = sum(1 for r in results if is_sample_game(r))
        print(f"  样本局数:   {actual_sample_count}")
    # 以 creator.augment 运行时属性为真相源：winner_path 模式会在 __init__ 中
    # 强制禁用增强，即使命令行未传 --no-augment，这里也必须打印"禁用"。
    print(f"  数据增强:   {'启用 (8 变换)' if creator.augment else '禁用'}")
    print(f"  样本数:     {len(samples)}")
    print(f"  输出路径:   {args.output}")

    # 过滤统计输出（复用 should_keep_game 消除重复逻辑和类型比较 bug）
    _print_filter_stats(results, temp_config)


def _print_filter_stats(results, temp_config):
    """输出对局级过滤统计信息。

    Args:
        results: start_ai_vs_ai() 返回的对局结果列表，
            每项字典含 winner / temp_side_this_game 键。
        temp_config: 温度采样配置，None 表示未启用。
    """
    filter_active = (
        temp_config is not None
        and temp_config.temp_side in ("black", "white", "random")
    )
    if filter_active:
        # 复用 should_keep_game 而非手写类型比较，
        # 避免 str("black") == int(1) 的类型比较 bug
        kept = sum(1 for r in results if should_keep_game(r))
        filtered_out = len(results) - kept
        print(f"\n样本过滤: 启用 (采样方获胜过滤)")
        print(f"  保留对局: {kept}")
        print(f"  过滤对局: {filtered_out}")
    else:
        print(f"\n样本过滤: 未启用 (非单方采样模式)")


if __name__ == "__main__":
    main()
