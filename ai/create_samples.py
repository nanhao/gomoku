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
        default=100,
        help='对局数，默认 100'
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
    print(f"  对局数 (games): {args.games}")
    print(f"  模式 (mode):    {args.mode}")

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
    # 以 creator.augment 运行时属性为真相源：winner_path 模式会在 __init__ 中
    # 强制禁用增强，即使命令行未传 --no-augment，这里也必须打印"禁用"。
    print(f"  数据增强:   {'启用 (8 变换)' if creator.augment else '禁用'}")
    print(f"  样本数:     {len(samples)}")
    print(f"  输出路径:   {args.output}")

    # 过滤统计输出（复用 should_keep_game 消除重复逻辑和类型比较 bug）
    filter_active = (
        temp_config is not None
        and temp_config.temp_side in ("black", "white", "random")
    )
    if filter_active:
        # 复用 should_keep_game 而非手写类型比较，
        # 避免 str("black") == int(1) 的类型比较 bug
        # results 来自 start_ai_vs_ai()（L159），其每项字典与 record.json 同构
        # （均含 winner/temp_side_this_game），可复用 should_keep_game 判定过滤
        kept = sum(1 for r in results if should_keep_game(r))
        filtered_out = len(results) - kept
        print(f"\n样本过滤: 启用 (采样方获胜过滤)")
        print(f"  保留对局: {kept}")
        print(f"  过滤对局: {filtered_out}")
    else:
        print(f"\n样本过滤: 未启用 (非单方采样模式)")


if __name__ == "__main__":
    main()
