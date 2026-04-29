"""独立 CLI 脚本：执行 AI 对弈并生成训练样本

用法:
    python ai/create_samples.py --model cnn_v3 --mode winning_move --games 100 --output ai/train_data/win_samples.json
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ai.registry import get_model
from ai.engine import AIEngine
from ai.game_manager import AIGameManager
from ai.data_recorder import DataRecorder
from ai.sample_creator import SampleCreator


def main():
    """主函数：解析参数、加载模型、执行对弈、生成样本、写入输出"""
    parser = argparse.ArgumentParser(
        description='AI 对弈样本生成器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python ai/create_samples.py --model cnn_v3.1 --output ai/train_data/win_samples.json
  python ai/create_samples.py --model cnn_v3.1 --mode winning_move --games 200 --output ai/train_data/win_samples.json
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
        help='sample_creator 模式，默认 winning_move'
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

    args = parser.parse_args()

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
        engine, engine, args.games, recorder, on_progress=on_progress
    )
    print()  # 换行，避免后续输出黏在进度行上

    # 打印对局统计
    black_wins = sum(1 for r in results if r["winner"] == 1)
    white_wins = sum(1 for r in results if r["winner"] == 2)
    draws = sum(1 for r in results if r["winner"] == 3)
    print(f"\n对局统计: 黑胜 {black_wins}, 白胜 {white_wins}, 平局 {draws}")

    # 获取轮次目录
    round_dir = recorder.get_round_dir()

    # 生成样本
    creator = SampleCreator(mode=args.mode, augment=not args.no_augment)
    samples = creator.create_from_round_dir(round_dir)

    # 确保输出目录存在
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 写入输出文件
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False)

    print(f"\n样本生成完成:")
    print(f"  对局数:     {len(results)}")
    print(f"  数据增强:   {'禁用' if args.no_augment else '启用 (8 变换)'}")
    print(f"  样本数:     {len(samples)}")
    print(f"  输出路径:   {args.output}")


if __name__ == "__main__":
    main()
