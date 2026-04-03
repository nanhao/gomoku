"""FC v1 样本检视入口脚本

将指定 sample.json 喂给模型，逐步打印模型输出与 sample output 的对比差异。

用法:
    python ai/fc_v1/inspect_main.py --sample train_data/fc_v1_cycle_1_4/1/sample.json
    python ai/fc_v1/inspect_main.py --sample train_data/fc_v1_cycle_1_4/1/sample.json --model-path ai/fc_v1/fc_v1.pth
"""

import sys
import os
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from ai.models.fc_v1.model import FCv1Model
from ai.models.fc_v1.sample_inspector import SampleInspector


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="FC v1 样本检视器：将 sample.json 喂给模型，打印每步模型输出与 target 的对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认模型（fc_v1.pth）检视样本
  python ai/fc_v1/inspect_main.py --sample train_data/fc_v1_cycle_1_4/1/sample.json

  # 指定模型路径
  python ai/fc_v1/inspect_main.py --sample train_data/fc_v1_cycle_1_4/1/sample.json --model-path ai/fc_v1/fc_v1.pth
        """
    )

    parser.add_argument(
        "--sample",
        type=str,
        required=True,
        help="sample.json 文件路径"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="模型文件路径，默认加载 ai/fc_v1/fc_v1.pth"
    )

    args = parser.parse_args()

    # 检查 sample 文件是否存在
    if not os.path.exists(args.sample):
        print(f"错误: sample 文件不存在: {args.sample}")
        sys.exit(1)

    # 加载模型
    print(f"加载模型...")
    try:
        model = FCv1Model(model_path=args.model_path)
    except Exception as e:
        print(f"错误: 模型加载失败: {e}")
        sys.exit(1)

    # 创建检视器
    inspector = SampleInspector(model)

    # 执行检视
    print(f"检视样本: {args.sample}")
    print("=" * 60)

    try:
        results = inspector.inspect(args.sample)
    except Exception as e:
        print(f"错误: 检视失败: {e}")
        sys.exit(1)

    # 打印 summary
    total_steps = len(results)
    win_step = next((r["step"] for r in results if r["win_pos"] is not None), None)
    avg_loss = sum(r["loss"] for r in results) / total_steps if total_steps > 0 else 0.0

    print(f"总步数: {total_steps}")
    print(f"胜利步: {win_step if win_step is not None else '无'}")
    print(f"平均loss: {avg_loss:.4f}")
    print("=" * 60)
    print()

    # 逐步打印对比
    for r in results:
        print(inspector.format_step(r))
        print()


if __name__ == "__main__":
    main()
