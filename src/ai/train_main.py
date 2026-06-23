"""AI 模型训练器入口脚本

独立运行的训练器入口，支持多种模型。

用法:
    # 从已有数据目录训练模型
    python ai/train_main.py --model fc_v1 --train-from-dir ai/train_data/fc_v1_self_play_1
"""

import sys
import os
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ai.registry import get_trainer
from ai.runners import run_train_from_dir


def build_parser():
    """构造命令行参数解析器

    将 parser 构建逻辑提取为独立函数，便于测试直接使用真实 parser，
    避免 UT-3 中手动镜像 parser 定义导致与真实代码不一致的风险。

    Returns:
        argparse.ArgumentParser: 参数解析器实例
    """
    parser = argparse.ArgumentParser(
        description='AI 模型训练器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从已有数据目录训练模型
  python ai/train_main.py --model fc_v1 --train-from-dir ai/train_data/fc_v1_self_play_1

  # mini-batch 训练
  python ai/train_main.py --model fc_v1 --train-from-dir ai/train_data/fc_v1_self_play_1 --batch-size 32 --shuffle
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='fc_v1',
        help='模型指定，如 fc_v1、cnn_v3.2，默认 fc_v1'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=1000,
        help='训练迭代数，默认 1000'
    )

    parser.add_argument(
        '--fresh',
        action='store_true',
        help='从新模型开始训练（不加载已保存的模型）'
    )

    parser.add_argument(
        '--train-from-dir',
        type=str,
        default=None,
        help='训练数据目录路径（递归加载其下所有 .json 文件），或直接指定 .json 文件'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='每批次样本数，提供时启用 mini-batch 训练，默认全量更新'
    )

    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='每个 epoch 是否打乱数据顺序（仅 mini-batch 模式有效）'
    )

    parser.add_argument(
        '--lr-scheduler',
        type=str,
        default=None,
        choices=['step', 'plateau'],
        help='学习率调度类型：step（固定间隔衰减）或 plateau（按平台衰减）'
    )

    parser.add_argument(
        '--lr-step-size',
        type=int,
        default=500,
        help='StepLR 衰减间隔（epoch），默认 500'
    )

    parser.add_argument(
        '--lr-gamma',
        type=float,
        default=0.9,
        help='StepLR 衰减系数，默认 0.9'
    )

    parser.add_argument(
        '--lr-patience',
        type=int,
        default=100,
        help='ReduceLROnPlateau 耐心值，默认 100'
    )

    parser.add_argument(
        '--lr-factor',
        type=float,
        default=0.5,
        help='ReduceLROnPlateau 衰减系数，默认 0.5'
    )

    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.0,
        help='验证集比例，0 表示不分割，推荐 0.2，默认 0.0'
    )

    # 新增：初始学习率参数（与 scheduler 解耦，无条件传递）
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='初始学习率，必须为正数，默认 0.001'
    )

    return parser


def main():
    """主函数：argparse 解析、参数校验、函数路由"""
    parser = build_parser()
    args = parser.parse_args()

    # 验证：必须指定 --train-from-dir
    if not args.train_from_dir:
        parser.error("请指定 --train-from-dir 参数")

    # 验证：--lr 必须为正数
    if args.lr <= 0:
        parser.error("--lr 必须为正数")

    # 创建训练器
    # lr 无条件加入 kwargs（与 --lr-scheduler 解耦）
    kwargs = {'lr': args.lr}
    if args.lr_scheduler:
        # scheduler 参数仅当 --lr-scheduler 指定时才追加
        kwargs.update({
            'lr_scheduler_type': args.lr_scheduler,
            'lr_step_size': args.lr_step_size,
            'lr_gamma': args.lr_gamma,
            'lr_patience': args.lr_patience,
            'lr_factor': args.lr_factor
        })

    try:
        trainer = get_trainer(args.model, fresh=args.fresh, **kwargs)
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)
    
    # 执行训练
    run_train_from_dir(args, trainer)


if __name__ == "__main__":
    main()
