"""AI 模型训练器入口脚本

独立运行的训练器入口，支持多种模型。

用法:
    # 训练周期（fc_v1 模型）
    python ai/train_main.py --model fc_v1 --cycle [--rounds 100] [--games 100]

    # 仅自我对弈生成数据（不训练）
    python ai/train_main.py --model fc_v1 --self-play --games 100

    # 从已有数据目录训练模型
    python ai/train_main.py --model fc_v1 --train-from-dir ai/train_data/fc_v1_self_play_1
"""

import sys
import os
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ai.registry import get_trainer
from ai.runners import run_self_play, run_train_from_dir, run_cycle


def main():
    """主函数：argparse 解析、参数校验、函数路由"""
    parser = argparse.ArgumentParser(
        description='AI 模型训练器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 训练周期 fc_v1（100轮，每轮100局）
  python ai/train_main.py --model fc_v1 --cycle

  # 训练周期 fc_v1（自定义参数）
  python ai/train_main.py --model fc_v1 --cycle --rounds 50 --games 50

  # 从新模型开始训练 fc_v1
  python ai/train_main.py --model fc_v1 --cycle --fresh

  # 仅自我对弈生成数据（不训练）
  python ai/train_main.py --model fc_v1 --self-play --games 100

  # 从已有数据目录训练模型
  python ai/train_main.py --model fc_v1 --train-from-dir ai/train_data/fc_v1_self_play_1 --epochs 1000
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='fc_v1',
        help='模型名称，如 fc_v1、cnn_v1、cnn_v2，默认 fc_v1'
    )
    
    parser.add_argument(
        '--cycle',
        action='store_true',
        help='开启训练周期模式'
    )
    
    parser.add_argument(
        '--rounds',
        type=int,
        default=10,
        help='训练周期轮次数 n，默认 10'
    )
    
    parser.add_argument(
        '--games',
        type=int,
        default=100,
        help='每轮对局数 m，默认 100'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=1000,
        help='训练迭代数，默认 1000'
    )
    
    parser.add_argument(
        '--rate',
        type=float,
        default=0.9,
        help='奖励衰减率 r，训练周期时传给 Evaluator，默认 0.9'
    )
    
    parser.add_argument(
        '--fresh',
        action='store_true',
        help='从新模型开始训练（不加载已保存的模型）'
    )

    parser.add_argument(
        '--self-play',
        action='store_true',
        help='仅执行自我对弈+评价，生成训练数据（不训练）'
    )

    parser.add_argument(
        '--train-from-dir',
        type=str,
        default=None,
        help='训练数据目录路径（自动发现其下所有 sample.json 合并训练），或直接指定 .json 文件'
    )

    parser.add_argument(
        '--merge-output',
        type=str,
        default=None,
        help='--self-play 模式下，对局完成后自动合并样本到指定文件'
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

    args = parser.parse_args()
    
    # 验证参数：三个模式互斥（--cycle、--self-play、--train-from-dir 三选一）
    active_modes = []
    if args.cycle:
        active_modes.append('--cycle')
    if args.self_play:
        active_modes.append('--self-play')
    if args.train_from_dir:
        active_modes.append('--train-from-dir')

    if len(active_modes) == 0:
        parser.error("请指定 --cycle、--self-play 或 --train-from-dir 参数")
    elif len(active_modes) > 1:
        parser.error(f"以下参数不能同时使用: {', '.join(active_modes)}")
    
    # 创建训练器
    scheduler_kwargs = {}
    if args.lr_scheduler:
        scheduler_kwargs = {
            'lr_scheduler_type': args.lr_scheduler,
            'lr_step_size': args.lr_step_size,
            'lr_gamma': args.lr_gamma,
            'lr_patience': args.lr_patience,
            'lr_factor': args.lr_factor
        }

    try:
        trainer = get_trainer(args.model, fresh=args.fresh, **scheduler_kwargs)
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)
    
    # 根据模式路由到对应的运行器
    if args.self_play:
        run_self_play(args, trainer)
    elif args.train_from_dir:
        run_train_from_dir(args, trainer)
    elif args.cycle:
        run_cycle(args, trainer)


if __name__ == "__main__":
    main()
