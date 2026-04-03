"""FC v1 训练器入口脚本

独立运行的训练器入口，不依赖 cli 包。

用法:
    # 单轮次训练
    python ai/fc_v1/train.py --sample-dir train_data/{轮次目录}/{局数目录}
    
    # 训练周期
    python ai/fc_v1/train.py --cycle [--rounds 100] [--games 100]
"""

import sys
import os
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from ai.models.fc_v1.trainer import FCv1Trainer


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='FC v1 模型训练器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单轮次训练（默认10轮）
  python ai/fc_v1/train.py --sample-dir train_data/random_vs_random_1/1
  
  # 单轮次训练（自定义轮数）
  python ai/fc_v1/train.py --sample-dir train_data/random_vs_random_1/1 --epochs 20
  
  # 训练周期（100轮，每轮100局）
  python ai/fc_v1/train.py --cycle
  
  # 训练周期（自定义参数）
  python ai/fc_v1/train.py --cycle --rounds 50 --games 50
  
  # 从新模型开始训练（不加载已保存模型）
  python ai/fc_v1/train.py --cycle --fresh
  
  # 单轮次训练，从新模型开始
  python ai/fc_v1/train.py --sample-dir train_data/random_vs_random_1/1 --fresh
        """
    )
    
    parser.add_argument(
        '--sample-dir',
        type=str,
        help='sample.json 所在目录路径（单轮次训练）'
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
        help='单轮次训练的训练轮数，默认 1000'
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
    
    args = parser.parse_args()
    
    # 验证参数
    if not args.cycle and not args.sample_dir:
        parser.error("请指定 --sample-dir 或 --cycle 参数")
    
    if args.cycle and args.sample_dir:
        parser.error("--cycle 和 --sample-dir 不能同时使用")
    
    # 创建训练器
    trainer = FCv1Trainer(fresh=args.fresh)
    
    if args.sample_dir:
        # 单轮次训练
        sample_dir = args.sample_dir
        
        # 检查目录是否存在
        if not os.path.exists(sample_dir):
            print(f"错误: 目录不存在: {sample_dir}")
            sys.exit(1)
        
        # 检查 sample.json 是否存在
        sample_path = os.path.join(sample_dir, "sample.json")
        if not os.path.exists(sample_path):
            print(f"错误: sample.json 不存在: {sample_path}")
            sys.exit(1)
        
        print(f"开始训练 fc_v1 模型...")
        print(f"使用数据: {sample_path}")
        
        try:
            # 执行训练
            loss = trainer.train_single(sample_dir, epochs=args.epochs)
            
            print(f"训练完成！平均损失: {loss:.4f}")
            
            # 保存模型
            model_path = trainer.model.save()
            print(f"模型已保存到: {model_path}")
            
        except Exception as e:
            print(f"训练失败: {e}")
            sys.exit(1)
    
    elif args.cycle:
        # 训练周期
        print(f"开始 fc_v1 训练周期...")
        print(f"轮次数: {args.rounds}")
        print(f"每轮对局数: {args.games}")
        print(f"奖励衰减率: {args.rate}")
        
        try:
            # 执行训练周期
            trainer.train_cycle(
                rounds=args.rounds,
                games=args.games,
                epochs=args.epochs,
                rate=args.rate
            )
            
            print("训练周期完成！")
            
        except NotImplementedError as e:
            print(f"功能未实现: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"训练失败: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
