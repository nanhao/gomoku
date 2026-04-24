"""AI 模型训练器入口脚本

独立运行的训练器入口，支持多种模型。

用法:
    # 单轮次训练（fc_v1 模型）
    python ai/train_main.py --model fc_v1 --sample-dir train_data/{轮次目录}/{局数目录}
    
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
import json
import tempfile

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ai.sample_loader import discover_all_data, SampleLoader
from ai.merge_samples import merge_directories

def get_trainer(model_name: str, fresh: bool = False, **kwargs):
    """
    根据模型名称创建对应的训练器

    Args:
        model_name: 模型名称，如 "fc_v1"、"cnn_v1"
        fresh: 是否从新模型开始训练
        **kwargs: 额外参数传递给训练器（如 lr_scheduler_type 等）

    Returns:
        对应的训练器实例

    Raises:
        ValueError: 不支持的模型名称
    """
    if model_name == "fc_v1":
        from ai.models.fc_v1.trainer import FCv1Trainer
        return FCv1Trainer(fresh=fresh, **kwargs)
    elif model_name == "cnn_v1":
        from ai.models.cnn_v1.trainer import CNNV1Trainer
        return CNNV1Trainer(fresh=fresh, **kwargs)
    elif model_name == "cnn_v2":
        from ai.models.cnn_v2.trainer import CNNV2Trainer
        return CNNV2Trainer(fresh=fresh, **kwargs)
    else:
        raise ValueError(f"不支持的模型: {model_name}。目前支持: fc_v1, cnn_v1, cnn_v2")


def discover_sample_dirs(train_data_dir: str) -> list:
    """发现目录下所有包含 sample.json 的子目录

    扫描 train_data_dir 下的直接子目录，返回包含 sample.json 的子目录列表。

    Args:
        train_data_dir: 训练数据根目录（如 ai/train_data/fc_v1_cycle_1_5）

    Returns:
        包含 sample.json 的子目录路径列表，按子目录名数字排序
    """
    # 目录不存在时返回空列表
    if not os.path.exists(train_data_dir):
        return []

    sample_dirs = []
    try:
        entries = os.listdir(train_data_dir)
    except OSError:
        return []

    for entry in entries:
        entry_path = os.path.join(train_data_dir, entry)
        # 仅检查直接子目录
        if os.path.isdir(entry_path):
            sample_path = os.path.join(entry_path, "sample.json")
            if os.path.exists(sample_path):
                sample_dirs.append(entry_path)

    # 按子目录名数字排序（如 "1", "2", "10" → 1, 2, 10）
    def _sort_key(path: str) -> int:
        dirname = os.path.basename(path)
        try:
            return int(dirname)
        except ValueError:
            # 非数字目录名排到最后
            return float('inf')

    sample_dirs.sort(key=_sort_key)
    return sample_dirs


def _create_temp_sample_dir(sample_data: list) -> str:
    """创建临时目录并将样本数据写入 sample.json

    用于兼容仅接收目录路径的 trainer 接口。

    Args:
        sample_data: 样本数据列表

    Returns:
        临时目录路径（调用方负责清理）
    """
    tmp_dir = tempfile.mkdtemp()
    sample_path = os.path.join(tmp_dir, "sample.json")
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False)
    return tmp_dir


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='AI 模型训练器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单轮次训练 fc_v1（默认10轮）
  python ai/train_main.py --model fc_v1 --sample-dir train_data/random_vs_random_1/1
  
  # 单轮次训练 fc_v1（自定义轮数）
  python ai/train_main.py --model fc_v1 --sample-dir train_data/random_vs_random_1/1 --epochs 20
  
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

    args = parser.parse_args()
    
    # 验证参数：四个模式互斥（--cycle、--self-play、--sample-dir、--train-from-dir 四选一）
    active_modes = []
    if args.cycle:
        active_modes.append('--cycle')
    if args.self_play:
        active_modes.append('--self-play')
    if args.sample_dir:
        active_modes.append('--sample-dir')
    if args.train_from_dir:
        active_modes.append('--train-from-dir')

    if len(active_modes) == 0:
        parser.error("请指定 --sample-dir、--cycle、--self-play 或 --train-from-dir 参数")
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
        
        print(f"开始训练 {args.model} 模型...")
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
    
    elif args.self_play:
        # 仅自我对弈生成数据模式
        print(f"开始 {args.model} 自我对弈数据生成...")
        print(f"参数:")
        print(f"  对局数 (games):      {args.games}")
        print(f"  奖励衰减率 (rate):   {args.rate}")

        try:
            from ai.self_play_generator import SelfPlayGenerator
            generator = SelfPlayGenerator(trainer.model)
            result = generator.generate(games=args.games, rate=args.rate)

            # 打印结果统计
            black_wins = sum(1 for r in result["results"] if r["winner"] == 1)
            white_wins = sum(1 for r in result["results"] if r["winner"] == 2)
            draws = sum(1 for r in result["results"] if r["winner"] == 3)
            print(f"\n对局统计: 黑胜 {black_wins}, 白胜 {white_wins}, 平局 {draws}")
            print(f"生成样本数: {result['sample_count']}")
            print(f"训练数据目录: {result['train_data_dir']}")
            print(f"（可使用 --train-from-dir {result['train_data_dir']} 训练模型）")

            # 如果指定了 --merge-output，自动合并样本
            if args.merge_output:
                print(f"\n正在自动合并样本到: {args.merge_output}")
                merge_result = merge_directories(
                    [result["train_data_dir"]],
                    args.merge_output
                )
                print(f"自动合并完成:")
                print(f"  合并文件数: {merge_result['total_files']}")
                print(f"  总样本数:   {merge_result['total_samples']}")
                print(f"  输出路径:   {merge_result['output_path']}")

        except Exception as e:
            print(f"自我对弈失败: {e}")
            sys.exit(1)

    elif args.train_from_dir:
        # 从已有数据训练模式（支持目录或 .json 文件）
        train_data_path = args.train_from_dir

        # 检查路径是否存在
        if not os.path.exists(train_data_path):
            print(f"错误: 路径不存在: {train_data_path}")
            sys.exit(1)

        # 判断是 .json 文件还是目录
        if os.path.isfile(train_data_path) and train_data_path.endswith('.json'):
            # 直接加载 .json 文件
            with open(train_data_path, 'r', encoding='utf-8') as f:
                sample_data = json.load(f)

            if not sample_data:
                print(f"错误: 训练数据为空: {train_data_path}")
                sys.exit(1)

            # 创建临时目录存放数据，兼容 trainer 接口
            tmp_dir = _create_temp_sample_dir(sample_data)
            sample_dirs = [tmp_dir]
            data_source_desc = f"文件: {train_data_path}"
        else:
            # 目录：自动发现 merge 文件和个体 sample.json
            loader = SampleLoader(trainer.device)
            sample_data = loader.discover_and_load(train_data_path)

            if not sample_data:
                print(f"错误: 未找到训练数据（{train_data_path} 下没有 merge 文件或包含 sample.json 的子目录）")
                sys.exit(1)

            # 创建临时目录存放合并后的数据，兼容 trainer 接口
            tmp_dir = _create_temp_sample_dir(sample_data)
            sample_dirs = [tmp_dir]
            data_source_desc = f"目录: {train_data_path}"

        print(f"开始训练 {args.model} 模型...")
        print(f"训练数据源: {data_source_desc}")
        print(f"加载样本数: {len(sample_data)}")
        print(f"训练参数:")
        print(f"  迭代数 (epochs):     {args.epochs}")
        if args.batch_size is not None:
            print(f"  批次大小 (batch):    {args.batch_size}")
            print(f"  数据打乱 (shuffle):  {args.shuffle}")
            print(f"  训练方法:            mini-batch")
        else:
            print(f"  批次大小 (batch):    N/A (全量更新)")
            print(f"  数据打乱 (shuffle):  N/A")
            print(f"  训练方法:            train_batch (全量更新)")

        try:
            # 根据 batch_size 选择训练方法
            if args.batch_size is not None and args.batch_size > 0:
                avg_loss = trainer.train_mini_batch(
                    sample_dirs,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    shuffle=args.shuffle
                )
            else:
                avg_loss = trainer.train_batch(
                    sample_dirs,
                    epochs=args.epochs
                )

            print(f"训练完成！平均损失: {avg_loss:.4f}")

            # 保存模型
            model_path = trainer.model.save()
            print(f"模型已保存到: {model_path}")

        except Exception as e:
            print(f"训练失败: {e}")
            sys.exit(1)

    elif args.cycle:
        # 训练周期
        print(f"开始 {args.model} 训练周期...")
        print(f"训练参数:")
        print(f"  轮次数 (rounds):     {args.rounds}")
        print(f"  每轮对局数 (games):  {args.games}")
        print(f"  每轮迭代数 (epochs): {args.epochs}")
        print(f"  奖励衰减率 (rate):   {args.rate}")
        if args.batch_size is not None:
            print(f"  批次大小 (batch):    {args.batch_size}")
            print(f"  数据打乱 (shuffle):  {args.shuffle}")
            print(f"  训练方法:            mini-batch")
        else:
            print(f"  批次大小 (batch):    N/A (全量更新)")
            print(f"  数据打乱 (shuffle):  N/A")
            print(f"  训练方法:            train_batch (全量更新)")
        if args.lr_scheduler:
            print(f"  学习率调度:          {args.lr_scheduler}")
            if args.lr_scheduler == 'step':
                print(f"    衰减间隔:          {args.lr_step_size} epoch")
                print(f"    衰减系数:          {args.lr_gamma}")
            else:
                print(f"    耐心值:            {args.lr_patience} epoch")
                print(f"    衰减系数:          {args.lr_factor}")
        else:
            print(f"  学习率调度:          无（固定学习率）")

        try:
            # 执行训练周期
            trainer.train_cycle(
                rounds=args.rounds,
                games=args.games,
                epochs=args.epochs,
                rate=args.rate,
                batch_size=args.batch_size,
                shuffle=args.shuffle
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