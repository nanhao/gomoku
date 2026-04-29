"""训练运行器模块

从 train_main.py 提取的三种运行模式业务逻辑，使 train_main.py 仅保留
argparse 解析、参数校验和函数路由。

提供的公共函数：
    - run_self_play(args, trainer): 自我对弈数据生成模式
    - run_train_from_dir(args, trainer): 从已有数据训练模式
    - run_cycle(args, trainer): 训练周期模式
    - discover_sample_dirs(train_data_dir): 发现含 sample.json 的子目录
"""

import sys
import os
import json
import tempfile
import shutil

from ai.sample_loader import SampleLoader
from ai.merge_samples import merge_directories
from ai.self_play_generator import SelfPlayGenerator


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
        临时目录路径（调用方应在 finally 块中清理）
    """
    tmp_dir = tempfile.mkdtemp()
    sample_path = os.path.join(tmp_dir, "sample.json")
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False)
    return tmp_dir


def run_self_play(args, trainer) -> None:
    """自我对弈数据生成模式

    仅执行自我对弈生成训练数据，不训练模型。
    可选 --merge-output 将生成的样本自动合并到指定文件。

    Args:
        args: 已解析的命令行参数（Namespace）
        trainer: 训练器实例

    Raises:
        SystemExit: 自我对弈失败时退出
    """
    print(f"开始 {args.model} 自我对弈数据生成...")
    print(f"参数:")
    print(f"  对局数 (games):      {args.games}")
    print(f"  奖励衰减率 (rate):   {args.rate}")

    try:
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


def run_train_from_dir(args, trainer) -> None:
    """从已有数据训练模式

    支持直接指定 .json 文件或目录路径。目录路径会自动发现其中的
    merge 文件和 sample.json 子目录。训练完成后自动清理临时目录。

    Args:
        args: 已解析的命令行参数（Namespace）
        trainer: 训练器实例

    Raises:
        SystemExit: 路径不存在、数据为空或训练失败时退出
    """
    train_data_path = args.train_from_dir
    tmp_dir = None  # 跟踪临时目录，确保 finally 中清理

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
            print(f"错误: 未找到训练数据（{train_data_path} 下没有 .json 文件）")
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
    if args.val_ratio > 0:
        print(f"  验证集比例 (val_ratio): {args.val_ratio}")
    else:
        print(f"  验证集比例 (val_ratio): 0.0 (不分割)")

    try:
        # 自动选择训练方法（提供 batch_size 时使用 mini-batch，否则全量更新）
        avg_loss = trainer.train_auto(
            sample_dirs,
            epochs=args.epochs,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            dense_log=True,
            val_ratio=args.val_ratio
        )

        print(f"训练完成！平均损失: {avg_loss:.4f}")

        # 保存模型
        model_path = trainer.model.save()
        print(f"模型已保存到: {model_path}")

    except Exception as e:
        print(f"训练失败: {e}")
        sys.exit(1)
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def run_cycle(args, trainer) -> None:
    """训练周期模式

    执行完整的训练周期：自我对弈 → 评价 → 训练，循环多轮。

    Args:
        args: 已解析的命令行参数（Namespace）
        trainer: 训练器实例

    Raises:
        SystemExit: 功能未实现或训练失败时退出
    """
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
    if args.val_ratio > 0:
        print(f"  验证集比例 (val_ratio): {args.val_ratio}")
    else:
        print(f"  验证集比例 (val_ratio): 0.0 (不分割)")

    try:
        # 执行训练周期
        trainer.train_cycle(
            rounds=args.rounds,
            games=args.games,
            epochs=args.epochs,
            rate=args.rate,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            val_ratio=args.val_ratio
        )

        print("训练周期完成！")

    except NotImplementedError as e:
        print(f"功能未实现: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"训练失败: {e}")
        sys.exit(1)
