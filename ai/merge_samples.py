"""训练样本合并脚本

独立运行，用于将分散的 sample.json 合并为单个大文件，减少训练时 I/O 开销。
"""

import argparse
import json
import os
import sys
from typing import List


def discover_sample_files(data_dir: str) -> List[str]:
    """递归发现目录下所有 sample.json 文件路径

    遍历 data_dir 及其所有子目录，收集名为 sample.json 的文件，
    忽略任何以 merge_ 开头的 .json 文件。

    Args:
        data_dir: 数据目录路径

    Returns:
        sample.json 文件路径列表，按路径排序
    """
    sample_files = []

    # 使用 os.walk 递归遍历所有子目录
    for root, _dirs, files in os.walk(data_dir):
        for filename in files:
            if filename == "sample.json":
                filepath = os.path.join(root, filename)
                sample_files.append(filepath)
            elif filename.startswith("merge_") and filename.endswith(".json"):
                # 明确忽略 merge_*.json 文件，避免被意外收集
                continue

    # 按路径字符串排序，保证输出结果稳定可预测
    sample_files.sort()
    return sample_files


def merge_sample_files(sample_files: List[str], output_path: str) -> dict:
    """将多个 sample.json 文件合并为一个文件

    逐个读取 sample_files 中的 JSON 数据，合并为一个列表后写入 output_path。
    输出使用紧凑格式（无缩进），减少文件体积。

    Args:
        sample_files: sample.json 文件路径列表
        output_path: 合并后的输出文件路径

    Returns:
        {"total_files": int, "total_samples": int, "output_path": str}
    """
    all_data = []

    # 逐个读取并合并数据
    for sample_file in sample_files:
        with open(sample_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        all_data.extend(data)

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 使用紧凑格式写入（无 indent，减少文件大小）
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False)

    return {
        "total_files": len(sample_files),
        "total_samples": len(all_data),
        "output_path": output_path,
    }


def merge_directories(directories: List[str], output_path: str) -> dict:
    """将多个目录下的 sample.json 合并为一个文件

    先在每个目录中递归发现所有 sample.json，再统一合并。

    Args:
        directories: 数据目录路径列表
        output_path: 合并后的输出文件路径

    Returns:
        {"total_files": int, "total_samples": int, "output_path": str}
    """
    all_sample_files = []

    # 遍历每个目录，收集其中的 sample.json
    for directory in directories:
        sample_files = discover_sample_files(directory)
        all_sample_files.extend(sample_files)

    # 去重并保持排序（同一文件可能通过不同目录被发现）
    # 使用 dict.fromkeys 保证顺序的同时去重
    all_sample_files = list(dict.fromkeys(all_sample_files))

    return merge_sample_files(all_sample_files, output_path)


def main():
    """CLI 入口"""
    parser = argparse.ArgumentParser(
        description='合并多个 sample.json 为单个大文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 合并单个目录下的所有 sample.json
  python ai/merge_samples.py ai/train_data/fc_v1_self_play_1 -o ai/train_data/merge_fc_v1.json

  # 合并多个目录
  python ai/merge_samples.py dir1 dir2 dir3 -o merged.json
        """
    )

    parser.add_argument(
        'directories',
        nargs='+',
        help='一个或多个包含 sample.json 的数据目录'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='合并后的输出文件路径（必需）'
    )

    args = parser.parse_args()

    # 校验目录是否存在
    for directory in args.directories:
        if not os.path.exists(directory):
            print(f"错误: 目录不存在: {directory}")
            sys.exit(1)
        if not os.path.isdir(directory):
            print(f"错误: 不是目录: {directory}")
            sys.exit(1)

    # 执行合并
    result = merge_directories(args.directories, args.output)

    # 打印统计结果
    print(f"合并完成:")
    print(f"  合并文件数: {result['total_files']}")
    print(f"  总样本数:   {result['total_samples']}")
    print(f"  输出路径:   {result['output_path']}")


if __name__ == "__main__":
    main()
