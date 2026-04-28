"""SampleLoader.load_and_split() 单元测试

测试训练数据按样本级别分割为训练集和验证集的功能。
覆盖比例正确性、无重叠、数据完整性、可复现性、边界条件等。
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch

from ai.sample_loader import SampleLoader


def _create_sample_dirs(tmp_dir: str, n_dirs: int, n_items_per_dir: int = 2) -> list:
    """创建多个包含 sample.json 的测试目录

    Args:
        tmp_dir: 临时根目录
        n_dirs: 目录数量
        n_items_per_dir: 每个目录的样本数量

    Returns:
        目录路径列表
    """
    dirs = []
    for d in range(n_dirs):
        sample_dir = os.path.join(tmp_dir, f"sample_{d}")
        os.makedirs(sample_dir, exist_ok=True)
        sample_data = []
        for i in range(n_items_per_dir):
            sample_data.append({
                "input": [float(d * 100 + i)] * 226,
                "output": [0.0] * 225,
                "mask": [1] * 225
            })
        with open(os.path.join(sample_dir, "sample.json"), 'w', encoding='utf-8') as f:
            json.dump(sample_data, f)
        dirs.append(sample_dir)
    return dirs


class TestLoadAndSplitRatio(unittest.TestCase):
    """测试分割比例正确性"""

    def test_split_ratio_correct(self):
        """验证分割比例大致正确（允许 ±1 误差）"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dirs = _create_sample_dirs(tmp_dir, n_dirs=5, n_items_per_dir=10)
            loader = SampleLoader(torch.device('cpu'))

            train_data, val_data = loader.load_and_split(dirs, val_ratio=0.2, seed=42)
            total = len(train_data) + len(val_data)
            self.assertEqual(total, 50)
            # 20% 即 10 个样本，允许 ±1 误差
            self.assertAlmostEqual(len(val_data), 10, delta=1)
            self.assertAlmostEqual(len(train_data), 40, delta=1)

    def test_split_ratio_30_percent(self):
        """val_ratio=0.3 时验证集约占 30%"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dirs = _create_sample_dirs(tmp_dir, n_dirs=4, n_items_per_dir=10)
            loader = SampleLoader(torch.device('cpu'))

            train_data, val_data = loader.load_and_split(dirs, val_ratio=0.3, seed=42)
            total = len(train_data) + len(val_data)
            self.assertEqual(total, 40)
            self.assertAlmostEqual(len(val_data), 12, delta=1)


class TestLoadAndSplitDataIntegrity(unittest.TestCase):
    """测试数据完整性"""

    def test_split_no_overlap(self):
        """训练集和验证集没有重叠样本"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dirs = _create_sample_dirs(tmp_dir, n_dirs=3, n_items_per_dir=5)
            loader = SampleLoader(torch.device('cpu'))

            train_data, val_data = loader.load_and_split(dirs, val_ratio=0.3, seed=42)
            # 用 input 第一个值作为唯一标识（测试数据构造时确保了唯一性）
            train_ids = {tuple(item["input"]) for item in train_data}
            val_ids = {tuple(item["input"]) for item in val_data}
            self.assertEqual(len(train_ids & val_ids), 0)

    def test_split_all_data_preserved(self):
        """训练集 + 验证集 = 全部数据"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dirs = _create_sample_dirs(tmp_dir, n_dirs=3, n_items_per_dir=5)
            loader = SampleLoader(torch.device('cpu'))

            train_data, val_data = loader.load_and_split(dirs, val_ratio=0.3, seed=42)
            all_ids = {tuple(item["input"]) for item in (train_data + val_data)}
            # 原始数据
            all_original = loader.load_multiple(dirs)
            original_ids = {tuple(item["input"]) for item in all_original}
            self.assertEqual(all_ids, original_ids)
            self.assertEqual(len(train_data) + len(val_data), len(all_original))


class TestLoadAndSplitReproducibility(unittest.TestCase):
    """测试可复现性"""

    def test_split_reproducible_with_same_seed(self):
        """相同 seed 产生相同分割"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dirs = _create_sample_dirs(tmp_dir, n_dirs=3, n_items_per_dir=5)
            loader = SampleLoader(torch.device('cpu'))

            train1, val1 = loader.load_and_split(dirs, val_ratio=0.3, seed=42)
            train2, val2 = loader.load_and_split(dirs, val_ratio=0.3, seed=42)

            self.assertEqual(
                [tuple(item["input"]) for item in train1],
                [tuple(item["input"]) for item in train2]
            )
            self.assertEqual(
                [tuple(item["input"]) for item in val1],
                [tuple(item["input"]) for item in val2]
            )

    def test_split_different_with_different_seed(self):
        """不同 seed 产生不同分割"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dirs = _create_sample_dirs(tmp_dir, n_dirs=5, n_items_per_dir=10)
            loader = SampleLoader(torch.device('cpu'))

            train1, val1 = loader.load_and_split(dirs, val_ratio=0.3, seed=42)
            train2, val2 = loader.load_and_split(dirs, val_ratio=0.3, seed=99)

            # 至少有一个样本的分组不同
            ids1_train = {tuple(item["input"]) for item in train1}
            ids2_train = {tuple(item["input"]) for item in train2}
            self.assertNotEqual(ids1_train, ids2_train)


class TestLoadAndSplitEdgeCases(unittest.TestCase):
    """测试边界条件"""

    def test_split_val_ratio_zero_returns_all_train(self):
        """val_ratio=0 时所有数据归训练集，验证集为空"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dirs = _create_sample_dirs(tmp_dir, n_dirs=2, n_items_per_dir=5)
            loader = SampleLoader(torch.device('cpu'))

            train_data, val_data = loader.load_and_split(dirs, val_ratio=0.0, seed=42)
            self.assertEqual(len(train_data), 10)
            self.assertEqual(len(val_data), 0)

    def test_split_val_ratio_invalid_raises(self):
        """val_ratio < 0 或 >= 1 时抛出 ValueError"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dirs = _create_sample_dirs(tmp_dir, n_dirs=1, n_items_per_dir=5)
            loader = SampleLoader(torch.device('cpu'))

            with self.assertRaises(ValueError):
                loader.load_and_split(dirs, val_ratio=-0.1)
            with self.assertRaises(ValueError):
                loader.load_and_split(dirs, val_ratio=1.0)
            with self.assertRaises(ValueError):
                loader.load_and_split(dirs, val_ratio=1.5)

    def test_split_single_sample(self):
        """只有 1 个样本时，归训练集，验证集为空"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dirs = _create_sample_dirs(tmp_dir, n_dirs=1, n_items_per_dir=1)
            loader = SampleLoader(torch.device('cpu'))

            train_data, val_data = loader.load_and_split(dirs, val_ratio=0.2, seed=42)
            self.assertEqual(len(train_data), 1)
            self.assertEqual(len(val_data), 0)

    def test_split_file_not_found(self):
        """某个目录下不存在 sample.json 时抛出 FileNotFoundError"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            valid_dir = _create_sample_dirs(tmp_dir, n_dirs=1, n_items_per_dir=2)[0]
            invalid_dir = os.path.join(tmp_dir, "no_sample")
            os.makedirs(invalid_dir, exist_ok=True)
            loader = SampleLoader(torch.device('cpu'))

            with self.assertRaises(FileNotFoundError):
                loader.load_and_split([valid_dir, invalid_dir], val_ratio=0.2)


if __name__ == '__main__':
    unittest.main()
