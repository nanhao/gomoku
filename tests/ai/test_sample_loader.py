"""sample_loader.py 测试

测试 SampleLoader 的数据加载、合并和目录发现功能。
"""

import json
import os
import sys
import tempfile
import unittest

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch

from ai.sample_loader import SampleLoader, compute_masked_l1_loss, discover_all_data
from tests.ai.helpers import create_sample_dir


class TestComputeMaskedL1Loss(unittest.TestCase):
    """测试 compute_masked_l1_loss 独立函数"""

    def test_single_mask_position(self):
        """单个 mask=1 位置时正确计算 L1 loss"""
        estimated = [0.0] * 225
        sample = [0.0] * 225
        mask = [0] * 225
        estimated[0] = 100.0
        sample[0] = 50.0
        mask[0] = 1

        loss = compute_masked_l1_loss(estimated, sample, mask)
        # |100 - 50| / 1 = 50
        self.assertEqual(loss, 50.0)

    def test_multiple_mask_positions(self):
        """多个 mask=1 位置时正确计算平均 L1 loss"""
        estimated = [0.0] * 225
        sample = [0.0] * 225
        mask = [0] * 225
        estimated[0] = 0.7
        sample[0] = 0.5
        mask[0] = 1
        estimated[1] = 0.3
        sample[1] = 0.1
        mask[1] = 1

        loss = compute_masked_l1_loss(estimated, sample, mask)
        # (|0.7-0.5| + |0.3-0.1|) / 2 = (0.2 + 0.2) / 2 = 0.2
        self.assertAlmostEqual(loss, 0.2, places=6)

    def test_zero_mask_returns_zero(self):
        """mask 全为 0 时返回 0.0"""
        estimated = [100.0] * 225
        sample = [0.0] * 225
        mask = [0] * 225

        loss = compute_masked_l1_loss(estimated, sample, mask)
        self.assertEqual(loss, 0.0)

    def test_identical_values_zero_loss(self):
        """估计值与样本值完全相同时 loss 为 0"""
        values = [0.5] * 225
        mask = [1] * 225

        loss = compute_masked_l1_loss(values, values, mask)
        self.assertEqual(loss, 0.0)


class TestLoadMergeFile(unittest.TestCase):
    """测试 load_merge_file 方法"""

    def test_load_merge_file(self):
        """加载 merge 文件返回正确数据"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            merge_path = os.path.join(tmp_dir, "merge_cnn_v2.json")
            sample_data = [
                {"input": [1.0] * 226, "output": [0.0] * 225, "mask": [1] * 225},
                {"input": [2.0] * 226, "output": [0.0] * 225, "mask": [1] * 225},
            ]
            with open(merge_path, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f)

            loader = SampleLoader(torch.device('cpu'))
            result = loader.load_merge_file(merge_path)

            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["input"], [1.0] * 226)
            self.assertEqual(result[1]["input"], [2.0] * 226)

    def test_load_merge_file_not_found(self):
        """merge 文件不存在时抛出 FileNotFoundError"""
        loader = SampleLoader(torch.device('cpu'))
        with self.assertRaises(FileNotFoundError):
            loader.load_merge_file("/nonexistent/merge_test.json")


class TestDiscoverAllData(unittest.TestCase):
    """测试 discover_all_data 函数"""

    def test_discover_all_data_json_only(self):
        """目录仅有 .json 文件时正确发现（不限 merge_ 前缀）"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 非 merge_ 前缀的 json 文件也应被发现
            json_path = os.path.join(tmp_dir, "chong4.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump([], f)

            result = discover_all_data(tmp_dir)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], json_path)

    def test_discover_all_data_merge_prefix(self):
        """merge_ 前缀的 json 文件同样被发现"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            merge_path = os.path.join(tmp_dir, "merge_cnn_v2.json")
            with open(merge_path, 'w', encoding='utf-8') as f:
                json.dump([], f)

            result = discover_all_data(tmp_dir)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], merge_path)

    def test_discover_all_data_recursive(self):
        """递归发现子目录中的 json 文件"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 根目录 json 文件
            root_json = os.path.join(tmp_dir, "data1.json")
            with open(root_json, 'w', encoding='utf-8') as f:
                json.dump([], f)

            # 一级子目录中的 sample.json
            sub_dir = os.path.join(tmp_dir, "round1")
            os.makedirs(sub_dir)
            sub_json = os.path.join(sub_dir, "sample.json")
            with open(sub_json, 'w', encoding='utf-8') as f:
                json.dump([], f)

            # 二级子目录中的 json 文件
            deep_dir = os.path.join(sub_dir, "extra")
            os.makedirs(deep_dir)
            deep_json = os.path.join(deep_dir, "bonus.json")
            with open(deep_json, 'w', encoding='utf-8') as f:
                json.dump([], f)

            result = discover_all_data(tmp_dir)
            # 应找到全部 3 个 json 文件
            self.assertEqual(len(result), 3)
            self.assertIn(root_json, result)
            self.assertIn(sub_json, result)
            self.assertIn(deep_json, result)

    def test_discover_all_data_ignores_non_json(self):
        """非 .json 文件被忽略"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 创建非 json 文件
            txt_path = os.path.join(tmp_dir, "readme.txt")
            with open(txt_path, 'w') as f:
                f.write("not data")
            pyc_path = os.path.join(tmp_dir, "__pycache__.pyc")
            with open(pyc_path, 'w') as f:
                f.write("")
            # 创建 json 文件
            json_path = os.path.join(tmp_dir, "data.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump([], f)

            result = discover_all_data(tmp_dir)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], json_path)

    def test_discover_all_data_empty(self):
        """空目录返回空列表"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = discover_all_data(tmp_dir)
            self.assertEqual(result, [])

    def test_discover_all_data_sorted(self):
        """返回结果按路径排序"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 按反向顺序创建 json 文件
            for name in ["c.json", "a.json", "b.json"]:
                with open(os.path.join(tmp_dir, name), 'w', encoding='utf-8') as f:
                    json.dump([], f)

            result = discover_all_data(tmp_dir)
            basenames = [os.path.basename(p) for p in result]
            self.assertEqual(basenames, ["a.json", "b.json", "c.json"])


class TestDiscoverAndLoad(unittest.TestCase):
    """测试 discover_and_load 方法"""

    def test_discover_and_load_json_files(self):
        """目录下 .json 文件全部被加载（不限 merge_ 前缀）"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 非 merge_ 前缀文件
            json_path = os.path.join(tmp_dir, "chong4.json")
            sample_data = [
                {"input": [1.0] * 226, "output": [0.0] * 225, "mask": [1] * 225},
            ]
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f)

            loader = SampleLoader(torch.device('cpu'))
            result = loader.discover_and_load(tmp_dir)

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["input"], [1.0] * 226)

    def test_discover_and_load_recursive(self):
        """递归加载子目录中的 json 文件"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 根目录 json
            root_path = os.path.join(tmp_dir, "root.json")
            with open(root_path, 'w', encoding='utf-8') as f:
                json.dump([{"input": [1.0] * 226, "output": [0.0] * 225, "mask": [1] * 225}], f)

            # 子目录 sample.json
            sample_dir = create_sample_dir(tmp_dir, "1", n_items=2)

            loader = SampleLoader(torch.device('cpu'))
            result = loader.discover_and_load(tmp_dir)

            # 根目录 1 条 + 子目录 2 条 = 3 条
            self.assertEqual(len(result), 3)

    def test_discover_and_load_both_prefixes(self):
        """merge_ 前缀和非 merge_ 前缀文件均被加载"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # merge_ 前缀
            merge_path = os.path.join(tmp_dir, "merge_cnn_v2.json")
            with open(merge_path, 'w', encoding='utf-8') as f:
                json.dump([{"input": [1.0] * 226, "output": [0.0] * 225, "mask": [1] * 225}], f)

            # 非 merge_ 前缀
            other_path = os.path.join(tmp_dir, "chong4.json")
            with open(other_path, 'w', encoding='utf-8') as f:
                json.dump([{"input": [2.0] * 226, "output": [0.0] * 225, "mask": [1] * 225}], f)

            loader = SampleLoader(torch.device('cpu'))
            result = loader.discover_and_load(tmp_dir)

            self.assertEqual(len(result), 2)

    def test_discover_and_load_empty(self):
        """空目录返回空列表"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            loader = SampleLoader(torch.device('cpu'))
            result = loader.discover_and_load(tmp_dir)
            self.assertEqual(result, [])


class TestSampleLoaderExistingMethods(unittest.TestCase):
    """测试 SampleLoader 现有方法仍正常工作"""

    def test_load_single(self):
        """load 方法加载单个文件"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = create_sample_dir(tmp_dir, "sample", n_items=1)
            sample_path = os.path.join(sample_dir, "sample.json")

            loader = SampleLoader(torch.device('cpu'))
            result = loader.load(sample_path)
            self.assertEqual(len(result), 1)

    def test_load_multiple(self):
        """load_multiple 加载多个目录"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dir1 = create_sample_dir(tmp_dir, "1", n_items=1)
            dir2 = create_sample_dir(tmp_dir, "2", n_items=2)

            loader = SampleLoader(torch.device('cpu'))
            result = loader.load_multiple([dir1, dir2])
            self.assertEqual(len(result), 3)

    def test_prepare_tensors(self):
        """prepare_tensors 转换为 PyTorch 张量"""
        sample_data = [
            {"input": [1.0] * 226, "output": [0.0] * 225, "mask": [1] * 225},
        ]

        loader = SampleLoader(torch.device('cpu'))
        inputs, targets, masks = loader.prepare_tensors(sample_data)

        self.assertEqual(inputs.shape, (1, 226))
        self.assertEqual(targets.shape, (1, 225))
        self.assertEqual(masks.shape, (1, 225))


if __name__ == '__main__':
    unittest.main()
