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

from ai.sample_loader import SampleLoader, discover_all_data
from tests.ai.helpers import create_sample_dir


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

    def test_discover_all_data_merge_only(self):
        """目录仅有 merge 文件时正确发现"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            merge_path = os.path.join(tmp_dir, "merge_cnn_v2.json")
            with open(merge_path, 'w', encoding='utf-8') as f:
                json.dump([], f)

            result = discover_all_data(tmp_dir)
            self.assertEqual(len(result["merge_files"]), 1)
            self.assertEqual(result["merge_files"][0], merge_path)
            self.assertEqual(len(result["sample_dirs"]), 0)

    def test_discover_all_data_samples_only(self):
        """目录仅有个体 sample.json 时正确发现"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = create_sample_dir(tmp_dir, "1", n_items=2)

            result = discover_all_data(tmp_dir)
            self.assertEqual(len(result["merge_files"]), 0)
            self.assertEqual(len(result["sample_dirs"]), 1)
            self.assertEqual(result["sample_dirs"][0], sample_dir)

    def test_discover_all_data_both(self):
        """目录同时有 merge 文件和个体 sample.json 时全部发现"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            merge_path = os.path.join(tmp_dir, "merge_fc_v1.json")
            with open(merge_path, 'w', encoding='utf-8') as f:
                json.dump([], f)
            sample_dir = create_sample_dir(tmp_dir, "1", n_items=1)

            result = discover_all_data(tmp_dir)
            self.assertEqual(len(result["merge_files"]), 1)
            self.assertEqual(len(result["sample_dirs"]), 1)

    def test_discover_all_data_empty(self):
        """空目录返回空列表"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = discover_all_data(tmp_dir)
            self.assertEqual(result["merge_files"], [])
            self.assertEqual(result["sample_dirs"], [])

    def test_discover_all_data_sorted(self):
        """返回结果按名称排序"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 按反向顺序创建，验证排序
            create_sample_dir(tmp_dir, "3", n_items=1)
            create_sample_dir(tmp_dir, "1", n_items=1)
            create_sample_dir(tmp_dir, "2", n_items=1)

            result = discover_all_data(tmp_dir)
            basenames = [os.path.basename(p) for p in result["sample_dirs"]]
            self.assertEqual(basenames, ["1", "2", "3"])

    def test_discover_all_data_ignores_merge_in_subdir(self):
        """子目录中的 merge_*.json 不应被收集"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sub_dir = os.path.join(tmp_dir, "sub")
            os.makedirs(sub_dir)
            # 子目录中创建 merge 文件
            with open(os.path.join(sub_dir, "merge_test.json"), 'w', encoding='utf-8') as f:
                json.dump([], f)
            # 根目录创建有效 merge 文件
            merge_path = os.path.join(tmp_dir, "merge_valid.json")
            with open(merge_path, 'w', encoding='utf-8') as f:
                json.dump([], f)

            result = discover_all_data(tmp_dir)
            # 只收集根目录下的 merge_*.json
            self.assertEqual(len(result["merge_files"]), 1)
            self.assertEqual(result["merge_files"][0], merge_path)
            self.assertEqual(len(result["sample_dirs"]), 0)


class TestDiscoverAndLoad(unittest.TestCase):
    """测试 discover_and_load 方法"""

    def test_discover_and_load_merge_only(self):
        """目录仅有 merge 文件时正确加载"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            merge_path = os.path.join(tmp_dir, "merge_cnn_v2.json")
            sample_data = [
                {"input": [1.0] * 226, "output": [0.0] * 225, "mask": [1] * 225},
            ]
            with open(merge_path, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f)

            loader = SampleLoader(torch.device('cpu'))
            result = loader.discover_and_load(tmp_dir)

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["input"], [1.0] * 226)

    def test_discover_and_load_samples_only(self):
        """目录仅有个体 sample.json 时正确加载"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            create_sample_dir(tmp_dir, "1", n_items=2)

            loader = SampleLoader(torch.device('cpu'))
            result = loader.discover_and_load(tmp_dir)

            self.assertEqual(len(result), 2)

    def test_discover_and_load_both(self):
        """目录同时有 merge 文件和个体 sample.json 时全部加载"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # merge 文件含 1 条
            merge_path = os.path.join(tmp_dir, "merge_cnn_v2.json")
            with open(merge_path, 'w', encoding='utf-8') as f:
                json.dump([{"input": [1.0] * 226, "output": [0.0] * 225, "mask": [1] * 225}], f)
            # 个体目录含 2 条
            create_sample_dir(tmp_dir, "1", n_items=2)

            loader = SampleLoader(torch.device('cpu'))
            result = loader.discover_and_load(tmp_dir)

            self.assertEqual(len(result), 3)

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
