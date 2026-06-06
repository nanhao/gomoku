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

from ai.sample_loader import SampleLoader, compute_masked_kl_loss, discover_all_data
from tests.ai.helpers import create_sample_dir


class TestComputeMaskedKLLoss(unittest.TestCase):
    """测试 compute_masked_kl_loss 独立函数"""

    def test_identical_distribution_loss_is_zero(self):
        """T1-01: 完全相同分布 KL 散度为 0"""
        import math
        # log_estimated = log([0.5, 0.5, 0, ..., 0])
        log_estimated = [0.0] * 225
        log_estimated[0] = math.log(0.5)
        log_estimated[1] = math.log(0.5)
        target = [0.0] * 225
        target[0] = 0.5
        target[1] = 0.5
        mask = [0] * 225
        mask[0] = 1
        mask[1] = 1

        loss = compute_masked_kl_loss(log_estimated, target, mask)
        self.assertAlmostEqual(loss, 0.0, places=5)

    def test_known_distribution_kl_divergence(self):
        """T1-02: 已知分布的 masked 平均 KL 散度 ≈ 0.0436"""
        import math
        # KL([0.5,0.5] || [0.7,0.3]) 总和 = 0.5*(ln0.5-ln0.7) + 0.5*(ln0.5-ln0.3) ≈ 0.0872
        # 但 compute_masked_kl_loss 按 mask=1 位置数 (2个) 取平均，返回 ≈ 0.0872/2 ≈ 0.0436
        log_estimated = [0.0] * 225
        log_estimated[0] = math.log(0.7)
        log_estimated[1] = math.log(0.3)
        target = [0.0] * 225
        target[0] = 0.5
        target[1] = 0.5
        mask = [0] * 225
        mask[0] = 1
        mask[1] = 1

        loss = compute_masked_kl_loss(log_estimated, target, mask)
        self.assertAlmostEqual(loss, 0.0436, places=3)

    def test_zero_mask_returns_zero(self):
        """T1-03: mask 全为 0 时返回 0.0"""
        log_estimated = [0.0] * 225
        target = [0.0] * 225
        mask = [0] * 225

        loss = compute_masked_kl_loss(log_estimated, target, mask)
        self.assertEqual(loss, 0.0)

    def test_target_with_zero_items(self):
        """T1-04: target 含零项时，该位置贡献为 0"""
        import math
        # target[0]=0 跳过 log(0)，target[1]=1 贡献为 log(1)-log(0.2) ≈ 1.6094
        log_estimated = [0.0] * 225
        log_estimated[0] = math.log(0.8)
        log_estimated[1] = math.log(0.2)
        target = [0.0] * 225
        target[0] = 0.0
        target[1] = 1.0
        mask = [0] * 225
        mask[0] = 1
        mask[1] = 1

        loss = compute_masked_kl_loss(log_estimated, target, mask)
        # 只有 target[1]=1.0 贡献: 1.0 * (log(1.0) - log(0.2)) ≈ 1.6094
        # 除以 count=2, 平均 ≈ 0.8047
        expected = 1.0 * (math.log(1.0) - math.log(0.2)) / 2
        self.assertAlmostEqual(loss, expected, places=3)

    def test_kl_divergence_asymmetry(self):
        """T1-05: KL 散度非对称性——交换 estimated 和 target 结果不同"""
        import math
        log_est_a = [0.0] * 225
        log_est_a[0] = math.log(0.7)
        log_est_a[1] = math.log(0.3)
        target_a = [0.0] * 225
        target_a[0] = 0.5
        target_a[1] = 0.5
        mask = [0] * 225
        mask[0] = 1
        mask[1] = 1

        loss_a = compute_masked_kl_loss(log_est_a, target_a, mask)

        # 交换: log_estimated 用 [0.5,0.5] 的 log, target 用 [0.7,0.3]
        log_est_b = [0.0] * 225
        log_est_b[0] = math.log(0.5)
        log_est_b[1] = math.log(0.5)
        target_b = [0.0] * 225
        target_b[0] = 0.7
        target_b[1] = 0.3

        loss_b = compute_masked_kl_loss(log_est_b, target_b, mask)

        # KL 散度非对称，两者不应相等
        self.assertNotAlmostEqual(loss_a, loss_b, places=3)

    def test_consistent_with_masked_loss(self):
        """T1-06: compute_masked_kl_loss 与 _masked_loss 结果一致"""
        import math
        log_estimated = [math.log(0.7), math.log(0.3)] + [0.0] * 223
        target = [0.5, 0.5] + [0.0] * 223
        mask = [1, 1] + [0] * 223

        py_loss = compute_masked_kl_loss(log_estimated, target, mask)

        # 用 PyTorch 计算 _masked_loss 对比
        from ai.models.fc_v1.trainer import FCv1Trainer
        trainer = FCv1Trainer(fresh=True)
        outputs_t = torch.tensor([[math.log(0.7), math.log(0.3)] + [0.0] * 223],
                                 dtype=torch.float32).to(trainer.device)
        targets_t = torch.tensor([[0.5, 0.5] + [0.0] * 223],
                                 dtype=torch.float32).to(trainer.device)
        masks_t = torch.tensor([[1, 1] + [0] * 223],
                                dtype=torch.float32).to(trainer.device)
        tensor_loss = trainer._masked_loss(outputs_t, targets_t, masks_t)

        self.assertAlmostEqual(py_loss, tensor_loss.item(), places=3,
                               msg="纯Python KL 散度与 _masked_loss 应一致")

    def test_positive_probability_raises_valueerror(self):
        """运行时防御: log_estimated 中存在 > 0 元素（概率值特征）时 raise ValueError"""
        # 误传概率值而非 log-probability
        log_estimated = [0.7, 0.3] + [0.0] * 223  # 0.7 > 0，应为 log 值
        target = [0.5, 0.5] + [0.0] * 223
        mask = [1, 1] + [0] * 223

        with self.assertRaises(ValueError):
            compute_masked_kl_loss(log_estimated, target, mask)


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
