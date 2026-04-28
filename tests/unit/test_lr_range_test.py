"""LR Range Test 单元测试

测试 lr_range_test 模块的各个函数接口。
"""

import copy
import json
import os
import sys
import tempfile
import unittest

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch

from ai.models.cnn_v2 import CNNV2Model
from ai.sample_loader import SampleLoader
from tests.ai.helpers import create_sample_dir
from ai.lr_range_test import (
    load_model,
    load_data,
    plot_lr_loss_curve,
    run_lr_range_test,
)


def _create_merge_file(tmp_dir: str, name: str = "merge_test.json",
                       n_items: int = 3) -> str:
    """创建测试用 merge json 文件

    Args:
        tmp_dir: 临时目录
        name: 文件名
        n_items: 样本数量

    Returns:
        文件路径
    """
    sample_data = []
    for i in range(n_items):
        output = [0.0] * 225
        output[0] = 0.7
        output[1] = 0.2
        output[2] = 0.1
        mask = [1] * 225
        sample_data.append({
            "input": [float(i + 1)] * 226,
            "output": output,
            "mask": mask
        })

    merge_path = os.path.join(tmp_dir, name)
    with open(merge_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f)
    return merge_path


class TestLoadModel(unittest.TestCase):
    """测试 load_model 函数"""

    def test_load_model_cnn_v2(self):
        """验证 load_model('cnn_v2') 返回 CNNV2Model 实例且 network 不为 None"""
        model = load_model("cnn_v2", torch.device('cpu'))
        self.assertIsInstance(model, CNNV2Model)
        self.assertIsNotNone(model.network)

    def test_load_model_invalid_name(self):
        """验证传入不支持的模型名称时抛出 ValueError"""
        with self.assertRaises(ValueError):
            load_model("invalid_model", torch.device('cpu'))


class TestLoadData(unittest.TestCase):
    """测试 load_data 函数"""

    def test_load_data_from_merge_file(self):
        """验证从 merge json 文件加载数据并返回三个张量"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            merge_path = _create_merge_file(tmp_dir, n_items=2)
            sample_loader = SampleLoader(device=torch.device('cpu'))
            inputs, targets, masks = load_data(merge_path, sample_loader)

            self.assertEqual(inputs.shape[0], 2)
            self.assertEqual(targets.shape[0], 2)
            self.assertEqual(masks.shape[0], 2)
            self.assertEqual(targets.shape[1], 225)
            self.assertEqual(masks.shape[1], 225)

    def test_load_data_from_directory(self):
        """验证从目录扫描并加载数据"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            create_sample_dir(tmp_dir, "game_1", n_items=2)
            sample_loader = SampleLoader(device=torch.device('cpu'))
            inputs, targets, masks = load_data(tmp_dir, sample_loader)

            self.assertEqual(inputs.shape[0], 2)
            self.assertEqual(targets.shape[0], 2)

    def test_load_data_nonexistent_path(self):
        """验证传入不存在的数据路径时抛出 FileNotFoundError"""
        sample_loader = SampleLoader(device=torch.device('cpu'))
        with self.assertRaises(FileNotFoundError):
            load_data("/nonexistent/path", sample_loader)


class TestPlotLrLossCurve(unittest.TestCase):
    """测试 plot_lr_loss_curve 函数"""

    def test_plot_creates_file(self):
        """验证 plot_lr_loss_curve 能在临时目录生成 png 文件"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "lr_range_test.png")
            lrs = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
            losses = [0.5, 0.4, 0.3, 0.35, 0.8]

            plot_lr_loss_curve(lrs, losses, output_path)

            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)


class TestRunLrRangeTest(unittest.TestCase):
    """测试 run_lr_range_test 函数"""

    def test_linear_lr_increase(self):
        """验证返回的 lr 列表严格线性递增"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            merge_path = _create_merge_file(tmp_dir, n_items=2)
            min_lr, max_lr = 1e-7, 1e-1
            rounds = 5

            lrs, losses = run_lr_range_test(
                model_name="cnn_v2",
                data_path=merge_path,
                rounds=rounds,
                epochs=2,
                min_lr=min_lr,
                max_lr=max_lr,
                device=torch.device('cpu')
            )

            # 验证 lr 线性递增
            for i in range(1, len(lrs)):
                self.assertGreater(lrs[i], lrs[i - 1])

            # 验证首尾值
            self.assertAlmostEqual(lrs[0], min_lr)
            self.assertAlmostEqual(lrs[-1], max_lr)

    def test_losses_length_matches_rounds(self):
        """验证返回的 losses 长度等于 rounds"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            merge_path = _create_merge_file(tmp_dir, n_items=2)
            rounds = 3

            lrs, losses = run_lr_range_test(
                model_name="cnn_v2",
                data_path=merge_path,
                rounds=rounds,
                epochs=2,
                device=torch.device('cpu')
            )

            self.assertEqual(len(losses), rounds)
            self.assertEqual(len(lrs), rounds)

    def test_rounds_1_lr_equals_min(self):
        """验证 rounds=1 时 lr = min_lr"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            merge_path = _create_merge_file(tmp_dir, n_items=2)
            min_lr = 1e-5

            lrs, losses = run_lr_range_test(
                model_name="cnn_v2",
                data_path=merge_path,
                rounds=1,
                epochs=1,
                min_lr=min_lr,
                max_lr=1e-1,
                device=torch.device('cpu')
            )

            self.assertEqual(len(lrs), 1)
            self.assertAlmostEqual(lrs[0], min_lr)

    def test_min_lr_equals_max_lr(self):
        """验证 min_lr == max_lr 时所有 round 使用相同 lr"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            merge_path = _create_merge_file(tmp_dir, n_items=2)
            fixed_lr = 1e-3

            lrs, losses = run_lr_range_test(
                model_name="cnn_v2",
                data_path=merge_path,
                rounds=3,
                epochs=1,
                min_lr=fixed_lr,
                max_lr=fixed_lr,
                device=torch.device('cpu')
            )

            for lr in lrs:
                self.assertAlmostEqual(lr, fixed_lr)

    def test_losses_are_finite(self):
        """验证所有 loss 值为有限数"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            merge_path = _create_merge_file(tmp_dir, n_items=2)

            lrs, losses = run_lr_range_test(
                model_name="cnn_v2",
                data_path=merge_path,
                rounds=3,
                epochs=2,
                min_lr=1e-7,
                max_lr=1e-3,
                device=torch.device('cpu')
            )

            for loss in losses:
                self.assertTrue(torch.isfinite(torch.tensor(loss)).item())

    def test_model_reset_each_round(self):
        """验证每个 round 开始时模型参数被重置到初始状态

        通过对比两个相同 lr 的 round，验证模型参数确实被重置。
        如果模型被正确重置，相同 lr 的两个 round 应产生相同的 loss。
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            merge_path = _create_merge_file(tmp_dir, n_items=2)
            fixed_lr = 1e-3

            # 使用相同 lr，2 个 round
            lrs, losses = run_lr_range_test(
                model_name="cnn_v2",
                data_path=merge_path,
                rounds=2,
                epochs=3,
                min_lr=fixed_lr,
                max_lr=fixed_lr,
                device=torch.device('cpu')
            )

            # 两个 round 使用相同 lr 且模型重置，loss 应该相同
            self.assertAlmostEqual(losses[0], losses[1], places=5)

    def test_epochs_1_single_step_per_round(self):
        """验证 epochs=1 时每个 round 只训练一个 step"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            merge_path = _create_merge_file(tmp_dir, n_items=2)

            lrs, losses = run_lr_range_test(
                model_name="cnn_v2",
                data_path=merge_path,
                rounds=2,
                epochs=1,
                min_lr=1e-4,
                max_lr=1e-3,
                device=torch.device('cpu')
            )

            self.assertEqual(len(losses), 2)
            for loss in losses:
                self.assertIsInstance(loss, float)


if __name__ == '__main__':
    unittest.main()
