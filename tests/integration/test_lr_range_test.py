"""LR Range Test 集成测试

端到端测试 lr_range_test 的完整流程。
"""

import json
import os
import sys
import tempfile
import unittest

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch

from ai.models.cnn_v2 import CNNV2Model
from tests.ai.helpers import create_sample_dir
from temp.lr_range_test import run_lr_range_test, plot_lr_loss_curve, load_model


def _create_merge_file(tmp_dir: str, name: str = "merge_test.json",
                       n_items: int = 5) -> str:
    """创建测试用 merge json 文件"""
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


class TestEndToEndCnnV2(unittest.TestCase):
    """端到端测试：CNNV2Model 完整 LR Range Test 流程"""

    def test_end_to_end_cnn_v2(self):
        """使用 CNNV2Model + 模拟数据运行完整流程，验证输出正确"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            merge_path = _create_merge_file(tmp_dir, n_items=3)
            output_path = os.path.join(tmp_dir, "lr_curve.png")

            # 运行 range test
            lrs, losses = run_lr_range_test(
                model_name="cnn_v2",
                data_path=merge_path,
                rounds=3,
                epochs=2,
                min_lr=1e-6,
                max_lr=1e-2,
                device=torch.device('cpu')
            )

            # 验证返回结果
            self.assertEqual(len(lrs), 3)
            self.assertEqual(len(losses), 3)

            # 验证 lr 递增
            for i in range(1, len(lrs)):
                self.assertGreater(lrs[i], lrs[i - 1])

            # 验证 loss 非空且有限
            for loss in losses:
                self.assertIsInstance(loss, float)
                self.assertTrue(torch.isfinite(torch.tensor(loss)).item())

            # 绘图并验证文件生成
            plot_lr_loss_curve(lrs, losses, output_path)
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

    def test_end_to_end_with_directory_data(self):
        """使用目录数据运行完整流程"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = os.path.join(tmp_dir, "train_data")
            create_sample_dir(data_dir, "game_1", n_items=3)
            output_path = os.path.join(tmp_dir, "lr_curve.png")

            lrs, losses = run_lr_range_test(
                model_name="cnn_v2",
                data_path=data_dir,
                rounds=2,
                epochs=2,
                min_lr=1e-6,
                max_lr=1e-3,
                device=torch.device('cpu')
            )

            self.assertEqual(len(lrs), 2)
            self.assertEqual(len(losses), 2)


class TestInvalidInputs(unittest.TestCase):
    """测试无效输入的异常处理"""

    def test_invalid_model_name(self):
        """验证传入不支持模型名称时抛出 ValueError"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            merge_path = _create_merge_file(tmp_dir)
            with self.assertRaises(ValueError):
                run_lr_range_test(
                    model_name="invalid_model",
                    data_path=merge_path,
                    rounds=1,
                    epochs=1,
                    device=torch.device('cpu')
                )

    def test_nonexistent_data_path(self):
        """验证传入不存在的数据路径时抛出 FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            run_lr_range_test(
                model_name="cnn_v2",
                data_path="/nonexistent/data.json",
                rounds=1,
                epochs=1,
                device=torch.device('cpu')
            )


if __name__ == '__main__':
    unittest.main()
