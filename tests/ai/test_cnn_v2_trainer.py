"""CNN v2 训练器测试

测试 CNN v2 训练器的功能。
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.models.cnn_v2.trainer import CNNV2Trainer
from ai.models.cnn_v2.model import CNNV2Model


class TestCNNv2Trainer(unittest.TestCase):
    """测试 CNN v2 训练器"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_dir = os.path.join(self.temp_dir, "sample_data")
        os.makedirs(self.sample_dir, exist_ok=True)

        self._create_test_sample()

    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_sample(self):
        """创建测试用的 sample.json"""
        sample_data = [
            {
                "input": [0] * 225 + [1],
                "output": self._create_output(0, 0, 1000000),
                "mask": [1] + [0] * 224
            },
            {
                "input": [1] + [0] * 224 + [2],
                "output": self._create_output(1, 0, 500000),
                "mask": [0] * 15 + [1] + [0] * 209
            }
        ]

        sample_path = os.path.join(self.sample_dir, "sample.json")
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f)

    def _create_output(self, x: int, y: int, reward: int) -> list:
        """创建 225 维输出向量"""
        output = [0] * 225
        idx = x * 15 + y
        output[idx] = reward
        return output

    def test_trainer_initialization(self):
        """测试初始化训练器"""
        trainer = CNNV2Trainer()
        self.assertIsNotNone(trainer)
        self.assertIsNotNone(trainer.model)
        self.assertEqual(trainer.model.name, "cnn_v2")

    def test_trainer_model_name(self):
        """测试 model_name 属性"""
        trainer = CNNV2Trainer()
        self.assertEqual(trainer.model_name, "cnn_v2")

    def test_trainer_train_single(self):
        """测试单轮次训练"""
        trainer = CNNV2Trainer()

        loss = trainer.train_single(self.sample_dir)

        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)

    def test_trainer_prepare_tensors(self):
        """测试准备张量"""
        trainer = CNNV2Trainer()
        sample_path = os.path.join(self.sample_dir, "sample.json")
        sample_data = trainer._load_sample_data(sample_path)

        inputs, targets, masks = trainer._prepare_tensors(sample_data)

        # CNN 输入是 4D: batch, channels, height, width
        self.assertEqual(inputs.shape, (2, 3, 15, 15))
        self.assertEqual(targets.shape, (2, 225))
        self.assertEqual(masks.shape, (2, 225))

    def test_trainer_fresh_parameter(self):
        """测试 fresh 参数创建新模型"""
        trainer = CNNV2Trainer(fresh=True)
        self.assertIsNotNone(trainer.model)
        self.assertIsInstance(trainer.model, CNNV2Model)

    def test_trainer_inherits_base_trainer(self):
        """测试 CNNTrainer 继承自 BaseTrainer"""
        from ai.base_trainer import BaseTrainer
        self.assertTrue(issubclass(CNNV2Trainer, BaseTrainer))


class TestCNNv2TrainerEdgeCases(unittest.TestCase):
    """测试 CNN v2 训练器边缘情况"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_train_single_with_nonexistent_sample(self):
        """测试：sample 文件不存在时抛出 FileNotFoundError"""
        trainer = CNNV2Trainer()
        nonexistent_dir = os.path.join(self.temp_dir, "nonexistent")

        with self.assertRaises(FileNotFoundError):
            trainer.train_single(nonexistent_dir)

    def test_train_batch_with_empty_list(self):
        """测试：空 sample_dirs 列表抛出 ValueError"""
        trainer = CNNV2Trainer()

        with self.assertRaises(ValueError) as context:
            trainer.train_batch([])

        self.assertIn("不能为空", str(context.exception))


if __name__ == '__main__':
    unittest.main()
