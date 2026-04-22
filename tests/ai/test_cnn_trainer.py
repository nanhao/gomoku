"""CNN 训练器测试

测试 CNN 训练器的功能。
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

from ai.models.cnn_v1.trainer import CNNTrainer
from ai.models.cnn_v1.model import CNNModel
from tests.ai.helpers import create_sample_dir, create_output


class TestCNNTrainer(unittest.TestCase):
    """测试 CNN 训练器"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_dir = os.path.join(self.temp_dir, "sample_data")
        os.makedirs(self.sample_dir, exist_ok=True)
        
        # 创建测试用的 sample.json
        self._create_test_sample()
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_sample(self):
        """创建测试用的 sample.json"""
        sample_data = [
            {
                "input": [0] * 225 + [1],  # 空棋盘，黑方
                "output": self._create_output(0, 0, 1000000),  # (0,0) 位置高 reward
                "mask": [1] + [0] * 224  # 只有第一个位置有效
            },
            {
                "input": [1] + [0] * 224 + [2],  # (0,0) 有黑子，白方
                "output": self._create_output(1, 0, 500000),  # (1,0) 位置高 reward
                "mask": [0] * 15 + [1] + [0] * 209  # 只有 (1,0) 位置有效
            }
        ]
        
        sample_path = os.path.join(self.sample_dir, "sample.json")
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f)
    
    def _create_output(self, x: int, y: int, reward: int) -> list:
        """创建 225 维输出向量，索引 i 对应位置 (i//15, i%15)"""
        output = [0] * 225
        idx = x * 15 + y
        output[idx] = reward
        return output
    
    def test_trainer_initialization_without_model(self):
        """测试不传入模型时初始化训练器"""
        trainer = CNNTrainer()
        self.assertIsNotNone(trainer)
        self.assertIsNotNone(trainer.model)
        self.assertEqual(trainer.model.name, "cnn_v1")
    
    def test_trainer_initialization_with_model(self):
        """测试传入模型时初始化训练器"""
        model = CNNModel()
        trainer = CNNTrainer(model)
        self.assertIsNotNone(trainer)
        self.assertEqual(trainer.model, model)
    
    def test_trainer_train_single(self):
        """测试单轮次训练"""
        trainer = CNNTrainer()
        
        # 训练
        loss = trainer.train_single(self.sample_dir)
        
        # 损失应该是数值
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)
    
    def test_trainer_train_single_returns_loss(self):
        """测试单轮次训练返回损失值"""
        trainer = CNNTrainer()
        
        loss = trainer.train_single(self.sample_dir)
        
        # 损失应该是非负数
        self.assertIsInstance(loss, (int, float))
        self.assertGreaterEqual(loss, 0)
    
    def test_trainer_load_sample_data(self):
        """测试加载 sample 数据"""
        trainer = CNNTrainer()
        sample_path = os.path.join(self.sample_dir, "sample.json")
        
        data = trainer._load_sample_data(sample_path)
        
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)
        self.assertIn("input", data[0])
        self.assertIn("output", data[0])
        self.assertIn("mask", data[0])
    
    def test_trainer_prepare_tensors(self):
        """测试准备张量"""
        trainer = CNNTrainer()
        sample_path = os.path.join(self.sample_dir, "sample.json")
        sample_data = trainer._load_sample_data(sample_path)
        
        inputs, targets, masks = trainer._prepare_tensors(sample_data)
        
        # 验证张量形状（CNN 输入是 4D: batch, channels, height, width）
        self.assertEqual(inputs.shape, (2, 3, 15, 15))  # 2 个样本，3 通道，15x15 棋盘
        self.assertEqual(targets.shape, (2, 225))  # 2 个样本，225 维输出
        self.assertEqual(masks.shape, (2, 225))  # 2 个样本，225 维 mask
        
        # 验证类型
        self.assertIsInstance(inputs, torch.Tensor)
        self.assertIsInstance(targets, torch.Tensor)
        self.assertIsInstance(masks, torch.Tensor)
    
    def test_trainer_compute_loss(self):
        """测试计算损失"""
        trainer = CNNTrainer()
        
        estimated = [0.1] * 225
        sample = [0.0] * 225
        sample[0] = 1.0
        mask = [1] + [0] * 224
        
        loss = trainer._compute_loss(estimated, sample, mask)
        
        # 损失应该是非负数
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)
    
    def test_trainer_train_batch(self):
        """测试批量训练"""
        trainer = CNNTrainer()
        
        # 创建第二个 sample 目录
        sample_dir2 = os.path.join(self.temp_dir, "sample_data2")
        os.makedirs(sample_dir2, exist_ok=True)
        sample_path2 = os.path.join(sample_dir2, "sample.json")
        with open(sample_path2, 'w', encoding='utf-8') as f:
            json.dump([
                {
                    "input": [0] * 225 + [1],
                    "output": self._create_output(2, 2, 1000000),
                    "mask": [1] + [0] * 224
                }
            ], f)
        
        # 批量训练
        sample_dirs = [self.sample_dir, sample_dir2]
        loss = trainer.train_batch(sample_dirs, epochs=5)
        
        # 损失应该是数值
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)
    
    def test_trainer_model_name(self):
        """测试 model_name 属性"""
        trainer = CNNTrainer()
        self.assertEqual(trainer.model_name, "cnn_v1")
    
    def test_trainer_inherits_base_trainer(self):
        """测试 CNNTrainer 继承自 BaseTrainer"""
        from ai.base_trainer import BaseTrainer
        self.assertTrue(issubclass(CNNTrainer, BaseTrainer))
    
    def test_trainer_fresh_parameter(self):
        """测试 fresh 参数创建新模型"""
        trainer = CNNTrainer(fresh=True)
        self.assertIsNotNone(trainer.model)
        self.assertIsInstance(trainer.model, CNNModel)


class TestCNNTrainerEdgeCases(unittest.TestCase):
    """测试 CNN 训练器边缘情况"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_train_single_with_nonexistent_sample(self):
        """测试：sample 文件不存在时抛出 FileNotFoundError"""
        trainer = CNNTrainer()
        nonexistent_dir = os.path.join(self.temp_dir, "nonexistent")
        
        with self.assertRaises(FileNotFoundError):
            trainer.train_single(nonexistent_dir)
    
    def test_train_batch_with_empty_list(self):
        """测试：空 sample_dirs 列表抛出 ValueError"""
        trainer = CNNTrainer()
        
        with self.assertRaises(ValueError) as context:
            trainer.train_batch([])
        
        self.assertIn("不能为空", str(context.exception))
    
    def test_load_sample_data_with_invalid_json(self):
        """测试：无效的 JSON 格式抛出 JSONDecodeError"""
        trainer = CNNTrainer()
        invalid_json_path = os.path.join(self.temp_dir, "invalid.json")
        
        with open(invalid_json_path, 'w', encoding='utf-8') as f:
            f.write("invalid json content")
        
        with self.assertRaises(json.JSONDecodeError):
            trainer._load_sample_data(invalid_json_path)


if __name__ == '__main__':
    unittest.main()
