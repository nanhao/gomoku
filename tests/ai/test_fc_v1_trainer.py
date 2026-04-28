"""FC v1 训练器基础测试

测试 FC v1 训练器的初始化、单轮训练、数据加载和损失计算。
其他测试已拆分至：
- test_fc_v1_trainer_integration.py  集成测试
- test_fc_v1_trainer_error.py        错误处理测试
- test_fc_v1_trainer_cycle.py        训练周期测试
- test_fc_v1_trainer_batch.py        批量训练与张量准备测试
- test_fc_v1_trainer_verbose.py      Verbose 输出测试
- test_fc_v1_trainer_fresh.py        Fresh 参数测试
- test_fc_v1_softmax_input.py        Softmax 概率输入测试
- test_train_main_script.py          训练入口脚本测试
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
from unittest.mock import MagicMock

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.models.fc_v1.trainer import FCv1Trainer
from ai.models.fc_v1.model import FCv1Model
from ai.sample_loader import compute_masked_l1_loss
from tests.ai.helpers import create_output


class TestFCv1Trainer(unittest.TestCase):
    """测试 FC v1 训练器"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_dir = os.path.join(self.temp_dir, "sample_data")
        os.makedirs(self.sample_dir, exist_ok=True)
        
        # 重定向模型路径，避免测试污染真实模型文件
        self._original_path = FCv1Model.DEFAULT_MODEL_PATH
        FCv1Model.DEFAULT_MODEL_PATH = os.path.join(self.temp_dir, "fc_v1.pth")
        
        # 创建测试用的 sample.json
        self._create_test_sample()
    
    def tearDown(self):
        """测试后清理"""
        FCv1Model.DEFAULT_MODEL_PATH = self._original_path
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_sample(self):
        """创建测试用的 sample.json"""
        sample_data = [
            {
                "input": [0] * 225 + [1],  # 空棋盘，黑方
                "output": create_output(0, 0, 1000000),  # (0,0) 位置高 reward
                "mask": [1] + [0] * 224  # 只有第一个位置有效
            },
            {
                "input": [1] + [0] * 224 + [2],  # (0,0) 有黑子，白方
                "output": create_output(1, 0, 500000),  # (1,0) 位置高 reward
                "mask": [0] * 15 + [1] + [0] * 209  # 只有 (1,0) 位置有效
            }
        ]
        
        sample_path = os.path.join(self.sample_dir, "sample.json")
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f)
    
    def test_trainer_initialization_without_model(self):
        """测试不传入模型时初始化训练器"""
        trainer = FCv1Trainer()
        self.assertIsNotNone(trainer)
        self.assertIsNotNone(trainer.model)
        self.assertEqual(trainer.model.name, "fc_v1")
    
    def test_trainer_initialization_with_model(self):
        """测试传入模型时初始化训练器"""
        model = FCv1Model()
        trainer = FCv1Trainer(model)
        self.assertIsNotNone(trainer)
        self.assertEqual(trainer.model, model)
    
    def test_trainer_train_single(self):
        """测试单轮次训练"""
        trainer = FCv1Trainer()
        trainer.loss_logger = MagicMock()
        trainer.loss_logger.dense_log = False

        # 训练
        loss = trainer.train_single(self.sample_dir)
        
        # 损失应该是数值
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)
    
    def test_trainer_train_single_returns_loss(self):
        """测试单轮次训练返回损失值"""
        trainer = FCv1Trainer()
        trainer.loss_logger = MagicMock()
        trainer.loss_logger.dense_log = False

        loss = trainer.train_single(self.sample_dir)
        
        # 损失应该是非负数
        self.assertIsInstance(loss, (int, float))
        self.assertGreaterEqual(loss, 0)
    
    def test_trainer_load_sample_data(self):
        """测试加载 sample 数据"""
        trainer = FCv1Trainer()
        
        sample_path = os.path.join(self.sample_dir, "sample.json")
        data = trainer._sample_loader.load(sample_path)
        
        # 应该返回列表
        self.assertIsInstance(data, list)
        # 应该有 2 条数据
        self.assertEqual(len(data), 2)
        
        # 检查第一条数据的结构
        first_item = data[0]
        self.assertIn("input", first_item)
        self.assertIn("output", first_item)
        self.assertIn("mask", first_item)
        
        # 检查维度
        self.assertEqual(len(first_item["input"]), 226)
        self.assertEqual(len(first_item["output"]), 225)
        self.assertEqual(len(first_item["mask"]), 225)
    
    def test_trainer_compute_loss(self):
        """测试损失计算"""
        trainer = FCv1Trainer()
        
        # 创建测试数据
        estimated = [0] * 225
        sample = [0] * 225
        mask = [0] * 225
                
        # 设置第一个位置的値（索引 0 对应位置 (0,0)）
        estimated[0] = 100  # reward
        sample[0] = 50
        mask[0] = 1
        
        loss = compute_masked_l1_loss(estimated, sample, mask)
        
        # 损失应该是 |100 - 50| = 50
        self.assertEqual(loss, 50)
    
    def test_trainer_compute_loss_with_zero_mask(self):
        """测试 mask 为 0 时的损失计算"""
        trainer = FCv1Trainer()
        
        # 创建测试数据
        estimated = [0] * 225
        sample = [0] * 225
        mask = [0] * 225  # 所有 mask 都是 0
                
        # 设置第一个位置的値
        estimated[0] = 100
        sample[0] = 50
        
        loss = compute_masked_l1_loss(estimated, sample, mask)
        
        # mask 为 0，损失应该是 0
        self.assertEqual(loss, 0)
    
    def test_trainer_compute_loss_multiple_positions(self):
        """测试多个位置的损失计算"""
        trainer = FCv1Trainer()
        
        # 创建测试数据
        estimated = create_output(0, 0, 100)
        sample = create_output(0, 0, 50)
        mask = [1] + [0] * 224
        
        loss = compute_masked_l1_loss(estimated, sample, mask)
        
        # 只有第一个位置参与计算，损失 = |100 - 50| = 50
        self.assertEqual(loss, 50)


if __name__ == '__main__':
    unittest.main()
