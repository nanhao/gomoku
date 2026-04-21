"""BaseTrainer 抽象基类测试

测试 BaseTrainer 的抽象基类特性。
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.base_trainer import BaseTrainer


class TestBaseTrainer(unittest.TestCase):
    """测试 BaseTrainer 抽象基类"""
    
    def test_base_trainer_cannot_instantiate_directly(self):
        """测试：直接实例化 BaseTrainer 应该报错"""
        with self.assertRaises(TypeError) as context:
            BaseTrainer()
        
        # 错误信息应包含抽象方法
        error_msg = str(context.exception)
        self.assertIn("abstract", error_msg.lower())
    
    def test_base_trainer_requires_create_model(self):
        """测试：子类必须实现 _create_model 方法"""
        class IncompleteTrainer(BaseTrainer):
            @property
            def model_name(self):
                return "incomplete"
        
        with self.assertRaises(TypeError) as context:
            IncompleteTrainer()
        
        error_msg = str(context.exception)
        self.assertIn("_create_model", error_msg)
    
    def test_base_trainer_requires_model_name(self):
        """测试：子类必须实现 model_name 属性"""
        class IncompleteTrainer(BaseTrainer):
            def _create_model(self, fresh):
                from ai.models.fc_v1.model import FCv1Model
                return FCv1Model(fresh=fresh)
        
        with self.assertRaises(TypeError) as context:
            IncompleteTrainer()
        
        error_msg = str(context.exception)
        self.assertIn("model_name", error_msg)


class TestBaseTrainerWithCompleteSubclass(unittest.TestCase):
    """使用完整子类测试 BaseTrainer 功能"""
    
    def setUp(self):
        """创建完整子类"""
        from ai.models.fc_v1.model import FCv1Model
        
        class TestTrainer(BaseTrainer):
            def _create_model(self, fresh):
                return FCv1Model(fresh=fresh)
            
            @property
            def model_name(self):
                return "test_model"
        
        self.TrainerClass = TestTrainer
    
    def test_complete_subclass_can_instantiate(self):
        """测试：完整子类可以实例化"""
        trainer = self.TrainerClass()
        self.assertIsNotNone(trainer)
        self.assertIsNotNone(trainer.model)
    
    def test_model_name_property(self):
        """测试：model_name 属性返回正确值"""
        trainer = self.TrainerClass()
        self.assertEqual(trainer.model_name, "test_model")
    
    def test_fresh_parameter_creates_new_model(self):
        """测试：fresh=True 创建新模型"""
        trainer = self.TrainerClass(fresh=True)
        self.assertIsNotNone(trainer.model)
    
    def test_train_single_generates_loss_log(self):
        """测试 train_single 训练后通过依赖注入生成 loss 日志"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 创建 sample.json（单个样本，FC 模型输入 226 维，输出 225 维）
            sample_dir = os.path.join(tmp_dir, "sample")
            os.makedirs(sample_dir)
            sample_data = [{
                "input": [0.0] * 226,
                "output": [1.0 / 225.0] * 225,
                "mask": [1] * 225
            }]
            with open(os.path.join(sample_dir, "sample.json"), 'w', encoding='utf-8') as f:
                json.dump(sample_data, f)
            
            trainer = self.TrainerClass(fresh=True)
            mock_logger = MagicMock()
            trainer.loss_logger = mock_logger
            
            trainer.train_single(sample_dir, epochs=1000, verbose=False)
            
            # 验证 log 和 plot 被调用（1000/200 = 5 次，断言 >= 4）
            self.assertTrue(mock_logger.log.called)
            self.assertGreaterEqual(mock_logger.log.call_count, 4)
            self.assertTrue(mock_logger.plot.called)
    
    def test_train_batch_generates_loss_curve(self):
        """测试 train_batch 训练后通过依赖注入生成 loss 曲线图"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dirs = []
            for i in range(2):
                sample_dir = os.path.join(tmp_dir, f"sample_{i}")
                os.makedirs(sample_dir)
                sample_data = [{
                    "input": [0.0] * 226,
                    "output": [1.0 / 225.0] * 225,
                    "mask": [1] * 225
                }]
                with open(os.path.join(sample_dir, "sample.json"), 'w', encoding='utf-8') as f:
                    json.dump(sample_data, f)
                sample_dirs.append(sample_dir)
            
            trainer = self.TrainerClass(fresh=True)
            mock_logger = MagicMock()
            trainer.loss_logger = mock_logger
            
            trainer.train_batch(sample_dirs, epochs=1000, verbose=False)
            
            # 验证 log 和 plot 被调用
            self.assertTrue(mock_logger.log.called)
            self.assertTrue(mock_logger.plot.called)


if __name__ == '__main__':
    unittest.main()
