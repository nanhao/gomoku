"""BaseTrainer 抽象基类测试

测试 BaseTrainer 的抽象基类特性。
"""

import json
import os
import sys
import tempfile
import shutil
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
        
        # 重定向模型路径，避免测试污染真实模型文件
        self.temp_dir = tempfile.mkdtemp()
        self._original_path = FCv1Model.DEFAULT_MODEL_PATH
        FCv1Model.DEFAULT_MODEL_PATH = os.path.join(self.temp_dir, "fc_v1.pth")
        
        class TestTrainer(BaseTrainer):
            def _create_model(self, fresh):
                return FCv1Model(fresh=fresh)
            
            @property
            def model_name(self):
                return "test_model"
        
        self.TrainerClass = TestTrainer
    
    def tearDown(self):
        """恢复模型路径并清理临时目录"""
        from ai.models.fc_v1.model import FCv1Model
        FCv1Model.DEFAULT_MODEL_PATH = self._original_path
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
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

    def test_no_scheduler_by_default(self):
        """默认情况下 lr_scheduler 应为 None"""
        trainer = self.TrainerClass(fresh=True)
        self.assertIsNone(trainer.lr_scheduler)

    def test_step_scheduler_created(self):
        """lr_scheduler_type='step' 时创建 StepLR"""
        trainer = self.TrainerClass(
            fresh=True,
            lr_scheduler_type="step",
            lr_step_size=2,
            lr_gamma=0.5
        )
        self.assertIsNotNone(trainer.lr_scheduler)
        self.assertEqual(trainer.lr_scheduler_type, "step")

    def test_plateau_scheduler_created(self):
        """lr_scheduler_type='plateau' 时创建 ReduceLROnPlateau"""
        trainer = self.TrainerClass(
            fresh=True,
            lr_scheduler_type="plateau",
            lr_patience=5,
            lr_factor=0.5
        )
        self.assertIsNotNone(trainer.lr_scheduler)
        self.assertEqual(trainer.lr_scheduler_type, "plateau")

    def test_step_lr_decay_during_training(self):
        """StepLR 在训练中使学习率按 gamma 衰减"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = os.path.join(tmp_dir, "sample")
            os.makedirs(sample_dir)
            sample_data = [{
                "input": [0.0] * 226,
                "output": [1.0 / 225.0] * 225,
                "mask": [1] * 225
            }]
            with open(os.path.join(sample_dir, "sample.json"), 'w', encoding='utf-8') as f:
                json.dump(sample_data, f)

            trainer = self.TrainerClass(
                fresh=True,
                lr_scheduler_type="step",
                lr_step_size=2,
                lr_gamma=0.5
            )
            # 注入 mock logger 避免污染真实 loss_logs 目录
            trainer.loss_logger = MagicMock()
            trainer.loss_logger.dense_log = False

            initial_lr = trainer.optimizer.param_groups[0]['lr']

            # 训练 3 个 epoch，step_size=2，gamma=0.5
            # epoch 0-1: lr 不变；epoch 2: lr *= 0.5
            trainer.train_single(sample_dir, epochs=3, verbose=False)

            final_lr = trainer.optimizer.param_groups[0]['lr']
            self.assertEqual(final_lr, initial_lr * 0.5)

    def test_plateau_lr_decay_on_flat_loss(self):
        """ReduceLROnPlateau 在 loss 不下降时降低学习率"""
        trainer = self.TrainerClass(
            fresh=True,
            lr_scheduler_type="plateau",
            lr_patience=0,
            lr_factor=0.5
        )
        initial_lr = trainer.optimizer.param_groups[0]['lr']

        # 直接调用 scheduler.step() 模拟 loss 平台期
        # patience=0，第一个不改善的 epoch 就触发
        trainer.lr_scheduler.step(0.01)  # epoch 1，best=0.01
        trainer.lr_scheduler.step(0.01)  # epoch 2，不改善，触发衰减

        final_lr = trainer.optimizer.param_groups[0]['lr']
        self.assertLess(final_lr, initial_lr)


if __name__ == '__main__':
    unittest.main()
