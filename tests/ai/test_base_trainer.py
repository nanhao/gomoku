"""BaseTrainer 抽象基类测试

测试 BaseTrainer 的抽象基类特性。
"""

import os
import sys
import unittest

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


if __name__ == '__main__':
    unittest.main()
