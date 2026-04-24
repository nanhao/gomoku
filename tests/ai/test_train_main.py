"""train_main.py 测试

测试 train_main 模块的功能。
"""

import os
import sys
import unittest

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.train_main import get_trainer


class TestGetTrainer(unittest.TestCase):
    """测试 get_trainer 函数"""
    
    def test_get_trainer_returns_fc_v1_trainer(self):
        """测试：fc_v1 返回 FCv1Trainer"""
        from ai.models.fc_v1.trainer import FCv1Trainer
        
        trainer = get_trainer("fc_v1")
        
        self.assertIsInstance(trainer, FCv1Trainer)
    
    def test_get_trainer_returns_cnn_trainer(self):
        """测试：cnn_v1 返回 CNNV1Trainer"""
        from ai.models.cnn_v1.trainer import CNNV1Trainer
        
        trainer = get_trainer("cnn_v1")
        
        self.assertIsInstance(trainer, CNNV1Trainer)
    
    def test_get_trainer_raises_for_invalid_model(self):
        """测试：无效模型名抛出 ValueError"""
        with self.assertRaises(ValueError) as context:
            get_trainer("invalid_model")
        
        error_msg = str(context.exception)
        self.assertIn("不支持的模型", error_msg)
        self.assertIn("invalid_model", error_msg)
    
    def test_get_trainer_fc_v1_with_fresh(self):
        """测试：fc_v1 支持 fresh 参数"""
        trainer = get_trainer("fc_v1", fresh=True)
        self.assertIsNotNone(trainer)
        self.assertIsNotNone(trainer.model)
    
    def test_get_trainer_cnn_v1_with_fresh(self):
        """测试：cnn_v1 支持 fresh 参数"""
        trainer = get_trainer("cnn_v1", fresh=True)
        self.assertIsNotNone(trainer)
        self.assertIsNotNone(trainer.model)


if __name__ == '__main__':
    unittest.main()
