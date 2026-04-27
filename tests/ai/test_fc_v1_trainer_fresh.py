"""FC v1 训练器 Fresh 参数测试

测试 FCv1Trainer 的 fresh 参数行为：
- fresh=True 创建新模型
- 默认行为加载已保存模型
- 传入 model 参数时使用该模型
"""

import os
import sys
import unittest
import tempfile
import shutil
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.models.fc_v1.trainer import FCv1Trainer
from ai.models.fc_v1.model import FCv1Model
from unittest.mock import MagicMock


class TestFCv1TrainerFreshParameter(unittest.TestCase):
    """测试 FCv1Trainer 的 fresh 参数"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        # 保存原始 DEFAULT_MODEL_PATH
        self.original_default_path = FCv1Model.DEFAULT_MODEL_PATH
        # 设置临时默认路径
        FCv1Model.DEFAULT_MODEL_PATH = os.path.join(self.temp_dir, "fc_v1.pth")
    
    def tearDown(self):
        """测试后清理"""
        FCv1Model.DEFAULT_MODEL_PATH = self.original_default_path
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_sample(self):
        """创建测试用的 sample.json"""
        sample_dir = os.path.join(self.temp_dir, "sample_data")
        os.makedirs(sample_dir, exist_ok=True)
        
        sample_data = [{
            "input": [0] * 226,
            "output": [0] * 225,
            "mask": [1] + [0] * 224
        }]
        
        sample_path = os.path.join(sample_dir, "sample.json")
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f)
        
        return sample_dir
    
    def test_trainer_fresh_parameter_creates_new_model(self):
        """测试 fresh=True 时创建新模型"""
        # 创建并保存一个模型
        model1 = FCv1Model(fresh=True)
        model1.save()
        
        # 使用 fresh=True 创建训练器
        trainer = FCv1Trainer(fresh=True)
        
        # 验证训练器创建成功
        self.assertIsNotNone(trainer.model)
        self.assertEqual(trainer.model.name, "fc_v1")
    
    def test_trainer_default_loads_existing_model(self):
        """测试默认行为加载已保存模型"""
        # 创建并保存一个模型
        model1 = FCv1Model(fresh=True)
        board_vector = [0] * 225 + [1]
        result1 = model1.predict(board_vector)
        model1.save()
        
        # 创建训练器（不传参数）
        trainer = FCv1Trainer()
        result2 = trainer.model.predict(board_vector)
        
        # 应该加载已保存的模型，结果相近（GPU/CPU 可能有微小精度差异）
        self.assertEqual(len(result1), len(result2))
        for r1, r2 in zip(result1, result2):
            self.assertAlmostEqual(r1, r2, places=5)
    
    def test_trainer_with_model_parameter(self):
        """测试传入 model 参数时使用该模型"""
        # 创建自定义模型
        custom_model = FCv1Model(fresh=True)
        board_vector = [0] * 225 + [1]
        result1 = custom_model.predict(board_vector)
        
        # 使用自定义模型创建训练器
        trainer = FCv1Trainer(model=custom_model)
        result2 = trainer.model.predict(board_vector)
        
        # 应该是同一个模型实例，结果相近（GPU/CPU 可能有微小精度差异）
        self.assertEqual(len(result1), len(result2))
        for r1, r2 in zip(result1, result2):
            self.assertAlmostEqual(r1, r2, places=5)
        self.assertIs(trainer.model, custom_model)
    
    def test_trainer_fresh_with_sample_dir(self):
        """测试 fresh=True 与单轮次训练结合"""
        sample_dir = self._create_test_sample()
        
        # 使用 fresh=True 创建训练器并训练
        trainer = FCv1Trainer(fresh=True)
        # 注入 mock logger 避免污染真实 loss_logs 目录
        trainer.loss_logger = MagicMock()
        trainer.loss_logger.dense_log = False
        loss = trainer.train_single(sample_dir, epochs=1)
        
        # 验证训练成功
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)


if __name__ == '__main__':
    unittest.main()
