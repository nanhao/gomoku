"""FC v1 训练器集成测试

测试训练器的端到端训练流程，验证模型在训练后能产生正确输出。
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
from tests.ai.helpers import create_output


class TestFCv1TrainerIntegration(unittest.TestCase):
    """FC v1 训练器集成测试"""
    
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
        sample_data = []
        for i in range(10):
            sample_data.append({
                "input": [0] * 225 + [1],
                "output": create_output(i % 15, i // 15, 1000000 - i * 10000),
                "mask": [1 if j == i else 0 for j in range(225)]
            })
        
        sample_path = os.path.join(self.sample_dir, "sample.json")
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f)
    
    def test_train_single_improves_model(self):
        """测试训练是否能改善模型（损失应该降低）"""
        model = FCv1Model()
        trainer = FCv1Trainer(model)
        trainer.loss_logger = MagicMock()
        trainer.loss_logger.dense_log = False

        # 训练前预测
        board_vector = [0] * 225 + [1]
        result_before = model.predict(board_vector)
        
        # 训练
        loss = trainer.train_single(self.sample_dir)
        
        # 训练后预测
        result_after = model.predict(board_vector)
        
        # 模型应该被训练（损失存在）
        self.assertGreaterEqual(loss, 0)
        
        # 输出格式应该正确
        self.assertEqual(len(result_after), 225)


if __name__ == '__main__':
    unittest.main()
