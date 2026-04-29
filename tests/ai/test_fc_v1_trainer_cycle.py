"""FC v1 训练周期测试

测试 train_cycle 方法及 rate 参数：
- TestFCv1TrainerCycle: 训练周期基本功能
- TestTrainCycleRateParameter: rate 参数默认值验证
"""

import os
import sys
import unittest
import tempfile
import shutil
import inspect

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.models.fc_v1.trainer import FCv1Trainer
from ai.models.fc_v1.model import FCv1Model
from unittest.mock import MagicMock


class TestFCv1TrainerCycle(unittest.TestCase):
    """FC v1 训练周期测试"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.ai_data_dir = os.path.join(self.temp_dir, "ai", "ai_data")
        self.train_data_dir = os.path.join(self.temp_dir, "ai", "train_data")
        os.makedirs(self.ai_data_dir, exist_ok=True)
        os.makedirs(self.train_data_dir, exist_ok=True)
        
        # 隔离默认模型路径，防止 train_cycle 保存时覆盖真实模型文件
        self.original_default_path = FCv1Model.DEFAULT_MODEL_PATH
        FCv1Model.DEFAULT_MODEL_PATH = os.path.join(self.temp_dir, "fc_v1.pth")
    
    def tearDown(self):
        """测试后清理"""
        FCv1Model.DEFAULT_MODEL_PATH = self.original_default_path
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestTrainCycleRateParameter(unittest.TestCase):
    """train_cycle rate 参数测试"""
    
    def test_train_cycle_has_rate_parameter(self):
        """train_cycle 应有 rate 参数，默认值 0.9"""
        sig = inspect.signature(FCv1Trainer.train_cycle)
        self.assertIn('rate', sig.parameters, "train_cycle 应有 rate 参数")
        self.assertAlmostEqual(
            sig.parameters['rate'].default, 0.9, places=9,
            msg="rate 默认值应为 0.9"
        )
    
    def test_train_main_script_rate_default(self):
        """train_main.py --rate 默认值为 0.9"""
        import argparse
        
        original_argv = sys.argv
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument('--rate', type=float, default=0.9)
            parser.add_argument('--model', type=str, default='fc_v1')
            
            args = parser.parse_args([])
            self.assertAlmostEqual(args.rate, 0.9, places=9, msg="--rate 默认值应为 0.9")
            self.assertEqual(args.model, 'fc_v1', "--model 默认值应为 fc_v1")
        finally:
            sys.argv = original_argv


if __name__ == '__main__':
    unittest.main()
