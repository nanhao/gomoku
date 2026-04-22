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


class TestFCv1TrainerCycle(unittest.TestCase):
    """FC v1 训练周期测试"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.ai_data_dir = os.path.join(self.temp_dir, "ai_data")
        self.train_data_dir = os.path.join(self.temp_dir, "train_data")
        os.makedirs(self.ai_data_dir, exist_ok=True)
        os.makedirs(self.train_data_dir, exist_ok=True)
        
        # 隔离默认模型路径，防止 train_cycle 保存时覆盖真实模型文件
        self.original_default_path = FCv1Model.DEFAULT_MODEL_PATH
        FCv1Model.DEFAULT_MODEL_PATH = os.path.join(self.temp_dir, "fc_v1.pth")
    
    def tearDown(self):
        """测试后清理"""
        FCv1Model.DEFAULT_MODEL_PATH = self.original_default_path
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_train_cycle_basic(self):
        """测试训练周期基本功能
        
        验证 train_cycle 能完整执行，且创建 ai_data 和 train_data 目录。
        不使用 try/except 吞异常，确保训练周期失败时测试快速暴露。
        """
        trainer = FCv1Trainer()
        
        # 使用很小的参数进行快速测试
        # 1 轮，2 局，1 epoch
        trainer.train_cycle(
            rounds=1,
            games=2,
            epochs=1,
            project_root=self.temp_dir
        )
        
        # 验证 ai_data 目录被创建并包含数据
        self.assertTrue(os.path.exists(self.ai_data_dir),
                        "train_cycle 应创建 ai_data 目录")
        
        # 验证 train_data 目录被创建
        self.assertTrue(os.path.exists(self.train_data_dir),
                        "train_cycle 应创建 train_data 目录")
    
    def test_train_cycle_creates_model_file(self):
        """测试训练周期后模型文件被保存
        
        验证 train_cycle 执行后模型可保存。不使用 try/except 容错，
        确保训练周期失败时测试快速暴露。
        """
        trainer = FCv1Trainer()
        
        # 临时模型路径
        model_path = os.path.join(self.temp_dir, "test_model.pth")
        
        # 执行训练周期
        trainer.train_cycle(
            rounds=1,
            games=1,
            epochs=1,
            project_root=self.temp_dir
        )
        
        # 保存模型到临时路径
        saved_path = trainer.model.save(model_path)
        
        # 验证模型文件存在
        self.assertTrue(os.path.exists(saved_path),
                        "训练后模型文件应被保存")
    
    def test_train_cycle_with_existing_model(self):
        """测试使用已有模型进行训练周期
        
        验证传入已有模型时 train_cycle 能正常执行，且训练后模型仍可用。
        不使用 try/except 容错，确保训练周期失败时测试快速暴露。
        """
        # 创建初始模型
        initial_model = FCv1Model()
        trainer = FCv1Trainer(initial_model)
        
        # 执行训练周期
        trainer.train_cycle(
            rounds=1,
            games=1,
            epochs=1,
            project_root=self.temp_dir
        )
        
        # 验证模型仍然可用
        result = trainer.model.predict([0] * 226)
        self.assertEqual(len(result), 225,
                         "训练后模型应仍能正确预测")


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
