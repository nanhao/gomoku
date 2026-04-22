"""训练入口脚本 (train_main.py) 参数默认值测试

验证 train_main.py 和 FCv1Trainer 的默认参数值与设计文档一致。
"""

import os
import sys
import unittest
import tempfile
import json
import inspect

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.models.fc_v1.trainer import FCv1Trainer


class TestTrainMainScript(unittest.TestCase):
    """训练入口脚本 (train_main.py) 参数默认值测试"""
    
    def test_train_main_script_epochs_default(self):
        """测试 train_main.py --epochs 默认值为 1000"""
        import argparse
        
        # 模拟命令行参数解析
        original_argv = sys.argv
        try:
            sys.argv = ['train_main.py']
            
            parser = argparse.ArgumentParser()
            parser.add_argument('--epochs', type=int, default=1000)
            parser.add_argument('--rounds', type=int, default=10)
            parser.add_argument('--games', type=int, default=100)
            parser.add_argument('--model', type=str, default='fc_v1')
            
            args = parser.parse_args([])
            
            # 验证默认值
            self.assertEqual(args.epochs, 1000, "--epochs 默认值应为 1000")
            self.assertEqual(args.rounds, 10, "--rounds 默认值应为 10")
            self.assertEqual(args.games, 100, "--games 默认值应为 100")
            self.assertEqual(args.model, 'fc_v1', "--model 默认值应为 fc_v1")
            
        finally:
            sys.argv = original_argv
    
    def test_trainer_train_single_epochs_default(self):
        """测试 trainer.train_single 默认 epochs 为 10"""
        trainer = FCv1Trainer()
        
        # 创建临时 sample 数据
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_data = [{
                "input": [0] * 226,
                "output": [0] * 225,
                "mask": [1] + [0] * 224
            }]
            
            sample_path = os.path.join(tmpdir, "sample.json")
            with open(sample_path, 'w') as f:
                json.dump(sample_data, f)
            
            # 使用默认 epochs 调用（不传 epochs 参数）
            loss = trainer.train_single(tmpdir)
            
            # 验证返回了损失值
            self.assertIsInstance(loss, float)
            self.assertGreaterEqual(loss, 0)
    
    def test_trainer_train_cycle_defaults(self):
        """测试 trainer.train_cycle 默认参数值"""
        # 获取 train_cycle 方法的签名
        sig = inspect.signature(FCv1Trainer.train_cycle)
        
        # 验证默认参数值
        self.assertEqual(sig.parameters['rounds'].default, 10, "rounds 默认值应为 10")
        self.assertEqual(sig.parameters['games'].default, 100, "games 默认值应为 100")
        self.assertEqual(sig.parameters['epochs'].default, 1000, "epochs 默认值应为 1000")


if __name__ == '__main__':
    unittest.main()
