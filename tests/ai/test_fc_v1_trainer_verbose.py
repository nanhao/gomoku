"""FC v1 训练器 Verbose 输出测试

测试 verbose 参数控制 epoch loss 输出行为。
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
import io

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.models.fc_v1.trainer import FCv1Trainer
from ai.models.fc_v1.model import FCv1Model


class TestFCv1TrainerVerboseOutput(unittest.TestCase):
    """测试 FCv1Trainer 的 verbose 输出功能"""
    
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
    
    def test_verbose_false_no_output(self):
        """测试 verbose=False（默认）时不输出每个 epoch 的 loss"""
        sample_dir = self._create_test_sample()
        trainer = FCv1Trainer(fresh=True)
        
        # 捕获 stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            trainer.train_single(sample_dir, epochs=3, verbose=False)
            output = captured_output.getvalue()
        finally:
            sys.stdout = sys.__stdout__
        
        # 验证没有 epoch 输出
        self.assertNotIn("Epoch", output)
        self.assertNotIn("loss", output.lower())
    
    def test_verbose_true_outputs_epoch_loss(self):
        """测试 verbose=True 时每 200 个 epoch 输出一次 loss"""
        sample_dir = self._create_test_sample()
        trainer = FCv1Trainer(fresh=True)
        
        # 捕获 stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            # 使用 400 个 epoch，应输出第 200 和第 400 个 epoch 的 loss
            trainer.train_single(sample_dir, epochs=400, verbose=True)
            output = captured_output.getvalue()
        finally:
            sys.stdout = sys.__stdout__
        
        # 验证在第 200 和第 400 个 epoch 输出了 loss
        self.assertIn("Epoch 200/400", output)
        self.assertIn("Epoch 400/400", output)
        self.assertIn("loss", output.lower())
        # 确认只输出了 2 行，而不是 400 行
        lines = [l for l in output.strip().split('\n') if 'Epoch' in l]
        self.assertEqual(len(lines), 2)
    
    def test_verbose_output_format(self):
        """测试 verbose 输出格式正确（每 200 个 epoch 打印一次）"""
        sample_dir = self._create_test_sample()
        trainer = FCv1Trainer(fresh=True)
        
        import re
        
        # 捕获 stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            # 使用 200 个 epoch，恰好在第 200 个 epoch 输出一次
            trainer.train_single(sample_dir, epochs=200, verbose=True)
            output = captured_output.getvalue()
        finally:
            sys.stdout = sys.__stdout__
        
        # 验证输出格式：Epoch X/Y, loss: Z.ZZZZ
        self.assertIn("Epoch 200/200", output)
        self.assertRegex(output, r"loss:\s*\d+\.\d+")


if __name__ == '__main__':
    unittest.main()
