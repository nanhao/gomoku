"""FC v1 训练器错误处理测试

测试训练器在异常输入时的行为：缺失文件、无效 JSON。
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


class TestFCv1TrainerErrorHandling(unittest.TestCase):
    """FC v1 训练器错误处理测试"""
    
    def test_train_single_missing_sample_file(self):
        """测试 sample 文件不存在时的处理"""
        trainer = FCv1Trainer()
        
        # 应该抛出 FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            trainer.train_single("/nonexistent/path")
    
    def test_load_sample_invalid_json(self):
        """测试加载无效 JSON 文件"""
        temp_dir = tempfile.mkdtemp()
        try:
            invalid_sample_path = os.path.join(temp_dir, "invalid.json")
            with open(invalid_sample_path, 'w') as f:
                f.write("invalid json content")
            
            trainer = FCv1Trainer()
            
            # 应该抛出 json.JSONDecodeError
            with self.assertRaises(json.JSONDecodeError):
                trainer._sample_loader.load(invalid_sample_path)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
