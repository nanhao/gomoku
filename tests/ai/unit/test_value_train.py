"""value_train 脚本单元测试

验证参数解析、辅助函数。
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))


class TestValueTrainHelpers(unittest.TestCase):
    """辅助函数测试"""

    def test_count_samples(self):
        """_count_samples 正确统计样本数"""
        from ai.scripts.value_train import _count_samples

        tmp_dir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp_dir, "sample.json")
            samples = [{"state": [0]*226, "z": 1} for _ in range(5)]
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(samples, f)
            self.assertEqual(_count_samples(path), 5)
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_count_samples_empty(self):
        """空列表返回 0"""
        from ai.scripts.value_train import _count_samples

        tmp_dir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp_dir, "sample.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump([], f)
            self.assertEqual(_count_samples(path), 0)
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_count_samples_not_exist(self):
        """不存在文件返回 0"""
        from ai.scripts.value_train import _count_samples
        self.assertEqual(_count_samples("/nonexistent/sample.json"), 0)

    def test_count_samples_not_array(self):
        """非数组格式返回 0"""
        from ai.scripts.value_train import _count_samples

        tmp_dir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp_dir, "sample.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({"not": "array"}, f)
            self.assertEqual(_count_samples(path), 0)
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestValueTrainIntegration(unittest.TestCase):
    """value_train 集成测试"""

    def test_run_train_end_to_end(self):
        """端到端训练流程"""
        from ai.models.cnn_v10.model import CNNV10Model
        from ai.scripts.value_train import run_train
        from unittest.mock import MagicMock

        # 创建 sample.json
        tmp_data_dir = tempfile.mkdtemp()
        tmp_model_dir = tempfile.mkdtemp()
        try:
            # 准备样本数据
            samples = [{"state": [0]*226, "z": 1 if i % 2 == 0 else -1}
                       for i in range(20)]
            sample_path = os.path.join(tmp_data_dir, "sample.json")
            with open(sample_path, 'w', encoding='utf-8') as f:
                json.dump(samples, f)

            # 保存初始模型
            model = CNNV10Model(fresh=True)
            pth_path = os.path.join(tmp_model_dir, "init.pth")
            model.save(pth_path)

            # Mock args
            args = MagicMock()
            args.model = pth_path
            args.model_name = None  # 自动从路径解析
            args.train_from_dir = [tmp_data_dir]
            args.epochs = 2
            args.output = None
            args.batch_size = 8
            args.lr = 0.001
            args.l2_reg = 1e-4
            args.val_split = 0.2
            args.no_shuffle = True
            args.device = "cpu"

            # 需要模拟路径解析：将 model 路径构造为可解析的格式
            # 使用 model-name 方式来避免路径解析问题
            gen_dir = os.path.join(tmp_model_dir, "cnn_v10", "gen1")
            os.makedirs(gen_dir, exist_ok=True)
            gen_pth = os.path.join(gen_dir, "cnn_v10.pth")
            import shutil as _shutil
            _shutil.copy2(pth_path, gen_pth)

            args.model = gen_pth
            out_path = run_train(args)
            self.assertIsNotNone(out_path)
        finally:
            import shutil
            shutil.rmtree(tmp_data_dir, ignore_errors=True)
            shutil.rmtree(tmp_model_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
