"""train_main 临时目录清理测试

TDD: 验证 --train-from-dir 模式下，临时目录在训练结束后被正确清理。
覆盖正常完成和异常退出两种场景。
"""

import os
import sys
import json
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.train_main import main


class TestTrainFromDirTempCleanup(unittest.TestCase):
    """测试 --train-from-dir 模式下临时目录的清理"""

    def setUp(self):
        """创建测试用的 .json 文件和子目录"""
        self.temp_dir = tempfile.mkdtemp(prefix="test_train_cleanup_")
        self.sample_data = [
            {"input": [1.0] * 226, "output": [0.0] * 225, "mask": [1] * 225}
        ]
        # 创建 .json 文件，用于测试 ".json 文件路径" 分支
        self.json_path = os.path.join(self.temp_dir, "test_data.json")
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.sample_data, f)
        # 创建子目录，用于测试 "目录路径" 分支
        self.data_dir = os.path.join(self.temp_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

    def tearDown(self):
        """清理测试临时文件"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _make_mock_trainer(self):
        """创建 mock trainer"""
        mock_trainer = MagicMock()
        mock_trainer.train_batch.return_value = 0.5
        mock_trainer.train_mini_batch.return_value = 0.5
        mock_trainer.model.save.return_value = "/fake/model/path"
        mock_trainer.device = "cpu"
        return mock_trainer

    # ---- .json 文件路径分支 ----

    @patch('shutil.rmtree')
    @patch('ai.runners._create_temp_sample_dir', return_value='/fake/tmp/sample_dir')
    @patch('ai.train_main.get_trainer')
    def test_json_path_cleans_temp_dir_on_success(self, mock_get_trainer, mock_create, mock_rmtree):
        """训练成功后清理临时目录（.json 文件路径）"""
        mock_get_trainer.return_value = self._make_mock_trainer()

        with patch('sys.argv', ['train_main.py', '--model', 'fc_v1',
                                '--train-from-dir', self.json_path]):
            main()

        mock_rmtree.assert_any_call('/fake/tmp/sample_dir', ignore_errors=True)

    @patch('shutil.rmtree')
    @patch('ai.runners._create_temp_sample_dir', return_value='/fake/tmp/sample_dir')
    @patch('ai.train_main.get_trainer')
    def test_json_path_cleans_temp_dir_on_failure(self, mock_get_trainer, mock_create, mock_rmtree):
        """训练失败时仍清理临时目录（.json 文件路径）"""
        mock_trainer = self._make_mock_trainer()
        mock_trainer.train_batch.side_effect = RuntimeError("训练异常")
        mock_get_trainer.return_value = mock_trainer

        with patch('sys.argv', ['train_main.py', '--model', 'fc_v1',
                                '--train-from-dir', self.json_path]):
            with self.assertRaises(SystemExit):
                main()

        mock_rmtree.assert_any_call('/fake/tmp/sample_dir', ignore_errors=True)

    # ---- 目录路径分支 ----

    @patch('shutil.rmtree')
    @patch('ai.runners._create_temp_sample_dir', return_value='/fake/tmp/sample_dir')
    @patch('ai.runners.SampleLoader')
    @patch('ai.train_main.get_trainer')
    def test_directory_path_cleans_temp_dir_on_success(self, mock_get_trainer, mock_loader_cls, mock_create, mock_rmtree):
        """训练成功后清理临时目录（目录路径）"""
        mock_get_trainer.return_value = self._make_mock_trainer()
        mock_loader = MagicMock()
        mock_loader.discover_and_load.return_value = self.sample_data
        mock_loader_cls.return_value = mock_loader

        with patch('sys.argv', ['train_main.py', '--model', 'fc_v1',
                                '--train-from-dir', self.data_dir]):
            main()

        mock_rmtree.assert_any_call('/fake/tmp/sample_dir', ignore_errors=True)

    @patch('shutil.rmtree')
    @patch('ai.runners._create_temp_sample_dir', return_value='/fake/tmp/sample_dir')
    @patch('ai.runners.SampleLoader')
    @patch('ai.train_main.get_trainer')
    def test_directory_path_cleans_temp_dir_on_failure(self, mock_get_trainer, mock_loader_cls, mock_create, mock_rmtree):
        """训练失败时仍清理临时目录（目录路径）"""
        mock_trainer = self._make_mock_trainer()
        mock_trainer.train_batch.side_effect = RuntimeError("训练异常")
        mock_get_trainer.return_value = mock_trainer
        mock_loader = MagicMock()
        mock_loader.discover_and_load.return_value = self.sample_data
        mock_loader_cls.return_value = mock_loader

        with patch('sys.argv', ['train_main.py', '--model', 'fc_v1',
                                '--train-from-dir', self.data_dir]):
            with self.assertRaises(SystemExit):
                main()

        mock_rmtree.assert_any_call('/fake/tmp/sample_dir', ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
