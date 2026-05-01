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
        mock_trainer.train_auto.return_value = 0.5
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
        mock_trainer.train_auto.side_effect = RuntimeError("训练异常")
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
        mock_trainer.train_auto.side_effect = RuntimeError("训练异常")
        mock_get_trainer.return_value = mock_trainer
        mock_loader = MagicMock()
        mock_loader.discover_and_load.return_value = self.sample_data
        mock_loader_cls.return_value = mock_loader

        with patch('sys.argv', ['train_main.py', '--model', 'fc_v1',
                                '--train-from-dir', self.data_dir]):
            with self.assertRaises(SystemExit):
                main()

        mock_rmtree.assert_any_call('/fake/tmp/sample_dir', ignore_errors=True)

    # ---- 真实目录端到端（方案 A：真 mkdtemp/rmtree + mock trainer）----

    @patch('ai.train_main.get_trainer')
    def test_json_path_temp_dir_actually_removed_on_disk(self, mock_get_trainer):
        """端到端：--train-from-dir <.json> 退出后临时目录真从磁盘消失

        破除上面 4 个用例"只验证 mock.rmtree.assert_any_call 被调用"
        的弱断言。本用例不 patch shutil.rmtree 和 _create_temp_sample_dir，
        让 tempfile.mkdtemp + shutil.rmtree 走真实逻辑，在磁盘上验证：
          - 训练前: _create_temp_sample_dir 创建的目录真实存在
          - 训练中: 目录仍然存在（未被提前清理）
          - 训练后: 目录已从磁盘消失

        为避免真训练耗时，依旧 mock get_trainer 返回桩 trainer。
        """
        import ai.runners as _runners

        captured_paths = []
        original_create = _runners._create_temp_sample_dir

        def _spy_create(sample_data):
            path = original_create(sample_data)
            captured_paths.append(path)
            # 此时 finally 尚未执行，目录应真实存在于磁盘
            self.assertTrue(
                os.path.isdir(path),
                f"_create_temp_sample_dir 应在磁盘上创建真实目录: {path}"
            )
            # sample.json 应已写入
            self.assertTrue(
                os.path.isfile(os.path.join(path, 'sample.json')),
                f"临时目录下应生成 sample.json: {path}"
            )
            return path

        mock_trainer = self._make_mock_trainer()

        def _train_auto_side_effect(*args, **kwargs):
            # 训练过程中临时目录仍应存在（证明 rmtree 未提前触发）
            self.assertTrue(captured_paths, "训练开始前应已创建临时目录")
            self.assertTrue(
                os.path.isdir(captured_paths[0]),
                f"训练过程中临时目录应仍在磁盘上: {captured_paths[0]}"
            )
            return 0.5

        mock_trainer.train_auto.side_effect = _train_auto_side_effect
        mock_get_trainer.return_value = mock_trainer

        with patch('ai.runners._create_temp_sample_dir', side_effect=_spy_create):
            with patch('sys.argv', ['train_main.py', '--model', 'fc_v1',
                                    '--train-from-dir', self.json_path]):
                main()

        # 关键断言：退出后临时目录真的从磁盘消失
        self.assertEqual(len(captured_paths), 1,
                         "应恰好创建 1 次临时目录")
        tmp_path = captured_paths[0]
        self.assertFalse(
            os.path.exists(tmp_path),
            f"训练退出后临时目录应已被清理（磁盘上不应存在）: {tmp_path}"
        )


if __name__ == '__main__':
    unittest.main()
