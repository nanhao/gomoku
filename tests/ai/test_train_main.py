"""train_main.py 测试

测试参数解析、模式路由和 runners 模块的集成行为。
get_trainer 的单元测试由 tests/ai/unit/test_registry.py 覆盖。
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.train_main import main


class TestTrainMainNoTrainFromDir(unittest.TestCase):
    """测试不指定 --train-from-dir 时的报错行为"""

    def test_no_train_from_dir_fails(self):
        """不指定 --train-from-dir 应报错退出"""
        with patch('sys.argv', ['train_main.py', '--model', 'fc_v1']):
            with self.assertRaises(SystemExit):
                main()

    def test_no_train_from_dir_error_message(self):
        """不指定 --train-from-dir 时报错消息应为指定参数提示"""
        with patch('sys.argv', ['train_main.py', '--model', 'fc_v1']):
            with self.assertRaises(SystemExit) as cm:
                main()
            # argparse.error 导致 SystemExit(2)
            self.assertEqual(cm.exception.code, 2)


class TestTrainFromDirModes(unittest.TestCase):
    """测试 --train-from-dir 支持目录和 .json 文件"""

    def test_train_from_dir_json_file(self):
        """--train-from-dir 直接指定 .json 文件时正确加载"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = os.path.join(tmp_dir, "data.json")
            sample_data = [
                {"input": [1.0] * 226, "output": [0.0] * 225, "mask": [1] * 225}
            ]
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f)

            with patch('sys.argv', [
                'train_main.py', '--model', 'fc_v1', '--train-from-dir', json_path,
                '--epochs', '1'
            ]):
                with patch('ai.train_main.get_trainer') as mock_get:
                    mock_trainer = MagicMock()
                    mock_trainer.train_auto = MagicMock(return_value=0.5)
                    mock_get.return_value = mock_trainer

                    main()

                    mock_trainer.train_auto.assert_called_once()
                    call_args = mock_trainer.train_auto.call_args
                    self.assertEqual(call_args[1]['epochs'], 1)

    def test_train_from_dir_with_merge(self):
        """--train-from-dir 目录含 merge 文件和 sample 时全部加载"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 创建 merge 文件（1 条样本）
            merge_path = os.path.join(tmp_dir, "merge_fc_v1.json")
            with open(merge_path, 'w', encoding='utf-8') as f:
                json.dump([{"input": [1.0] * 226, "output": [0.0] * 225, "mask": [1] * 225}], f)

            # 创建个体 sample 目录（2 条样本）
            sample_dir = os.path.join(tmp_dir, "1")
            os.makedirs(sample_dir)
            with open(os.path.join(sample_dir, "sample.json"), 'w', encoding='utf-8') as f:
                json.dump([
                    {"input": [2.0] * 226, "output": [0.0] * 225, "mask": [1] * 225},
                    {"input": [3.0] * 226, "output": [0.0] * 225, "mask": [1] * 225},
                ], f)

            with patch('sys.argv', [
                'train_main.py', '--model', 'fc_v1', '--train-from-dir', tmp_dir,
                '--epochs', '1'
            ]):
                with patch('ai.train_main.get_trainer') as mock_get:
                    mock_trainer = MagicMock()
                    mock_trainer.train_auto = MagicMock(return_value=0.3)
                    mock_get.return_value = mock_trainer

                    main()

                    mock_trainer.train_auto.assert_called_once()
                    call_args = mock_trainer.train_auto.call_args
                    self.assertEqual(call_args[1]['epochs'], 1)

    def test_train_from_dir_empty_raises(self):
        """--train-from-dir 指向空目录时应报错退出"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch('sys.argv', [
                'train_main.py', '--model', 'fc_v1', '--train-from-dir', tmp_dir,
                '--epochs', '1'
            ]):
                with patch('ai.train_main.get_trainer') as mock_get:
                    mock_trainer = MagicMock()
                    mock_get.return_value = mock_trainer

                    with self.assertRaises(SystemExit):
                        main()

    def test_train_from_dir_with_mini_batch(self):
        """--train-from-dir 目录支持 mini-batch 训练"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = os.path.join(tmp_dir, "1")
            os.makedirs(sample_dir)
            with open(os.path.join(sample_dir, "sample.json"), 'w', encoding='utf-8') as f:
                json.dump([
                    {"input": [1.0] * 226, "output": [0.0] * 225, "mask": [1] * 225},
                ], f)

            with patch('sys.argv', [
                'train_main.py', '--model', 'fc_v1', '--train-from-dir', tmp_dir,
                '--epochs', '1', '--batch-size', '1'
            ]):
                with patch('ai.train_main.get_trainer') as mock_get:
                    mock_trainer = MagicMock()
                    mock_trainer.train_auto = MagicMock(return_value=0.2)
                    mock_get.return_value = mock_trainer

                    main()

                    mock_trainer.train_auto.assert_called_once()
                    call_kwargs = mock_trainer.train_auto.call_args[1]
                    self.assertEqual(call_kwargs['batch_size'], 1)
                    self.assertEqual(call_kwargs['epochs'], 1)

    def test_train_from_dir_mini_batch_enables_dense_log(self):
        """--train-from-dir + --batch-size 应自动启用密集日志（dense_log=True）"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = os.path.join(tmp_dir, "1")
            os.makedirs(sample_dir)
            with open(os.path.join(sample_dir, "sample.json"), 'w', encoding='utf-8') as f:
                json.dump([
                    {"input": [1.0] * 226, "output": [0.0] * 225, "mask": [1] * 225},
                ], f)

            with patch('sys.argv', [
                'train_main.py', '--model', 'fc_v1', '--train-from-dir', tmp_dir,
                '--epochs', '1', '--batch-size', '32'
            ]):
                with patch('ai.train_main.get_trainer') as mock_get:
                    mock_trainer = MagicMock()
                    mock_trainer.train_auto = MagicMock(return_value=0.1)
                    mock_get.return_value = mock_trainer

                    main()

                    call_kwargs = mock_trainer.train_auto.call_args[1]
                    self.assertTrue(call_kwargs.get('dense_log', False),
                                    "--train-from-dir + --batch-size 应传 dense_log=True")

    def test_train_from_dir_no_batch_passes_dense_log(self):
        """--train-from-dir 不指定 --batch-size 时，runners 仍传 dense_log=True 给 train_auto（内部走 train_batch 路径，train_batch 不使用 dense_log）"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = os.path.join(tmp_dir, "1")
            os.makedirs(sample_dir)
            with open(os.path.join(sample_dir, "sample.json"), 'w', encoding='utf-8') as f:
                json.dump([
                    {"input": [1.0] * 226, "output": [0.0] * 225, "mask": [1] * 225},
                ], f)

            with patch('sys.argv', [
                'train_main.py', '--model', 'fc_v1', '--train-from-dir', tmp_dir,
                '--epochs', '1'
            ]):
                with patch('ai.train_main.get_trainer') as mock_get:
                    mock_trainer = MagicMock()
                    mock_trainer.train_auto = MagicMock(return_value=0.3)
                    mock_get.return_value = mock_trainer

                    main()

                    mock_trainer.train_auto.assert_called_once()
                    # train_auto 统一接收 dense_log=True，但 batch_size=None 时
                    # 内部走 train_batch 路径（train_batch 不使用 dense_log）
                    call_kwargs = mock_trainer.train_auto.call_args[1]
                    self.assertIsNone(call_kwargs.get('batch_size'))


if __name__ == '__main__':
    unittest.main()
