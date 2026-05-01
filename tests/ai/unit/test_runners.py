"""runners.py 单元测试

直接测试 ai/runners.py 中的公共函数，覆盖各运行模式的核心逻辑路径。
"""

import json
import os
import sys
import tempfile
import shutil
import unittest
import argparse
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from ai.runners import (
    run_train_from_dir, run_cycle,
    discover_sample_dirs, _create_temp_sample_dir,
)


def _make_args(**overrides) -> argparse.Namespace:
    """构造测试用 args Namespace，提供合理的默认值"""
    defaults = {
        'model': 'fc_v1',
        'cycle': False,
        'train_from_dir': None,
        'rounds': 10,
        'games': 100,
        'epochs': 1000,
        'rate': 0.9,
        'fresh': False,
        'batch_size': None,
        'shuffle': False,
        'lr_scheduler': None,
        'lr_step_size': 500,
        'lr_gamma': 0.9,
        'lr_patience': 100,
        'lr_factor': 0.5,
        'val_ratio': 0.0,
        'no_temp_sampling': False,
        'temperature': 1.0,
        'temp_moves': 12,
        'temp_side': 'both',
        'temp_seed': None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestCreateTempSampleDir(unittest.TestCase):
    """测试 _create_temp_sample_dir 辅助函数"""

    def test_creates_dir_with_sample_json(self):
        """应创建包含 sample.json 的临时目录"""
        sample_data = [{"input": [1.0], "output": [0.0], "mask": [1]}]
        tmp_dir = _create_temp_sample_dir(sample_data)
        try:
            self.assertTrue(os.path.isdir(tmp_dir))
            sample_path = os.path.join(tmp_dir, "sample.json")
            self.assertTrue(os.path.exists(sample_path))
            with open(sample_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            self.assertEqual(loaded, sample_data)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_empty_data_still_creates_dir(self):
        """空数据列表也应创建目录"""
        tmp_dir = _create_temp_sample_dir([])
        try:
            self.assertTrue(os.path.isdir(tmp_dir))
            sample_path = os.path.join(tmp_dir, "sample.json")
            self.assertTrue(os.path.exists(sample_path))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestRunTrainFromDir(unittest.TestCase):
    """测试 run_train_from_dir 运行模式"""

    def test_nonexistent_path_exits(self):
        """不存在的路径应 sys.exit(1)"""
        args = _make_args(train_from_dir='/nonexistent/path/xyz')
        trainer = MagicMock()
        with self.assertRaises(SystemExit):
            run_train_from_dir(args, trainer)

    def test_empty_json_file_exits(self):
        """空 .json 文件应 sys.exit(1)"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = os.path.join(tmp_dir, "empty.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump([], f)

            args = _make_args(train_from_dir=json_path)
            trainer = MagicMock()
            with self.assertRaises(SystemExit):
                run_train_from_dir(args, trainer)

    def test_json_file_trains_with_batch(self):
        """.json 文件 + batch_size 应调用 train_auto（内部走 mini-batch 路径）"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = os.path.join(tmp_dir, "data.json")
            sample_data = [{"input": [1.0] * 226, "output": [0.0] * 225, "mask": [1] * 225}]
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f)

            args = _make_args(train_from_dir=json_path, epochs=1, batch_size=16, shuffle=True)
            trainer = MagicMock()
            trainer.train_auto.return_value = 0.5

            run_train_from_dir(args, trainer)

            trainer.train_auto.assert_called_once()
            call_kwargs = trainer.train_auto.call_args[1]
            self.assertEqual(call_kwargs['batch_size'], 16)
            self.assertTrue(call_kwargs['shuffle'])
            self.assertTrue(call_kwargs['dense_log'])

    def test_directory_trains_with_train_batch(self):
        """目录路径 + 无 batch_size 应调用 train_auto（内部走全量更新路径）"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sub_dir = os.path.join(tmp_dir, "1")
            os.makedirs(sub_dir)
            with open(os.path.join(sub_dir, "sample.json"), 'w', encoding='utf-8') as f:
                json.dump([{"input": [1.0] * 226, "output": [0.0] * 225, "mask": [1] * 225}], f)

            args = _make_args(train_from_dir=tmp_dir, epochs=1)
            trainer = MagicMock()
            trainer.train_auto.return_value = 0.3

            run_train_from_dir(args, trainer)

            trainer.train_auto.assert_called_once()
            call_kwargs = trainer.train_auto.call_args[1]
            self.assertIsNone(call_kwargs['batch_size'])

    @patch('shutil.rmtree')
    def test_cleanup_on_success(self, mock_rmtree):
        """训练成功后应清理临时目录"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = os.path.join(tmp_dir, "data.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump([{"input": [1.0], "output": [0.0], "mask": [1]}], f)

            args = _make_args(train_from_dir=json_path)
            trainer = MagicMock()
            trainer.train_auto.return_value = 0.5

            with patch('ai.runners._create_temp_sample_dir', return_value='/fake/tmp/dir'):
                run_train_from_dir(args, trainer)

            mock_rmtree.assert_any_call('/fake/tmp/dir', ignore_errors=True)

    @patch('shutil.rmtree')
    def test_cleanup_on_failure(self, mock_rmtree):
        """训练失败后也应清理临时目录"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = os.path.join(tmp_dir, "data.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump([{"input": [1.0], "output": [0.0], "mask": [1]}], f)

            args = _make_args(train_from_dir=json_path)
            trainer = MagicMock()
            trainer.train_auto.side_effect = RuntimeError("boom")

            with patch('ai.runners._create_temp_sample_dir', return_value='/fake/tmp/dir'):
                with self.assertRaises(SystemExit):
                    run_train_from_dir(args, trainer)

            mock_rmtree.assert_any_call('/fake/tmp/dir', ignore_errors=True)


class TestRunCycle(unittest.TestCase):
    """测试 run_cycle 运行模式"""

    def test_calls_train_cycle(self):
        """应调用 trainer.train_cycle 并传递参数"""
        args = _make_args(cycle=True, rounds=5, games=20, epochs=100,
                          rate=0.95, batch_size=32, shuffle=True)
        trainer = MagicMock()

        run_cycle(args, trainer)

        trainer.train_cycle.assert_called_once()
        call_kwargs = trainer.train_cycle.call_args.kwargs
        self.assertEqual(call_kwargs['rounds'], 5)
        self.assertEqual(call_kwargs['games'], 20)
        self.assertEqual(call_kwargs['epochs'], 100)
        self.assertEqual(call_kwargs['rate'], 0.95)
        self.assertEqual(call_kwargs['batch_size'], 32)
        self.assertTrue(call_kwargs['shuffle'])
        self.assertEqual(call_kwargs['val_ratio'], 0.0)
        # 默认启用温度采样，temp_config 为 TempConfig 实例
        self.assertIn('temp_config', call_kwargs)
        temp_config = call_kwargs['temp_config']
        self.assertIsNotNone(temp_config)
        self.assertEqual(temp_config.temperature, 1.0)
        self.assertEqual(temp_config.temp_moves, 12)
        self.assertEqual(temp_config.temp_side, 'both')
        self.assertIsNone(temp_config.seed)

    def test_no_temp_sampling_passes_none(self):
        """--no-temp-sampling 应使 temp_config=None"""
        args = _make_args(cycle=True, no_temp_sampling=True)
        trainer = MagicMock()

        run_cycle(args, trainer)

        call_kwargs = trainer.train_cycle.call_args.kwargs
        self.assertIsNone(call_kwargs['temp_config'])

    def test_not_implemented_exits(self):
        """NotImplementedError 应导致 sys.exit(1)"""
        args = _make_args(cycle=True)
        trainer = MagicMock()
        trainer.train_cycle.side_effect = NotImplementedError("not yet")

        with self.assertRaises(SystemExit):
            run_cycle(args, trainer)

    def test_generic_exception_exits(self):
        """通用异常也应导致 sys.exit(1)"""
        args = _make_args(cycle=True)
        trainer = MagicMock()
        trainer.train_cycle.side_effect = RuntimeError("boom")

        with self.assertRaises(SystemExit):
            run_cycle(args, trainer)


if __name__ == '__main__':
    unittest.main()
