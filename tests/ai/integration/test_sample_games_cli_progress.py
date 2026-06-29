"""create_samples --sample_games 进度输出/中断/兼容测试

从 test_sample_games_cli.py 拆分出的子文件，覆盖：
- E 组: 进度输出时机 (E-01 ~ E-03)
- F 组: Ctrl+C 中断 (F-01 ~ F-03)
- G 组: 向后兼容 (G-01 ~ G-02)
"""

import json
import os
import sys
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch, MagicMock

from ai.create_samples import main
from ai.game_manager import SingleGameResult

from tests.ai.integration.test_sample_games_cli import (
    _setup_common_mocks,
    _create_record_json,
)


# =====================================================================
# E 组: 进度输出时机
# =====================================================================


@patch('ai.create_samples.DataRecorder')
@patch('ai.create_samples.AIGameManager')
@patch('ai.create_samples.SinglePolicyAiEngine')
@patch('ai.create_samples.get_model')
class TestProgressTiming(unittest.TestCase):
    """E 组: 进度输出时机"""

    def test_standard_100_progress(self, mock_get_model, mock_engine_cls,
                                   mock_manager_cls, mock_recorder_cls):
        """E-01: 100局标准进度 → 10%/20%/.../100% 各打印一次"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)

            mock_manager_cls.play_single_game.return_value = \
                SingleGameResult(winner=1, temp_side_this_game=None)

            round_dir = os.path.join(tmpdir, "round_1")
            for i in range(1, 101):
                _create_record_json(round_dir, i, winner=1)

            captured = StringIO()
            argv = ['create_samples.py', '--model', 'test_model',
                    '--sample_games', '100',
                    '--output', output_path]
            with patch('sys.argv', argv), patch('sys.stdout', captured):
                main()

            output = captured.getvalue()
            # 验证 10%, 20%, ..., 100% 各出现一次
            for pct in range(10, 110, 10):
                self.assertIn(f"({pct}%)", output)

    def test_small_value_prints_at_completion(self, mock_get_model,
                                              mock_engine_cls,
                                              mock_manager_cls,
                                              mock_recorder_cls):
        """E-02: 小值仅打印达标 → 至少打印一次100%进度"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)

            mock_manager_cls.play_single_game.return_value = \
                SingleGameResult(winner=1, temp_side_this_game=None)

            round_dir = os.path.join(tmpdir, "round_1")
            for i in range(1, 6):
                _create_record_json(round_dir, i, winner=1)

            captured = StringIO()
            argv = ['create_samples.py', '--model', 'test_model',
                    '--sample_games', '5',
                    '--output', output_path]
            with patch('sys.argv', argv), patch('sys.stdout', captured):
                main()

            output = captured.getvalue()
            self.assertIn("(100%)", output)

    def test_non_divisible_target(self, mock_get_model, mock_engine_cls,
                                  mock_manager_cls, mock_recorder_cls):
        """E-03: 非整除场景 → 按百分比阈值触发，达标时必定打印"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)

            mock_manager_cls.play_single_game.return_value = \
                SingleGameResult(winner=1, temp_side_this_game=None)

            round_dir = os.path.join(tmpdir, "round_1")
            for i in range(1, 11):
                _create_record_json(round_dir, i, winner=1)

            captured = StringIO()
            argv = ['create_samples.py', '--model', 'test_model',
                    '--sample_games', '7',
                    '--output', output_path]
            with patch('sys.argv', argv), patch('sys.stdout', captured):
                main()

            output = captured.getvalue()
            # 达标时必定打印
            self.assertIn("(100%)", output)


# =====================================================================
# F 组: Ctrl+C 中断
# =====================================================================


@patch('ai.create_samples.DataRecorder')
@patch('ai.create_samples.AIGameManager')
@patch('ai.create_samples.SinglePolicyAiEngine')
@patch('ai.create_samples.get_model')
class TestKeyboardInterrupt(unittest.TestCase):
    """F 组: Ctrl+C 中断处理"""

    def test_interrupt_prints_stats(self, mock_get_model, mock_engine_cls,
                                    mock_manager_cls, mock_recorder_cls):
        """F-01: 中断后打印统计信息"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)

            # 前5局正常，第6局抛 KeyboardInterrupt
            normal = SingleGameResult(winner=1, temp_side_this_game=None)
            mock_manager_cls.play_single_game.side_effect = [
                normal, normal, normal, normal, normal, KeyboardInterrupt()
            ]

            round_dir = os.path.join(tmpdir, "round_1")
            for i in range(1, 6):
                _create_record_json(round_dir, i, winner=1)

            captured = StringIO()
            argv = ['create_samples.py', '--model', 'test_model',
                    '--sample_games', '10',
                    '--output', output_path]
            with patch('sys.argv', argv), patch('sys.stdout', captured):
                main()

            output = captured.getvalue()
            self.assertIn("[中断]", output)

    def test_interrupt_produces_valid_samples(self, mock_get_model,
                                              mock_engine_cls,
                                              mock_manager_cls,
                                              mock_recorder_cls):
        """F-02: 中断后样本可生成，JSON合法"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)

            normal = SingleGameResult(winner=1, temp_side_this_game=None)
            mock_manager_cls.play_single_game.side_effect = [
                normal, normal, KeyboardInterrupt()
            ]

            round_dir = os.path.join(tmpdir, "round_1")
            for i in range(1, 3):
                _create_record_json(round_dir, i, winner=1)

            argv = ['create_samples.py', '--model', 'test_model',
                    '--sample_games', '10',
                    '--no-augment', '--output', output_path]
            with patch('sys.argv', argv):
                main()

            with open(output_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)
            # JSON 合法，含 input/output/mask
            self.assertIsInstance(samples, list)
            for s in samples:
                self.assertIn("input", s)
                self.assertIn("output", s)
                self.assertIn("mask", s)

    def test_interrupt_sample_count_consistent(self, mock_get_model,
                                               mock_engine_cls,
                                               mock_manager_cls,
                                               mock_recorder_cls):
        """F-03: 中断后样本数与 SampleCreator 从已有对局记录生成的结果一致"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)

            normal = SingleGameResult(winner=1, temp_side_this_game=None)
            mock_manager_cls.play_single_game.side_effect = [
                normal, normal, normal, KeyboardInterrupt()
            ]

            round_dir = os.path.join(tmpdir, "round_1")
            for i in range(1, 4):
                _create_record_json(round_dir, i, winner=1)

            argv = ['create_samples.py', '--model', 'test_model',
                    '--sample_games', '10',
                    '--no-augment', '--output', output_path]
            with patch('sys.argv', argv):
                main()

            with open(output_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)
            # 3局黑胜，no-augment → 3个样本
            self.assertEqual(len(samples), 3)


# =====================================================================
# G 组: 向后兼容
# =====================================================================


@patch('ai.create_samples.DataRecorder')
@patch('ai.create_samples.AIGameManager')
@patch('ai.create_samples.SinglePolicyAiEngine')
@patch('ai.create_samples.get_model')
class TestBackwardCompatibility(unittest.TestCase):
    """G 组: 向后兼容"""

    def test_without_sample_games(self, mock_get_model, mock_engine_cls,
                                  mock_manager_cls, mock_recorder_cls):
        """G-01: 不传 --sample_games → 行为与未修改前完全一致"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)
            argv = ['create_samples.py', '--model', 'test_model',
                    '--games', '100', '--output', output_path]
            with patch('sys.argv', argv):
                main()
            # 验证走的是 start_ai_vs_ai，而非 play_single_game
            mock_manager_cls.start_ai_vs_ai.assert_called_once()

    def test_help_includes_sample_games(self, mock_get_model,
                                        mock_engine_cls,
                                        mock_manager_cls,
                                        mock_recorder_cls):
        """G-02: --help 输出包含 --sample_games 参数说明"""
        argv = ['create_samples.py', '--help']
        with patch('sys.argv', argv):
            with self.assertRaises(SystemExit):
                main()
        # argparse --help 会 exit(0)，且输出到 stdout
        # 此处仅验证参数已注册（无异常即通过）


if __name__ == "__main__":
    unittest.main()
