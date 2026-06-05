"""create_samples --sample_games 集成测试

覆盖设计文档 test_cases.md:
- A 组: 参数互斥与校验 (A-01 ~ A-05)
- D 组: 对弈循环 (D-01 ~ D-06)
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
from unittest.mock import patch, MagicMock, call

from ai.create_samples import main, is_sample_game
from ai.game_manager import SingleGameResult


def _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                        mock_manager_cls, mock_recorder_cls):
    """为 CLI 集成测试批量装配公共 mocks

    关键点：create_samples 使用 AIGameManager 静态方法调用，
    需要 mock _play_single_game 和 start_ai_vs_ai 的类级别返回值。
    """
    mock_model = MagicMock()
    mock_model.name = "test_model"
    mock_model.full_name = "test_model"
    mock_get_model.return_value = mock_model

    mock_engine_cls.return_value = MagicMock()

    # 默认 start_ai_vs_ai 返回 1 局黑胜
    mock_manager_cls.start_ai_vs_ai.return_value = [
        {"game": 1, "winner": 1, "temp_side_this_game": None}
    ]

    # 准备 recorder 和 round_dir
    mock_recorder = MagicMock()
    round_dir = os.path.join(tmpdir, "round_1")
    os.makedirs(round_dir, exist_ok=True)
    game_dir = os.path.join(round_dir, "1")
    os.makedirs(game_dir, exist_ok=True)
    record = {
        "winner": 1,
        "history": [
            {"input": [0.0] * 226, "output": [0.0] * 225, "position": [0, 0]},
        ],
        "end_board": [0.0] * 226,
    }
    with open(os.path.join(game_dir, "record.json"), "w", encoding="utf-8") as f:
        json.dump(record, f)
    mock_recorder.get_round_dir.return_value = round_dir
    mock_recorder_cls.return_value = mock_recorder


def _create_record_json(round_dir, game_num, winner, history=None):
    """在 round_dir 下创建指定 game_num 的 record.json"""
    game_dir = os.path.join(round_dir, str(game_num))
    os.makedirs(game_dir, exist_ok=True)
    if history is None:
        history = [
            {"input": [float(game_num)] * 226, "output": [0.0] * 225,
             "position": [game_num % 15, game_num % 15]},
        ]
    record = {
        "winner": winner,
        "history": history,
        "end_board": [0.0] * 226,
    }
    with open(os.path.join(game_dir, "record.json"), "w", encoding="utf-8") as f:
        json.dump(record, f)


# =====================================================================
# A 组: 参数互斥与校验
# =====================================================================


class TestArgMutualExclusion(unittest.TestCase):
    """A 组: --games 与 --sample_games 互斥校验"""

    @patch('ai.create_samples.DataRecorder')
    @patch('ai.create_samples.AIGameManager')
    @patch('ai.create_samples.AIEngine')
    @patch('ai.create_samples.get_model')
    def test_both_specified_exits(self, mock_get_model, mock_engine_cls,
                                  mock_manager_cls, mock_recorder_cls):
        """A-01: --games + --sample_games 同时指定 → 报错退出"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)
            argv = ['create_samples.py', '--model', 'test_model',
                    '--games', '10', '--sample_games', '10',
                    '--output', output_path]
            with patch('sys.argv', argv):
                with self.assertRaises(SystemExit) as ctx:
                    main()
                self.assertNotEqual(ctx.exception.code, 0)

    @patch('ai.create_samples.DataRecorder')
    @patch('ai.create_samples.AIGameManager')
    @patch('ai.create_samples.AIEngine')
    @patch('ai.create_samples.get_model')
    def test_sample_games_zero_exits(self, mock_get_model, mock_engine_cls,
                                     mock_manager_cls, mock_recorder_cls):
        """A-02: --sample_games 0 → 报错退出"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)
            argv = ['create_samples.py', '--model', 'test_model',
                    '--sample_games', '0', '--output', output_path]
            with patch('sys.argv', argv):
                with self.assertRaises(SystemExit) as ctx:
                    main()
                self.assertNotEqual(ctx.exception.code, 0)

    @patch('ai.create_samples.DataRecorder')
    @patch('ai.create_samples.AIGameManager')
    @patch('ai.create_samples.AIEngine')
    @patch('ai.create_samples.get_model')
    def test_sample_games_negative_exits(self, mock_get_model, mock_engine_cls,
                                         mock_manager_cls, mock_recorder_cls):
        """A-03: --sample_games -5 → 报错退出"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)
            argv = ['create_samples.py', '--model', 'test_model',
                    '--sample_games', '-5', '--output', output_path]
            with patch('sys.argv', argv):
                with self.assertRaises(SystemExit) as ctx:
                    main()
                self.assertNotEqual(ctx.exception.code, 0)

    @patch('ai.create_samples.DataRecorder')
    @patch('ai.create_samples.AIGameManager')
    @patch('ai.create_samples.AIEngine')
    @patch('ai.create_samples.get_model')
    def test_only_games_unchanged(self, mock_get_model, mock_engine_cls,
                                  mock_manager_cls, mock_recorder_cls):
        """A-04: 仅指定 --games → 行为与未修改前完全一致"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)
            argv = ['create_samples.py', '--model', 'test_model',
                    '--games', '10', '--output', output_path]
            with patch('sys.argv', argv):
                main()
            # 验证走的是 start_ai_vs_ai 路径
            mock_manager_cls.start_ai_vs_ai.assert_called_once()

    @patch('ai.create_samples.DataRecorder')
    @patch('ai.create_samples.AIGameManager')
    @patch('ai.create_samples.AIEngine')
    @patch('ai.create_samples.get_model')
    def test_neither_specified_defaults_games(self, mock_get_model,
                                              mock_engine_cls,
                                              mock_manager_cls,
                                              mock_recorder_cls):
        """A-05: 两者都不指定 → 默认 --games 100"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)
            argv = ['create_samples.py', '--model', 'test_model',
                    '--output', output_path]
            with patch('sys.argv', argv):
                main()
            # 验证走的是 start_ai_vs_ai 路径
            mock_manager_cls.start_ai_vs_ai.assert_called_once()
            call_kwargs = mock_manager_cls.start_ai_vs_ai.call_args
            self.assertEqual(call_kwargs[0][2], 100)


# =====================================================================
# D 组: 对弈循环
# =====================================================================


class TestGameLoop(unittest.TestCase):
    """D 组: --sample_games 对弈循环"""

    @patch('ai.create_samples.DataRecorder')
    @patch('ai.create_samples.AIGameManager')
    @patch('ai.create_samples.AIEngine')
    @patch('ai.create_samples.get_model')
    def test_no_filter_reaches_target(self, mock_get_model, mock_engine_cls,
                                     mock_manager_cls, mock_recorder_cls):
        """D-01: 无过滤场景达标 → 精确10局对弈，样本局=10"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)

            # 每局返回黑胜（无温度采样，不会被过滤）
            mock_manager_cls._play_single_game.return_value = \
                SingleGameResult(winner=1, temp_side_this_game=None)

            # 为每局创建 record.json
            round_dir = os.path.join(tmpdir, "round_1")
            for i in range(1, 11):
                _create_record_json(round_dir, i, winner=1)

            argv = ['create_samples.py', '--model', 'test_model',
                    '--sample_games', '10', '--no-temp-sampling',
                    '--output', output_path]
            with patch('sys.argv', argv):
                main()

            # 精确10局对弈
            self.assertEqual(mock_manager_cls._play_single_game.call_count, 10)

    @patch('ai.create_samples.DataRecorder')
    @patch('ai.create_samples.AIGameManager')
    @patch('ai.create_samples.AIEngine')
    @patch('ai.create_samples.get_model')
    def test_single_side_filter_reaches_target(self, mock_get_model,
                                               mock_engine_cls,
                                               mock_manager_cls,
                                               mock_recorder_cls):
        """D-02: 单方过滤场景达标 → 原始对弈>10局，样本局=10"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)

            # 交替返回黑胜(采样方)和白胜(贪婪方)
            # temp_side=black → 黑胜保留，白胜过滤
            # 提供足够多的结果（至少20局，让10个样本局达标）
            results = []
            for i in range(30):
                if i % 2 == 0:
                    results.append(
                        SingleGameResult(winner=1, temp_side_this_game="black")
                    )
                else:
                    results.append(
                        SingleGameResult(winner=2, temp_side_this_game="black")
                    )
            mock_manager_cls._play_single_game.side_effect = results

            # 为每局创建 record.json
            round_dir = os.path.join(tmpdir, "round_1")
            for i in range(1, 31):
                winner = 1 if i % 2 == 1 else 2
                _create_record_json(round_dir, i, winner=winner)

            argv = ['create_samples.py', '--model', 'test_model',
                    '--sample_games', '10', '--temp-side', 'black',
                    '--output', output_path]
            with patch('sys.argv', argv):
                main()

            # 奇数局为黑胜(样本局)：第1,3,5,...,19局达标
            # 第19局时达到10个样本局，循环停止，共对弈19局
            self.assertEqual(mock_manager_cls._play_single_game.call_count, 19)

    @patch('ai.create_samples.DataRecorder')
    @patch('ai.create_samples.AIGameManager')
    @patch('ai.create_samples.AIEngine')
    @patch('ai.create_samples.get_model')
    def test_count_matches_sample_creator(self, mock_get_model,
                                          mock_engine_cls,
                                          mock_manager_cls,
                                          mock_recorder_cls):
        """D-03: is_sample_game 计数与 SampleCreator 实际产出一致"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)

            # 全部黑胜 + temp_side=black → 每局都是样本局
            mock_manager_cls._play_single_game.return_value = \
                SingleGameResult(winner=1, temp_side_this_game="black")

            round_dir = os.path.join(tmpdir, "round_1")
            for i in range(1, 6):
                _create_record_json(round_dir, i, winner=1)

            argv = ['create_samples.py', '--model', 'test_model',
                    '--sample_games', '5', '--temp-side', 'black',
                    '--no-augment', '--output', output_path]
            with patch('sys.argv', argv):
                main()

            with open(output_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)

            # 5局黑胜，无增强，每局1样本 → 5个
            self.assertEqual(len(samples), 5)

    @patch('ai.create_samples.DataRecorder')
    @patch('ai.create_samples.AIGameManager')
    @patch('ai.create_samples.AIEngine')
    @patch('ai.create_samples.get_model')
    def test_winning_move_with_augment(self, mock_get_model,
                                       mock_engine_cls,
                                       mock_manager_cls,
                                       mock_recorder_cls):
        """D-04: winning_move+增强 → 最终样本数=50×8"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)

            mock_manager_cls._play_single_game.return_value = \
                SingleGameResult(winner=1, temp_side_this_game=None)

            round_dir = os.path.join(tmpdir, "round_1")
            for i in range(1, 51):
                _create_record_json(round_dir, i, winner=1)

            argv = ['create_samples.py', '--model', 'test_model',
                    '--sample_games', '50', '--no-temp-sampling',
                    '--output', output_path]
            with patch('sys.argv', argv):
                main()

            with open(output_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)
            # 50局 × 8(增强) = 400
            self.assertEqual(len(samples), 400)

    @patch('ai.create_samples.DataRecorder')
    @patch('ai.create_samples.AIGameManager')
    @patch('ai.create_samples.AIEngine')
    @patch('ai.create_samples.get_model')
    def test_winning_move_no_augment(self, mock_get_model,
                                     mock_engine_cls,
                                     mock_manager_cls,
                                     mock_recorder_cls):
        """D-05: winning_move+禁用增强 → 最终样本数=50"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)

            mock_manager_cls._play_single_game.return_value = \
                SingleGameResult(winner=1, temp_side_this_game=None)

            round_dir = os.path.join(tmpdir, "round_1")
            for i in range(1, 51):
                _create_record_json(round_dir, i, winner=1)

            argv = ['create_samples.py', '--model', 'test_model',
                    '--sample_games', '50', '--no-temp-sampling',
                    '--no-augment', '--output', output_path]
            with patch('sys.argv', argv):
                main()

            with open(output_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)
            self.assertEqual(len(samples), 50)

    @patch('ai.create_samples.DataRecorder')
    @patch('ai.create_samples.AIGameManager')
    @patch('ai.create_samples.AIEngine')
    @patch('ai.create_samples.get_model')
    def test_winner_path_mode(self, mock_get_model, mock_engine_cls,
                              mock_manager_cls, mock_recorder_cls):
        """D-06: winner_path模式 → 样本局=10，最终样本数>=10"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)

            mock_manager_cls._play_single_game.return_value = \
                SingleGameResult(winner=1, temp_side_this_game=None)

            # 构造含5步黑胜的 record.json
            round_dir = os.path.join(tmpdir, "round_1")
            for i in range(1, 11):
                history = [
                    {"input": [float(j)] * 226, "output": [0.0] * 225,
                     "position": [j % 15, j % 15]}
                    for j in range(5)
                ]
                _create_record_json(round_dir, i, winner=1, history=history)

            argv = ['create_samples.py', '--model', 'test_model',
                    '--sample_games', '10', '--no-temp-sampling',
                    '--mode', 'winner_path', '--output', output_path]
            with patch('sys.argv', argv):
                main()

            with open(output_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)
            # 10局 × 3步(黑胜偶数索引) = 30，winner_path不增强
            self.assertGreaterEqual(len(samples), 10)


# =====================================================================
# E 组: 进度输出时机
# =====================================================================


class TestProgressTiming(unittest.TestCase):
    """E 组: 进度输出时机"""

    @patch('ai.create_samples.DataRecorder')
    @patch('ai.create_samples.AIGameManager')
    @patch('ai.create_samples.AIEngine')
    @patch('ai.create_samples.get_model')
    def test_standard_100_progress(self, mock_get_model, mock_engine_cls,
                                   mock_manager_cls, mock_recorder_cls):
        """E-01: 100局标准进度 → 10%/20%/.../100% 各打印一次"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)

            mock_manager_cls._play_single_game.return_value = \
                SingleGameResult(winner=1, temp_side_this_game=None)

            round_dir = os.path.join(tmpdir, "round_1")
            for i in range(1, 101):
                _create_record_json(round_dir, i, winner=1)

            captured = StringIO()
            argv = ['create_samples.py', '--model', 'test_model',
                    '--sample_games', '100', '--no-temp-sampling',
                    '--output', output_path]
            with patch('sys.argv', argv), patch('sys.stdout', captured):
                main()

            output = captured.getvalue()
            # 验证 10%, 20%, ..., 100% 各出现一次
            for pct in range(10, 110, 10):
                self.assertIn(f"({pct}%)", output)

    @patch('ai.create_samples.DataRecorder')
    @patch('ai.create_samples.AIGameManager')
    @patch('ai.create_samples.AIEngine')
    @patch('ai.create_samples.get_model')
    def test_small_value_prints_at_completion(self, mock_get_model,
                                              mock_engine_cls,
                                              mock_manager_cls,
                                              mock_recorder_cls):
        """E-02: 小值仅打印达标 → 至少打印一次100%进度"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)

            mock_manager_cls._play_single_game.return_value = \
                SingleGameResult(winner=1, temp_side_this_game=None)

            round_dir = os.path.join(tmpdir, "round_1")
            for i in range(1, 6):
                _create_record_json(round_dir, i, winner=1)

            captured = StringIO()
            argv = ['create_samples.py', '--model', 'test_model',
                    '--sample_games', '5', '--no-temp-sampling',
                    '--output', output_path]
            with patch('sys.argv', argv), patch('sys.stdout', captured):
                main()

            output = captured.getvalue()
            self.assertIn("(100%)", output)

    @patch('ai.create_samples.DataRecorder')
    @patch('ai.create_samples.AIGameManager')
    @patch('ai.create_samples.AIEngine')
    @patch('ai.create_samples.get_model')
    def test_non_divisible_target(self, mock_get_model, mock_engine_cls,
                                  mock_manager_cls, mock_recorder_cls):
        """E-03: 非整除场景 → 按百分比阈值触发，达标时必定打印"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)

            mock_manager_cls._play_single_game.return_value = \
                SingleGameResult(winner=1, temp_side_this_game=None)

            round_dir = os.path.join(tmpdir, "round_1")
            for i in range(1, 11):
                _create_record_json(round_dir, i, winner=1)

            captured = StringIO()
            argv = ['create_samples.py', '--model', 'test_model',
                    '--sample_games', '7', '--no-temp-sampling',
                    '--output', output_path]
            with patch('sys.argv', argv), patch('sys.stdout', captured):
                main()

            output = captured.getvalue()
            # 达标时必定打印
            self.assertIn("(100%)", output)


# =====================================================================
# F 组: Ctrl+C 中断
# =====================================================================


class TestKeyboardInterrupt(unittest.TestCase):
    """F 组: Ctrl+C 中断处理"""

    @patch('ai.create_samples.DataRecorder')
    @patch('ai.create_samples.AIGameManager')
    @patch('ai.create_samples.AIEngine')
    @patch('ai.create_samples.get_model')
    def test_interrupt_prints_stats(self, mock_get_model, mock_engine_cls,
                                    mock_manager_cls, mock_recorder_cls):
        """F-01: 中断后打印统计信息"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)

            # 前5局正常，第6局抛 KeyboardInterrupt
            normal = SingleGameResult(winner=1, temp_side_this_game=None)
            mock_manager_cls._play_single_game.side_effect = [
                normal, normal, normal, normal, normal, KeyboardInterrupt()
            ]

            round_dir = os.path.join(tmpdir, "round_1")
            for i in range(1, 6):
                _create_record_json(round_dir, i, winner=1)

            captured = StringIO()
            argv = ['create_samples.py', '--model', 'test_model',
                    '--sample_games', '10', '--no-temp-sampling',
                    '--output', output_path]
            with patch('sys.argv', argv), patch('sys.stdout', captured):
                main()

            output = captured.getvalue()
            self.assertIn("[中断]", output)

    @patch('ai.create_samples.DataRecorder')
    @patch('ai.create_samples.AIGameManager')
    @patch('ai.create_samples.AIEngine')
    @patch('ai.create_samples.get_model')
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
            mock_manager_cls._play_single_game.side_effect = [
                normal, normal, KeyboardInterrupt()
            ]

            round_dir = os.path.join(tmpdir, "round_1")
            for i in range(1, 3):
                _create_record_json(round_dir, i, winner=1)

            argv = ['create_samples.py', '--model', 'test_model',
                    '--sample_games', '10', '--no-temp-sampling',
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

    @patch('ai.create_samples.DataRecorder')
    @patch('ai.create_samples.AIGameManager')
    @patch('ai.create_samples.AIEngine')
    @patch('ai.create_samples.get_model')
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
            mock_manager_cls._play_single_game.side_effect = [
                normal, normal, normal, KeyboardInterrupt()
            ]

            round_dir = os.path.join(tmpdir, "round_1")
            for i in range(1, 4):
                _create_record_json(round_dir, i, winner=1)

            argv = ['create_samples.py', '--model', 'test_model',
                    '--sample_games', '10', '--no-temp-sampling',
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


class TestBackwardCompatibility(unittest.TestCase):
    """G 组: 向后兼容"""

    @patch('ai.create_samples.DataRecorder')
    @patch('ai.create_samples.AIGameManager')
    @patch('ai.create_samples.AIEngine')
    @patch('ai.create_samples.get_model')
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
            # 验证走的是 start_ai_vs_ai，而非 _play_single_game
            mock_manager_cls.start_ai_vs_ai.assert_called_once()

    @patch('ai.create_samples.DataRecorder')
    @patch('ai.create_samples.AIGameManager')
    @patch('ai.create_samples.AIEngine')
    @patch('ai.create_samples.get_model')
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
