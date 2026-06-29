"""create_samples --sample_games 集成测试

覆盖设计文档 test_cases.md:
- A 组: 参数互斥与校验 (A-01 ~ A-07)
- D 组: 对弈循环 (D-01 ~ D-06)

E/F/G 组已拆分到 test_sample_games_cli_progress.py。
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
    需要 mock play_single_game 和 start_ai_vs_ai 的类级别返回值。
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
    # 构造变异步：position [0,0] 位置概率 0.1，其他位置有 0.9 的最大值
    variant_output = [0.0] * 225
    variant_output[0] = 0.1  # position [0,0] → idx 0，不是最大值
    variant_output[1] = 0.9  # 最大值在 idx 1
    record = {
        "winner": 1,
        "history": [
            {"input": [0.0] * 226, "output": variant_output, "position": [0, 0]},
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
        # 构造变异步：position 位置概率 0.1，相邻位置有 0.9 的最大值
        idx = (game_num % 15) * 15 + (game_num % 15)
        variant_output = [0.0] * 225
        variant_output[idx] = 0.1
        variant_output[(idx + 1) % 225] = 0.9
        history = [
            {"input": [float(game_num)] * 226, "output": variant_output,
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


@patch('ai.create_samples.DataRecorder')
@patch('ai.create_samples.AIGameManager')
@patch('ai.create_samples.SinglePolicyAiEngine')
@patch('ai.create_samples.get_model')
class TestArgMutualExclusion(unittest.TestCase):
    """A 组: --games 与 --sample_games 互斥校验"""

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

    def test_games_zero_exits(self, mock_get_model, mock_engine_cls,
                              mock_manager_cls, mock_recorder_cls):
        """A-06: --games 0 → 报错退出"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)
            argv = ['create_samples.py', '--model', 'test_model',
                    '--games', '0', '--output', output_path]
            with patch('sys.argv', argv):
                with self.assertRaises(SystemExit) as ctx:
                    main()
                self.assertNotEqual(ctx.exception.code, 0)

    def test_games_negative_exits(self, mock_get_model, mock_engine_cls,
                                  mock_manager_cls, mock_recorder_cls):
        """A-07: --games -5 → 报错退出"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)
            argv = ['create_samples.py', '--model', 'test_model',
                    '--games', '-5', '--output', output_path]
            with patch('sys.argv', argv):
                with self.assertRaises(SystemExit) as ctx:
                    main()
                self.assertNotEqual(ctx.exception.code, 0)


# =====================================================================
# D 组: 对弈循环
# =====================================================================


@patch('ai.create_samples.DataRecorder')
@patch('ai.create_samples.AIGameManager')
@patch('ai.create_samples.SinglePolicyAiEngine')
@patch('ai.create_samples.get_model')
class TestGameLoop(unittest.TestCase):
    """D 组: --sample_games 对弈循环"""

    def test_no_filter_reaches_target(self, mock_get_model, mock_engine_cls,
                                     mock_manager_cls, mock_recorder_cls):
        """D-01: 无过滤场景达标 → 精确10局对弈，样本局=10"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)

            # 每局返回黑胜（无温度采样，不会被过滤）
            mock_manager_cls.play_single_game.return_value = \
                SingleGameResult(winner=1, temp_side_this_game=None)

            # 为每局创建 record.json
            round_dir = os.path.join(tmpdir, "round_1")
            for i in range(1, 11):
                _create_record_json(round_dir, i, winner=1)

            argv = ['create_samples.py', '--model', 'test_model',
                    '--sample_games', '10',
                    '--output', output_path]
            with patch('sys.argv', argv):
                main()

            # 精确10局对弈
            self.assertEqual(mock_manager_cls.play_single_game.call_count, 10)

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
            mock_manager_cls.play_single_game.side_effect = results

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
            self.assertEqual(mock_manager_cls.play_single_game.call_count, 19)

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
            mock_manager_cls.play_single_game.return_value = \
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

    def test_winning_move_with_augment(self, mock_get_model,
                                       mock_engine_cls,
                                       mock_manager_cls,
                                       mock_recorder_cls):
        """D-04: winning_move+增强 → 最终样本数=50×8"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)

            mock_manager_cls.play_single_game.return_value = \
                SingleGameResult(winner=1, temp_side_this_game=None)

            round_dir = os.path.join(tmpdir, "round_1")
            for i in range(1, 51):
                _create_record_json(round_dir, i, winner=1)

            argv = ['create_samples.py', '--model', 'test_model',
                    '--sample_games', '50',
                    '--output', output_path]
            with patch('sys.argv', argv):
                main()

            with open(output_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)
            # 50局 × 8(增强) = 400
            self.assertEqual(len(samples), 400)

    def test_winning_move_no_augment(self, mock_get_model,
                                     mock_engine_cls,
                                     mock_manager_cls,
                                     mock_recorder_cls):
        """D-05: winning_move+禁用增强 → 最终样本数=50"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)

            mock_manager_cls.play_single_game.return_value = \
                SingleGameResult(winner=1, temp_side_this_game=None)

            round_dir = os.path.join(tmpdir, "round_1")
            for i in range(1, 51):
                _create_record_json(round_dir, i, winner=1)

            argv = ['create_samples.py', '--model', 'test_model',
                    '--sample_games', '50',
                    '--no-augment', '--output', output_path]
            with patch('sys.argv', argv):
                main()

            with open(output_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)
            self.assertEqual(len(samples), 50)

    def test_winner_path_mode(self, mock_get_model, mock_engine_cls,
                              mock_manager_cls, mock_recorder_cls):
        """D-06: winner_path模式 → 样本局=10，最终样本数>=10"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)

            mock_manager_cls.play_single_game.return_value = \
                SingleGameResult(winner=1, temp_side_this_game=None)

            # 构造含5步黑胜的 record.json（每步均为变异步）
            round_dir = os.path.join(tmpdir, "round_1")
            for i in range(1, 11):
                history = []
                for j in range(5):
                    idx_j = (j % 15) * 15 + (j % 15)
                    v_output = [0.0] * 225
                    v_output[idx_j] = 0.1
                    v_output[(idx_j + 1) % 225] = 0.9
                    history.append(
                        {"input": [float(j)] * 226, "output": v_output,
                         "position": [j % 15, j % 15]}
                    )
                _create_record_json(round_dir, i, winner=1, history=history)

            argv = ['create_samples.py', '--model', 'test_model',
                    '--sample_games', '10',
                    '--output', output_path]
            with patch('sys.argv', argv):
                main()

            with open(output_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)
            # 10局 × 3步(黑胜偶数索引) = 30，winner_path不增强
            self.assertGreaterEqual(len(samples), 10)


if __name__ == "__main__":
    unittest.main()
