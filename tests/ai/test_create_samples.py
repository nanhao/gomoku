"""create_samples.py 集成测试

覆盖 CLI 脚本的完整流程：
- 输出文件存在且格式正确
- 输出样本数与对局数一致（排除平局）
- is_variant_move 过滤后样本数正确
- CLI 参数测试（--mode/--no-temp-sampling 移除，--temp-side 默认值变更）
- 统计输出测试

注意：使用 mock 替代真实模型对弈，避免依赖 GPU/模型权重。
所有测试数据使用变异步（output[idx] != max(output)），确保 move_filter 不过滤。
"""

import json
import os
import sys
import argparse
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.create_samples import main


@patch('ai.create_samples.DataRecorder')
@patch('ai.create_samples.AIGameManager')
@patch('ai.create_samples.SinglePolicyAiEngine')
@patch('ai.create_samples.get_model')
class TestCliCreatesOutputFile(unittest.TestCase):
    """执行后输出文件存在且格式正确"""

    def test_output_file_exists_and_valid(self, mock_get_model, mock_engine_cls, mock_manager_cls, mock_recorder_cls):
        """CLI 执行后输出文件存在且包含有效样本"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")

            # mock 模型
            mock_model = MagicMock()
            mock_model.name = "test_model"
            mock_get_model.return_value = mock_model

            # mock engine
            mock_engine = MagicMock()
            mock_engine_cls.return_value = mock_engine

            # mock 对弈结果：1局黑胜
            mock_manager = MagicMock()
            mock_manager.start_ai_vs_ai.return_value = [
                {"game": 1, "winner": 1, "temp_side_this_game": None}
            ]
            mock_manager_cls.return_value = mock_manager

            # mock recorder 及轮次目录
            mock_recorder = MagicMock()
            round_dir = os.path.join(tmpdir, "round_1")
            os.makedirs(round_dir, exist_ok=True)
            # 创建模拟 record.json（变异步数据：确保 move_filter 不过滤）
            game_dir = os.path.join(round_dir, "1")
            os.makedirs(game_dir, exist_ok=True)
            variant_output = [0.0] * 225
            variant_output[0] = 0.1  # 落子位 [0,0]
            variant_output[1] = 0.9  # max 在别处 → variant
            record = {
                "winner": 1,
                "history": [
                    {"input": [0.0] * 226, "output": variant_output, "position": [0, 0]},
                ],
                "end_board": [0.0] * 226
            }
            with open(os.path.join(game_dir, "record.json"), 'w', encoding='utf-8') as f:
                json.dump(record, f)
            mock_recorder.get_round_dir.return_value = round_dir
            mock_recorder_cls.return_value = mock_recorder

            with patch('sys.argv', ['create_samples.py', '--model', 'test_model', '--games', '1', '--output', output_path]):
                main()

            self.assertTrue(os.path.exists(output_path))
            with open(output_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)

            self.assertIsInstance(samples, list)
            # 启用增强后 1 个原始样本变为 8 个
            self.assertEqual(len(samples), 8)
            self.assertIn("input", samples[0])
            self.assertIn("output", samples[0])
            self.assertIn("mask", samples[0])
            self.assertEqual(samples[0]["mask"], [1] * 225)


@patch('ai.create_samples.DataRecorder')
@patch('ai.create_samples.AIGameManager')
@patch('ai.create_samples.SinglePolicyAiEngine')
@patch('ai.create_samples.get_model')
class TestCliOutputHasCorrectSampleCount(unittest.TestCase):
    """输出样本数与对局数一致（排除平局）"""

    def test_excludes_draws(self, mock_get_model, mock_engine_cls, mock_manager_cls, mock_recorder_cls):
        """平局对局不产生样本"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")

            mock_model = MagicMock()
            mock_model.name = "test_model"
            mock_get_model.return_value = mock_model

            mock_engine = MagicMock()
            mock_engine_cls.return_value = mock_engine

            # 3局：黑胜、平局、白胜
            mock_manager = MagicMock()
            mock_manager.start_ai_vs_ai.return_value = [
                {"game": 1, "winner": 1, "temp_side_this_game": None},
                {"game": 2, "winner": 3, "temp_side_this_game": None},
                {"game": 3, "winner": 2, "temp_side_this_game": None},
            ]
            mock_manager_cls.return_value = mock_manager

            mock_recorder = MagicMock()
            round_dir = os.path.join(tmpdir, "round_1")
            os.makedirs(round_dir, exist_ok=True)

            # 为每局创建 record（变异步数据：确保 move_filter 不过滤）
            # 白胜需要 ≥2 步 history 才能有白方步
            for game_num, winner, extra_steps in [(1, 1, 0), (2, 3, 0), (3, 2, 1)]:
                game_dir = os.path.join(round_dir, str(game_num))
                os.makedirs(game_dir, exist_ok=True)
                history = []
                for s in range(1 + extra_steps):
                    pos = [game_num - 1 + s, game_num - 1 + s]
                    idx = pos[0] * 15 + pos[1]
                    variant_output = [0.0] * 225
                    variant_output[idx] = 0.1
                    variant_output[(idx + 1) % 225] = 0.9  # max 在别处
                    history.append({
                        "input": [float(game_num + s)] * 226,
                        "output": variant_output,
                        "position": pos,
                    })
                record = {
                    "winner": winner,
                    "history": history,
                    "end_board": [0.0] * 226
                }
                with open(os.path.join(game_dir, "record.json"), 'w', encoding='utf-8') as f:
                    json.dump(record, f)

            mock_recorder.get_round_dir.return_value = round_dir
            mock_recorder_cls.return_value = mock_recorder

            with patch('sys.argv', ['create_samples.py', '--model', 'test_model', '--games', '3', '--output', output_path]):
                main()

            with open(output_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)

            # 平局被跳过，2 个有效局 × 8 增强 = 16 个样本
            self.assertEqual(len(samples), 16)

    def test_all_draws_produces_empty(self, mock_get_model, mock_engine_cls, mock_manager_cls, mock_recorder_cls):
        """全部平局时输出空列表"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")

            mock_model = MagicMock()
            mock_model.name = "test_model"
            mock_get_model.return_value = mock_model

            mock_engine = MagicMock()
            mock_engine_cls.return_value = mock_engine

            mock_manager = MagicMock()
            mock_manager.start_ai_vs_ai.return_value = [
                {"game": 1, "winner": 3, "temp_side_this_game": None},
            ]
            mock_manager_cls.return_value = mock_manager

            mock_recorder = MagicMock()
            round_dir = os.path.join(tmpdir, "round_1")
            os.makedirs(round_dir, exist_ok=True)
            game_dir = os.path.join(round_dir, "1")
            os.makedirs(game_dir, exist_ok=True)
            record = {
                "winner": 3,
                "history": [
                    {"input": [0.0] * 226, "output": [0.0] * 225, "position": [0, 0]},
                ],
                "end_board": [0.0] * 226
            }
            with open(os.path.join(game_dir, "record.json"), 'w', encoding='utf-8') as f:
                json.dump(record, f)
            mock_recorder.get_round_dir.return_value = round_dir
            mock_recorder_cls.return_value = mock_recorder

            with patch('sys.argv', ['create_samples.py', '--model', 'test_model', '--games', '1', '--output', output_path]):
                main()

            with open(output_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)

            self.assertEqual(samples, [])


@patch('ai.create_samples.DataRecorder')
@patch('ai.create_samples.AIGameManager')
@patch('ai.create_samples.SinglePolicyAiEngine')
@patch('ai.create_samples.get_model')
class TestCliNoAugment(unittest.TestCase):
    """--no-augment 参数测试"""

    def test_no_augment_keeps_sample_count(self, mock_get_model, mock_engine_cls, mock_manager_cls, mock_recorder_cls):
        """使用 --no-augment 时样本数不翻倍"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")

            mock_model = MagicMock()
            mock_model.name = "test_model"
            mock_get_model.return_value = mock_model

            mock_engine = MagicMock()
            mock_engine_cls.return_value = mock_engine

            mock_manager = MagicMock()
            mock_manager.start_ai_vs_ai.return_value = [
                {"game": 1, "winner": 1, "temp_side_this_game": None},
            ]
            mock_manager_cls.return_value = mock_manager

            mock_recorder = MagicMock()
            round_dir = os.path.join(tmpdir, "round_1")
            os.makedirs(round_dir, exist_ok=True)
            game_dir = os.path.join(round_dir, "1")
            os.makedirs(game_dir, exist_ok=True)
            # 变异步数据：确保 move_filter 不过滤
            variant_output = [0.0] * 225
            variant_output[0] = 0.1
            variant_output[1] = 0.9
            record = {
                "winner": 1,
                "history": [
                    {"input": [0.0] * 226, "output": variant_output, "position": [0, 0]},
                ],
                "end_board": [0.0] * 226
            }
            with open(os.path.join(game_dir, "record.json"), 'w', encoding='utf-8') as f:
                json.dump(record, f)
            mock_recorder.get_round_dir.return_value = round_dir
            mock_recorder_cls.return_value = mock_recorder

            with patch('sys.argv', ['create_samples.py', '--model', 'test_model', '--games', '1', '--no-augment', '--output', output_path]):
                main()

            with open(output_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)

            # 禁用增强，1 个原始样本保持 1 个
            self.assertEqual(len(samples), 1)


# ---------------------------------------------------------------------------
# 温度采样 CLI 参数测试（对应 move_sampler_design.md 用例 29-32）
# 注意：--no-temp-sampling 已移除，温度采样始终启用
# ---------------------------------------------------------------------------


def _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                        mock_manager_cls, mock_recorder_cls):
    """为温度采样 CLI 测试批量装配公共 mocks。

    关键点：因 create_samples 使用 AIGameManager.start_ai_vs_ai(...) 这种
    静态调用，需要直接在 mock class 本身上配置 start_ai_vs_ai 的返回值，
    这样才能正确捕获 call_args。
    """
    mock_model = MagicMock()
    mock_model.name = "test_model"
    mock_model.full_name = "test_model"
    mock_get_model.return_value = mock_model

    mock_engine_cls.return_value = MagicMock()

    # 关键：静态调用走到 mock_manager_cls.start_ai_vs_ai，不是 instance
    mock_manager_cls.start_ai_vs_ai.return_value = [{"game": 1, "winner": 1, "temp_side_this_game": None}]

    # 准备 recorder 并填一个空 round_dir，供 SampleCreator 扫描
    mock_recorder = MagicMock()
    round_dir = os.path.join(tmpdir, "round_1")
    os.makedirs(round_dir, exist_ok=True)
    game_dir = os.path.join(round_dir, "1")
    os.makedirs(game_dir, exist_ok=True)
    # 使用变异步数据：落子位 [0,0]=0.1，max=0.9 在别处 → variant
    variant_output = [0.0] * 225
    variant_output[0] = 0.1
    variant_output[1] = 0.9
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


@patch('ai.create_samples.DataRecorder')
@patch('ai.create_samples.AIGameManager')
@patch('ai.create_samples.SinglePolicyAiEngine')
@patch('ai.create_samples.get_model')
class TestCliTempSamplingFlags(unittest.TestCase):
    """create_samples.py 的温度采样 CLI 参数解析 + 透传行为"""

    def test_cli_default_enables_temp_sampling(
            self, mock_get_model, mock_engine_cls,
            mock_manager_cls, mock_recorder_cls):
        """用例 29：不传 --no-temp-sampling → 透传给 start_ai_vs_ai
        的 temp_config 非 None，且字段为默认值（T=1.0，moves=12，side=random）。"""
        from ai.move_sampler import TempConfig, TEMP_SIDE_RANDOM

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)

            argv = ['create_samples.py', '--model', 'test_model',
                    '--games', '1', '--output', output_path]
            with patch('sys.argv', argv):
                main()

            call_kwargs = mock_manager_cls.start_ai_vs_ai.call_args.kwargs
            temp_config = call_kwargs.get("temp_config")
            self.assertIsNotNone(
                temp_config,
                "默认应启用温度采样，temp_config 不应为 None"
            )
            self.assertIsInstance(temp_config, TempConfig)
            self.assertEqual(temp_config.temperature, 1.0)
            self.assertEqual(temp_config.temp_moves, 12)
            # 意图 9：--temp-side 默认值改为 random
            self.assertEqual(temp_config.temp_side, TEMP_SIDE_RANDOM)

    def test_cli_no_temp_sampling_rejected(
            self, mock_get_model, mock_engine_cls,
            mock_manager_cls, mock_recorder_cls):
        """意图 9：--no-temp-sampling 已移除，传入应触发 SystemExit。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)

            argv = ['create_samples.py', '--model', 'test_model',
                    '--games', '1', '--output', output_path,
                    '--no-temp-sampling']
            with patch('sys.argv', argv):
                with self.assertRaises(SystemExit):
                    main()

    def test_cli_temp_side_parse(
            self, mock_get_model, mock_engine_cls,
            mock_manager_cls, mock_recorder_cls):
        """用例 31：--temp-side random → TempConfig.temp_side == 'random'。"""
        from ai.move_sampler import TEMP_SIDE_RANDOM

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)

            argv = ['create_samples.py', '--model', 'test_model',
                    '--games', '1', '--output', output_path,
                    '--temp-side', 'random',
                    '--temperature', '0.5',
                    '--temp-moves', '8',
                    '--temp-seed', '2025']
            with patch('sys.argv', argv):
                main()

            call_kwargs = mock_manager_cls.start_ai_vs_ai.call_args.kwargs
            temp_config = call_kwargs.get("temp_config")
            self.assertIsNotNone(temp_config)
            self.assertEqual(temp_config.temp_side, TEMP_SIDE_RANDOM)
            self.assertEqual(temp_config.temperature, 0.5)
            self.assertEqual(temp_config.temp_moves, 8)
            self.assertEqual(temp_config.seed, 2025)

    def test_cli_invalid_temperature_fails(
            self, mock_get_model, mock_engine_cls,
            mock_manager_cls, mock_recorder_cls):
        """用例 32：--temperature 0 → 进程以非 0 退出码结束。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)

            argv = ['create_samples.py', '--model', 'test_model',
                    '--games', '1', '--output', output_path,
                    '--temperature', '0']
            with patch('sys.argv', argv):
                with self.assertRaises((SystemExit, ValueError)):
                    main()


class TestCliModeRemoved(unittest.TestCase):
    """CLI --mode 参数已移除测试（CLI-01, CLI-02）"""

    def test_no_mode_arg_runs_normally(self):
        """CLI-01: 不传 --mode → argparse 正常解析，不报错"""
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, required=True)
        parser.add_argument('--output', type=str, required=True)
        parser.add_argument('--games', type=int, default=100)
        # --mode 不在 parser 中，传了会报 unrecognized
        args = parser.parse_args(['--model', 'test', '--output', 'out.json'])
        self.assertEqual(args.model, 'test')

    def test_mode_arg_causes_system_exit(self):
        """CLI-02: 传 --mode winner_path → SystemExit（unrecognized arguments）"""
        with patch('sys.argv', ['create_samples.py', '--model', 'test',
                                '--output', 'out.json',
                                '--mode', 'winner_path']):
            with self.assertRaises(SystemExit):
                main()


class TestCliTempSideDefault(unittest.TestCase):
    """CLI --temp-side 参数测试（CLI-04, CLI-05）"""

    def test_temp_side_default_random(self):
        """CLI-04: 不传 --temp-side → 默认值 random"""
        from ai.move_sampler import TEMP_SIDE_RANDOM
        parser = argparse.ArgumentParser()
        parser.add_argument('--temp-side', type=str, default=TEMP_SIDE_RANDOM,
                            choices=['black', 'white', 'both', 'random'])
        args = parser.parse_args([])
        self.assertEqual(args.temp_side, TEMP_SIDE_RANDOM)

    def test_temp_side_black(self):
        """CLI-05: 传 --temp-side black → 使用 black"""
        parser = argparse.ArgumentParser()
        parser.add_argument('--temp-side', type=str, default='random',
                            choices=['black', 'white', 'both', 'random'])
        args = parser.parse_args(['--temp-side', 'black'])
        self.assertEqual(args.temp_side, 'black')


@patch('ai.create_samples.DataRecorder')
@patch('ai.create_samples.AIGameManager')
@patch('ai.create_samples.SinglePolicyAiEngine')
@patch('ai.create_samples.get_model')
class TestCliStatsOutput(unittest.TestCase):
    """统计输出测试（STAT-01 ~ STAT-04）"""

    def test_stats_output_with_filter(self, mock_get_model, mock_engine_cls,
                                       mock_manager_cls, mock_recorder_cls):
        """STAT-01: 单方采样时有步级统计输出"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)

            captured = StringIO()
            argv = ['create_samples.py', '--model', 'test_model',
                    '--games', '1', '--output', output_path]
            with patch('sys.argv', argv), patch('sys.stdout', captured):
                main()

            stdout = captured.getvalue()
            # STAT-01: 步级过滤统计输出
            self.assertIn("步级过滤: 启用 (变异步过滤)", stdout)
            self.assertIn("总步数", stdout)
            self.assertIn("变异步", stdout)
            self.assertIn("跳过", stdout)

    def test_augment_enabled_output(self, mock_get_model, mock_engine_cls,
                                     mock_manager_cls, mock_recorder_cls):
        """STAT-03: augment=True → 输出含"启用 (D4 群 8 变换)" """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)

            captured = StringIO()
            argv = ['create_samples.py', '--model', 'test_model',
                    '--games', '1', '--output', output_path]
            with patch('sys.argv', argv), patch('sys.stdout', captured):
                main()

            stdout = captured.getvalue()
            self.assertIn("启用 (D4 群 8 变换)", stdout)

    def test_augment_disabled_output(self, mock_get_model, mock_engine_cls,
                                      mock_manager_cls, mock_recorder_cls):
        """STAT-04: augment=False → 输出含"禁用" """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            _setup_common_mocks(tmpdir, mock_get_model, mock_engine_cls,
                                mock_manager_cls, mock_recorder_cls)

            captured = StringIO()
            argv = ['create_samples.py', '--model', 'test_model',
                    '--games', '1', '--output', output_path, '--no-augment']
            with patch('sys.argv', argv), patch('sys.stdout', captured):
                main()

            stdout = captured.getvalue()
            self.assertIn("禁用", stdout)


if __name__ == '__main__':
    unittest.main()
