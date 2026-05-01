"""create_samples.py 集成测试

覆盖 CLI 脚本的完整流程：
- 输出文件存在且格式正确
- 输出样本数与对局数一致（排除平局）

注意：使用 mock 替代真实模型对弈，避免依赖 GPU/模型权重。
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.create_samples import main


class TestCliCreatesOutputFile(unittest.TestCase):
    """执行后输出文件存在且格式正确"""

    @patch('ai.create_samples.get_model')
    @patch('ai.create_samples.AIEngine')
    @patch('ai.create_samples.AIGameManager')
    @patch('ai.create_samples.DataRecorder')
    def test_output_file_exists_and_valid(self, mock_recorder_cls, mock_manager_cls, mock_engine_cls, mock_get_model):
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
                {"game": 1, "winner": 1}
            ]
            mock_manager_cls.return_value = mock_manager

            # mock recorder 及轮次目录
            mock_recorder = MagicMock()
            round_dir = os.path.join(tmpdir, "round_1")
            os.makedirs(round_dir, exist_ok=True)
            # 创建模拟 record.json
            game_dir = os.path.join(round_dir, "1")
            os.makedirs(game_dir, exist_ok=True)
            record = {
                "winner": 1,
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


class TestCliOutputHasCorrectSampleCount(unittest.TestCase):
    """输出样本数与对局数一致（排除平局）"""

    @patch('ai.create_samples.get_model')
    @patch('ai.create_samples.AIEngine')
    @patch('ai.create_samples.AIGameManager')
    @patch('ai.create_samples.DataRecorder')
    def test_excludes_draws(self, mock_recorder_cls, mock_manager_cls, mock_engine_cls, mock_get_model):
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
                {"game": 1, "winner": 1},
                {"game": 2, "winner": 3},
                {"game": 3, "winner": 2},
            ]
            mock_manager_cls.return_value = mock_manager

            mock_recorder = MagicMock()
            round_dir = os.path.join(tmpdir, "round_1")
            os.makedirs(round_dir, exist_ok=True)

            # 为每局创建 record
            for game_num, winner in [(1, 1), (2, 3), (3, 2)]:
                game_dir = os.path.join(round_dir, str(game_num))
                os.makedirs(game_dir, exist_ok=True)
                record = {
                    "winner": winner,
                    "history": [
                        {"input": [float(game_num)] * 226, "output": [0.0] * 225, "position": [game_num - 1, game_num - 1]},
                    ],
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

    @patch('ai.create_samples.get_model')
    @patch('ai.create_samples.AIEngine')
    @patch('ai.create_samples.AIGameManager')
    @patch('ai.create_samples.DataRecorder')
    def test_all_draws_produces_empty(self, mock_recorder_cls, mock_manager_cls, mock_engine_cls, mock_get_model):
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
                {"game": 1, "winner": 3},
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


class TestCliNoAugment(unittest.TestCase):
    """--no-augment 参数测试"""

    @patch('ai.create_samples.get_model')
    @patch('ai.create_samples.AIEngine')
    @patch('ai.create_samples.AIGameManager')
    @patch('ai.create_samples.DataRecorder')
    def test_no_augment_keeps_sample_count(self, mock_recorder_cls, mock_manager_cls, mock_engine_cls, mock_get_model):
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
                {"game": 1, "winner": 1},
            ]
            mock_manager_cls.return_value = mock_manager

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


class TestCliWinnerPathMode(unittest.TestCase):
    """--mode winner_path 端到端贯通（第三十八次追加）"""

    @patch('ai.create_samples.get_model')
    @patch('ai.create_samples.AIEngine')
    @patch('ai.create_samples.AIGameManager')
    @patch('ai.create_samples.DataRecorder')
    def test_winner_path_mode_cli(self, mock_recorder_cls, mock_manager_cls, mock_engine_cls, mock_get_model):
        """--mode winner_path 时产物 sample 数 = 获胜方步数-1，且不做 D4 增强"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "winner_path.json")

            mock_model = MagicMock()
            mock_model.name = "test_model"
            mock_get_model.return_value = mock_model

            mock_engine = MagicMock()
            mock_engine_cls.return_value = mock_engine

            # 1 局黑胜
            mock_manager = MagicMock()
            mock_manager.start_ai_vs_ai.return_value = [
                {"game": 1, "winner": 1},
            ]
            mock_manager_cls.return_value = mock_manager

            mock_recorder = MagicMock()
            round_dir = os.path.join(tmpdir, "round_1")
            os.makedirs(round_dir, exist_ok=True)
            game_dir = os.path.join(round_dir, "1")
            os.makedirs(game_dir, exist_ok=True)
            # 构造 5 步黑胜局：黑方步 = [0, 2, 4]，剔首 → [2, 4] → 2 个 sample
            history = []
            for i in range(5):
                history.append({
                    "input": [float(i)] * 226,
                    "output": [0.0] * 225,
                    "position": [i // 15, i % 15],
                })
            record = {
                "winner": 1,
                "history": history,
                "end_board": [0.0] * 226,
            }
            with open(os.path.join(game_dir, "record.json"), 'w', encoding='utf-8') as f:
                json.dump(record, f)
            mock_recorder.get_round_dir.return_value = round_dir
            mock_recorder_cls.return_value = mock_recorder

            with patch('sys.argv', [
                'create_samples.py', '--model', 'test_model',
                '--mode', 'winner_path',
                '--games', '1', '--output', output_path,
            ]):
                main()

            with open(output_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)

            # 剔首后获胜方保留 2 步，winner_path 强制不增强 → 2 个样本
            self.assertEqual(len(samples), 2)
            # 每个 sample 结构齐全、mask 全 1
            for s in samples:
                self.assertIn("input", s)
                self.assertIn("output", s)
                self.assertIn("mask", s)
                self.assertEqual(s["mask"], [1] * 225)
                # input 应是保留步的原始 input（值为 2.0 或 4.0）
                self.assertIn(s["input"][0], (2.0, 4.0))


if __name__ == '__main__':
    unittest.main()
