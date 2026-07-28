"""value_self_play 脚本单元测试

验证参数解析、辅助函数、单局对弈。
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# 添加 ai/scripts 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))


class TestValueSelfPlayHelpers(unittest.TestCase):
    """辅助函数测试"""

    def test_parse_model_spec_from_path(self):
        """从路径解析模型 spec"""
        from ai.scripts.value_self_play import _parse_model_spec_from_path
        spec = _parse_model_spec_from_path(
            os.path.join("some", "path", "cnn_v10", "gen1", "cnn_v10.pth")
        )
        self.assertEqual(spec, "cnn_v10.1")

    def test_parse_model_spec_invalid_path_raises(self):
        """无效路径抛异常"""
        from ai.scripts.value_self_play import _parse_model_spec_from_path
        with self.assertRaises(ValueError):
            _parse_model_spec_from_path("no_gen_dir/model.pth")

    def test_describe_result_win(self):
        """胜局描述"""
        from ai.scripts.value_self_play import _describe_result
        samples = [{"state": [0]*225 + [1], "z": 1}]
        desc = _describe_result(samples)
        self.assertIn("胜", desc)

    def test_describe_result_empty(self):
        """空样本"""
        from ai.scripts.value_self_play import _describe_result
        desc = _describe_result([])
        self.assertEqual(desc, "无样本")

    def test_save_progress(self):
        """_save_progress 写入 sample.json + summary.json"""
        from ai.scripts.value_self_play import _save_progress
        from unittest.mock import MagicMock

        tmp_dir = tempfile.mkdtemp()
        try:
            samples = [{"state": [0]*226, "z": 1}]
            args = MagicMock()
            args.model = "test.pth"
            args.games = 10
            args.temperature = 1.0
            args.temp_steps = 8
            args.temp_side = "random"
            args.top_k = None
            args.temp_seed = None

            _save_progress(samples, tmp_dir, args, 1)

            # 验证 sample.json
            sample_path = os.path.join(tmp_dir, "sample.json")
            self.assertTrue(os.path.exists(sample_path))
            with open(sample_path, 'r') as f:
                loaded = json.load(f)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["z"], 1)

            # 验证 summary.json
            summary_path = os.path.join(tmp_dir, "summary.json")
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            self.assertEqual(summary["completed_games"], 1)
            self.assertEqual(summary["total_samples"], 1)
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_build_hyperparams(self):
        """_build_hyperparams 正确构建字典"""
        from ai.scripts.value_self_play import _build_hyperparams
        from unittest.mock import MagicMock

        args = MagicMock()
        args.temperature = 0.5
        args.temp_steps = 10
        args.temp_side = "both"
        args.top_k = 5
        args.temp_seed = 99

        hp = _build_hyperparams(args)
        self.assertEqual(hp["temperature"], 0.5)
        self.assertEqual(hp["temp_steps"], 10)
        self.assertEqual(hp["temp_side"], "both")
        self.assertEqual(hp["top_k"], 5)
        self.assertEqual(hp["seed"], 99)


class TestValueSelfPlayIntegration(unittest.TestCase):
    """value_self_play 集成测试"""

    def test_play_one_game(self):
        """play_one_game 产生正确格式样本"""
        from ai.models.cnn_v10.model import CNNV10Model
        from ai.scripts.value_self_play_worker import ValueSelfPlayConfig
        from ai.scripts.value_self_play import play_one_game

        model = CNNV10Model(fresh=True)
        cfg = ValueSelfPlayConfig()
        samples = play_one_game(model, "cpu", cfg, 0, "test")

        self.assertGreater(len(samples), 0)
        for s in samples:
            self.assertIn("state", s)
            self.assertIn("z", s)
            self.assertEqual(len(s["state"]), 226)
            self.assertNotIn("pi", s)


if __name__ == '__main__':
    unittest.main()
