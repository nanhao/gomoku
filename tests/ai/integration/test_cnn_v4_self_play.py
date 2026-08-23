"""cnn_v4_self_play 脚本集成测试（intention 54）

验证全链路：
- run_self_play 端到端：对弈 → 原子落盘 sample.json + summary.json
- 真实 cnn_v4.3 权重加载与 predict 输出（只读，不修改模型文件）

输出目录使用临时目录，测试结束自动清理，不修改真实数据文件。
"""

import json
import os
import sys
import shutil
import tempfile
import unittest
from unittest import mock

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
_src_dir = os.path.join(project_root, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from ai.base_model import ModelPredict  # noqa: E402
from tests.ai.unit.test_cnn_v4_self_play import SequentialFakeModel  # noqa: E402


class TestRunSelfPlayEndToEnd(unittest.TestCase):
    """run_self_play 端到端集成测试（注入 Fake 模型）"""

    def setUp(self):
        """创建临时输出目录"""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "boots", "1")

    def tearDown(self):
        """清理临时目录"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _make_args(self, games=2, **kwargs):
        """构造 argparse Namespace 参数"""
        from argparse import Namespace
        defaults = dict(
            model="cnn_v4.3",
            games=games,
            temperature=1.0,
            temp_steps=12,
            output=self.output_dir,
            device="cpu",
        )
        defaults.update(kwargs)
        return Namespace(**defaults)

    def test_run_self_play_produces_sample_and_summary(self):
        """跑 2 局后生成 sample.json 与 summary.json，内容与格式正确"""
        from ai.scripts import cnn_v4_self_play

        black = [0, 1, 2, 3, 4]
        white = [14 * 15 + 0, 14 * 15 + 2, 14 * 15 + 4, 14 * 15 + 6]
        fake_model = SequentialFakeModel(black, white)

        with mock.patch.object(cnn_v4_self_play, "get_model", return_value=fake_model):
            args = self._make_args(games=2)
            cnn_v4_self_play.run_self_play(args)

        # sample.json 存在且为样本数组
        sample_path = os.path.join(self.output_dir, "sample.json")
        self.assertTrue(os.path.exists(sample_path))
        with open(sample_path, encoding="utf-8") as f:
            samples = json.load(f)
        self.assertIsInstance(samples, list)
        # 每局黑方第 5 步连五：9 手/局 × 2 局
        self.assertEqual(len(samples), 18)
        for s in samples:
            self.assertEqual(len(s["state"]), 226)
            self.assertEqual(len(s["pi"]), 225)
            self.assertIn(s["z"], (1, -1, 0))

        # summary.json 记录元信息
        summary_path = os.path.join(self.output_dir, "summary.json")
        self.assertTrue(os.path.exists(summary_path))
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        self.assertEqual(summary["model"], "cnn_v4.3")
        self.assertEqual(summary["games"], 2)
        self.assertEqual(summary["total_samples"], 18)
        self.assertEqual(summary["completed_games"], 2)
        self.assertEqual(summary["hyperparams"]["temperature"], 1.0)
        self.assertEqual(summary["hyperparams"]["temp_steps"], 12)

    def test_run_self_play_resumes_existing_samples(self):
        """已有 sample.json 时从现有样本继续追加"""
        from ai.scripts import cnn_v4_self_play

        black = [0, 1, 2, 3, 4]
        white = [14 * 15 + 0, 14 * 15 + 2, 14 * 15 + 4, 14 * 15 + 6]
        fake_model = SequentialFakeModel(black, white)

        # 预置 1 局旧样本
        os.makedirs(self.output_dir, exist_ok=True)
        old_samples = [{"state": [0] * 226, "pi": [0.0] * 225, "z": 0}]
        with open(os.path.join(self.output_dir, "sample.json"), "w", encoding="utf-8") as f:
            json.dump(old_samples, f)

        with mock.patch.object(cnn_v4_self_play, "get_model", return_value=fake_model):
            args = self._make_args(games=2)
            cnn_v4_self_play.run_self_play(args)

        with open(os.path.join(self.output_dir, "sample.json"), encoding="utf-8") as f:
            samples = json.load(f)
        # 旧样本 1 + 新对局 18 = 19
        self.assertEqual(len(samples), 19)


class TestRealModelLoading(unittest.TestCase):
    """真实 cnn_v4.3 权重加载与推理（只读）"""

    def test_get_model_and_predict_cnn_v4_3(self):
        """get_model('cnn_v4.3') 加载真实权重并输出 225 维概率"""
        from ai.registry import get_model
        from ai.models.cnn_v4.model import CNNV4Model

        model = get_model("cnn_v4.3")
        self.assertIsInstance(model, CNNV4Model)
        self.assertEqual(model.full_name, "cnn_v4.3")

        # 空棋盘预测：黑方先手
        board_vector = [0] * 225 + [1]
        result = model.predict(board_vector)
        self.assertEqual(len(result.policy), 225)
        total = sum(result.policy)
        self.assertAlmostEqual(total, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
