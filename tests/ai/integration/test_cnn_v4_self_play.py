"""cnn_v4_self_play 脚本集成测试（intention 54/55）

验证全链路：
- run_self_play 端到端：对弈 → 按 chunk 直接写入样本缓冲区（chunk + manifest）
- 攒满 chunk_games 局落盘、不足 chunk 正常收尾落盘
- 从 manifest 续接局号、Ctrl+C 中断丢弃未满 chunk
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
        """创建临时缓冲区目录"""
        self.temp_dir = tempfile.mkdtemp()
        self.buffer_dir = os.path.join(self.temp_dir, "sample_buffer")
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

    def _make_args(self, games=2, **kwargs):
        """构造 argparse Namespace 参数"""
        from argparse import Namespace
        defaults = dict(
            model="cnn_v4.3",
            games=games,
            temperature=1.0,
            temp_steps=12,
            output_dir=self.buffer_dir,
            buffer_max_size=80000,
            chunk_games=10,
            device="cpu",
        )
        defaults.update(kwargs)
        return Namespace(**defaults)

    def _make_fake_model(self):
        """黑方 5 步连五的 Fake 模型（每局 9 手）"""
        black = [0, 1, 2, 3, 4]
        white = [14 * 15 + 0, 14 * 15 + 2, 14 * 15 + 4, 14 * 15 + 6]
        return SequentialFakeModel(black, white)

    def _read_manifest(self):
        """读取缓冲区 manifest.json"""
        with open(os.path.join(self.buffer_dir, "manifest.json"), encoding="utf-8") as f:
            return json.load(f)

    def test_run_self_play_produces_chunks_and_manifest(self):
        """跑 2 局（< chunk_games）后生成 1 个 chunk + manifest，内容与格式正确"""
        from ai.scripts import cnn_v4_self_play

        fake_model = self._make_fake_model()
        with mock.patch.object(cnn_v4_self_play, "get_model", return_value=fake_model):
            cnn_v4_self_play.run_self_play(self._make_args(games=2))

        # manifest 记录 1 个 chunk：局 1-2，18 条样本
        manifest = self._read_manifest()
        self.assertEqual(manifest["max_size"], 80000)
        self.assertEqual(manifest["chunk_games"], 10)
        self.assertEqual(manifest["total_samples"], 18)
        self.assertEqual(len(manifest["chunks"]), 1)
        self.assertEqual(manifest["chunks"][0]["game_start"], 1)
        self.assertEqual(manifest["chunks"][0]["game_end"], 2)
        self.assertEqual(manifest["chunks"][0]["sample_count"], 18)
        self.assertEqual(manifest["chunks"][0]["file"], "chunk_000001_000002.json")

        # chunk 文件存在且样本格式正确（每局 9 手）
        chunk_path = os.path.join(self.buffer_dir, "chunk_000001_000002.json")
        self.assertTrue(os.path.exists(chunk_path))
        with open(chunk_path, encoding="utf-8") as f:
            samples = json.load(f)
        self.assertEqual(len(samples), 18)
        for s in samples:
            self.assertEqual(len(s["state"]), 226)
            self.assertEqual(len(s["pi"]), 225)
            self.assertIn(s["z"], (1, -1, 0))

        # 不再生成 sample.json / summary.json（缓冲区格式）
        self.assertFalse(os.path.exists(os.path.join(self.buffer_dir, "sample.json")))
        self.assertFalse(os.path.exists(os.path.join(self.buffer_dir, "summary.json")))

    def test_run_self_play_flushes_partial_chunk(self):
        """正常结束时不足 chunk_games 的剩余局收尾落盘（3 局 chunk_games=2 → 2 chunk）"""
        from ai.scripts import cnn_v4_self_play

        fake_model = self._make_fake_model()
        args = self._make_args(games=3, chunk_games=2)
        with mock.patch.object(cnn_v4_self_play, "get_model", return_value=fake_model):
            cnn_v4_self_play.run_self_play(args)

        manifest = self._read_manifest()
        self.assertEqual(manifest["total_samples"], 27)
        self.assertEqual(len(manifest["chunks"]), 2)
        self.assertEqual(manifest["chunks"][0]["game_start"], 1)
        self.assertEqual(manifest["chunks"][0]["game_end"], 2)
        self.assertEqual(manifest["chunks"][1]["game_start"], 3)
        self.assertEqual(manifest["chunks"][1]["game_end"], 3)
        self.assertEqual(manifest["chunks"][1]["file"], "chunk_000003_000003.json")

    def test_run_self_play_resumes_from_manifest(self):
        """已有 manifest 时从最大 game_end + 1 续接局号"""
        from ai.scripts import cnn_v4_self_play

        # 预置：局 1-2 的旧 chunk + manifest
        os.makedirs(self.buffer_dir, exist_ok=True)
        old_samples = [{"state": [0] * 226, "pi": [0.0] * 225, "z": 0}]
        with open(os.path.join(self.buffer_dir, "chunk_000001_000002.json"),
                  "w", encoding="utf-8") as f:
            json.dump(old_samples, f)
        with open(os.path.join(self.buffer_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({
                "max_size": 80000,
                "chunk_games": 10,
                "total_samples": 1,
                "chunks": [{
                    "game_start": 1, "game_end": 2,
                    "sample_count": 1, "file": "chunk_000001_000002.json",
                }],
            }, f)

        fake_model = self._make_fake_model()
        with mock.patch.object(cnn_v4_self_play, "get_model", return_value=fake_model):
            cnn_v4_self_play.run_self_play(self._make_args(games=2))

        # 新局从 3 开始：chunk_000003_000004.json
        manifest = self._read_manifest()
        self.assertEqual(manifest["total_samples"], 1 + 18)
        self.assertEqual(len(manifest["chunks"]), 2)
        self.assertEqual(manifest["chunks"][1]["game_start"], 3)
        self.assertEqual(manifest["chunks"][1]["game_end"], 4)
        self.assertTrue(os.path.exists(
            os.path.join(self.buffer_dir, "chunk_000003_000004.json")))

    def test_run_self_play_interrupt_drops_pending(self):
        """Ctrl+C 中断：已落盘 chunk 保留，未满 chunk 的局样本丢弃"""
        from ai.scripts import cnn_v4_self_play

        fake_model = self._make_fake_model()
        args = self._make_args(games=4, chunk_games=2)

        original_play = cnn_v4_self_play._play_one_game
        call_count = {"n": 0}

        def _interrupt_on_third(model, temperature, temp_steps, rng=None):
            call_count["n"] += 1
            if call_count["n"] == 3:
                cnn_v4_self_play._interrupted = True
            return original_play(model, temperature, temp_steps, rng)

        with mock.patch.object(cnn_v4_self_play, "get_model", return_value=fake_model), \
                mock.patch.object(cnn_v4_self_play, "_play_one_game",
                                  side_effect=_interrupt_on_third):
            cnn_v4_self_play.run_self_play(args)

        # 第 3 局样本在内存中已生成但未落盘（中断丢弃），仅局 1-2 的 chunk 存在
        manifest = self._read_manifest()
        self.assertEqual(manifest["total_samples"], 18)
        self.assertEqual(len(manifest["chunks"]), 1)
        self.assertEqual(manifest["chunks"][0]["game_end"], 2)
        self.assertTrue(os.path.exists(
            os.path.join(self.buffer_dir, "chunk_000001_000002.json")))
        self.assertFalse(os.path.exists(
            os.path.join(self.buffer_dir, "chunk_000003_000004.json")))
        # 清理全局中断标志，避免影响其他测试
        cnn_v4_self_play._interrupted = False


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
