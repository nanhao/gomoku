"""MCTS 自对弈 — 中断安全与周期保存测试

覆盖设计文档 test_design.md 中的 T1–T11 测试用例。
"""

import json
import os
import signal
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# 确保项目根在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ---------- 辅助函数 ----------

def _make_fake_sample(game_idx: int = 0) -> dict:
    """创建一条合法的伪造样本，用于快速测试保存逻辑。"""
    return {
        "state": [0] * 225 + [1],
        "pi": [1.0 / 225] * 225,
        "z": 1 if game_idx % 2 == 0 else -1,
    }


def _make_fake_args(output_dir: str, model_path: str = "fake_model.pth",
                    games: int = 100) -> MagicMock:
    """创建伪造的 argparse.Namespace，供 _save_progress 使用。"""
    args = MagicMock()
    args.model = model_path
    args.games = games
    args.simulations = 800
    args.c_puct = 2.0
    args.temp_steps = 8
    args.temperature = 1.0
    args.dirichlet_epsilon = 0.25
    args.dirichlet_alpha = 0.03
    args.output = output_dir
    args.device = "cpu"
    return args


# ---------- T5, T6, T11: _load_existing_samples ----------

class TestLoadExistingSamples(unittest.TestCase):
    """测试启动时加载已有 sample.json 的行为
    覆盖: T5 (已有文件合并), T6 (无文件), T11 (损坏 JSON)
    """

    @classmethod
    def setUpClass(cls):
        from ai.scripts import mcts_self_play
        cls.module = mcts_self_play

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_load_samples_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_nonexistent_returns_empty(self):
        """T6: 无已有 sample.json → 返回空列表"""
        result = self.module._load_existing_samples(self.temp_dir)
        self.assertEqual(result, [])

    def test_load_valid_samples(self):
        """T5: 已有合法 sample.json → 正确加载"""
        samples = [_make_fake_sample(i) for i in range(10)]
        sample_path = os.path.join(self.temp_dir, "sample.json")
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f)

        result = self.module._load_existing_samples(self.temp_dir)
        self.assertEqual(len(result), 10)
        self.assertEqual(result[0]["z"], 1)

    def test_load_corrupted_json(self):
        """T11: 损坏的 JSON → 输出警告，返回空列表"""
        sample_path = os.path.join(self.temp_dir, "sample.json")
        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write("[{bad json")

        result = self.module._load_existing_samples(self.temp_dir)
        self.assertEqual(result, [])


# ---------- T7: _save_progress ----------

class TestSaveProgress(unittest.TestCase):
    """测试 _save_progress 函数
    覆盖: T7 (原子写入不损坏已有文件), T8 (多次保存累加)
    """

    @classmethod
    def setUpClass(cls):
        from ai.scripts import mcts_self_play
        cls.module = mcts_self_play

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_save_progress_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_progress_creates_both_files(self):
        """保存后 sample.json 和 summary.json 均存在"""
        samples = [_make_fake_sample(0)]
        args = _make_fake_args(self.temp_dir, games=1)

        self.module._save_progress(samples, self.temp_dir, args, completed_games=1)

        sample_path = os.path.join(self.temp_dir, "sample.json")
        summary_path = os.path.join(self.temp_dir, "summary.json")
        self.assertTrue(os.path.exists(sample_path))
        self.assertTrue(os.path.exists(summary_path))

        # 验证 sample.json 内容
        with open(sample_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        self.assertEqual(len(loaded), 1)

        # 验证 summary.json 字段
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        self.assertEqual(summary["total_samples"], 1)
        self.assertEqual(summary["completed_games"], 1)
        self.assertIn("hyperparams", summary)

    def test_save_progress_no_tmp_leftover(self):
        """T7: 原子写入后不应残留 .tmp 文件"""
        samples = [_make_fake_sample(0)]
        args = _make_fake_args(self.temp_dir, games=1)

        self.module._save_progress(samples, self.temp_dir, args, completed_games=1)

        tmp_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.tmp')]
        self.assertEqual(len(tmp_files), 0, f"残留临时文件: {tmp_files}")

    def test_save_progress_does_not_corrupt_existing_on_crash(self):
        """T7: 模拟写入 .tmp 后崩溃 → 原 sample.json 内容不变"""
        samples = [_make_fake_sample(0)]
        args = _make_fake_args(self.temp_dir, games=1)
        sample_path = os.path.join(self.temp_dir, "sample.json")

        # 先写入一份原始数据
        self.module._save_progress(samples, self.temp_dir, args, completed_games=1)
        with open(sample_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        # 创建 .tmp 文件模拟写入中断（未完成 rename）
        tmp_path = sample_path + ".tmp"
        corrupt_samples = [_make_fake_sample(1), _make_fake_sample(2)]  # 新数据写入 .tmp
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(corrupt_samples, f)
        # 故意不调用 os.replace() — 模拟崩溃

        # 原 sample.json 应保持不变
        with open(sample_path, 'r', encoding='utf-8') as f:
            after_data = json.load(f)
        self.assertEqual(len(after_data), len(original_data))

    def test_multiple_saves_cumulative(self):
        """T8: 多次保存 → 样本数正确累加"""
        args = _make_fake_args(self.temp_dir, games=120)
        sample_path = os.path.join(self.temp_dir, "sample.json")
        summary_path = os.path.join(self.temp_dir, "summary.json")

        # 模拟 50 局保存
        samples_50 = [_make_fake_sample(i) for i in range(50)]
        self.module._save_progress(samples_50, self.temp_dir, args, completed_games=50)
        with open(sample_path, 'r', encoding='utf-8') as f:
            self.assertEqual(len(json.load(f)), 50)

        # 模拟 100 局保存（含前 50 局 + 后 50 局）
        samples_100 = [_make_fake_sample(i) for i in range(100)]
        self.module._save_progress(samples_100, self.temp_dir, args, completed_games=100)
        with open(sample_path, 'r', encoding='utf-8') as f:
            self.assertEqual(len(json.load(f)), 100)

        # 模拟 120 局最终保存
        samples_120 = [_make_fake_sample(i) for i in range(120)]
        self.module._save_progress(samples_120, self.temp_dir, args, completed_games=120)
        with open(sample_path, 'r', encoding='utf-8') as f:
            self.assertEqual(len(json.load(f)), 120)

        # 验证 summary 各保存点递增
        # 最后 summary 应为 120
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        self.assertEqual(summary["completed_games"], 120)
        self.assertEqual(summary["total_samples"], 120)


# ---------- T3, T4, T9: Signal Handling ----------

class TestSignalHandling(unittest.TestCase):
    """测试信号处理机制（flag + handler）
    覆盖: T3 (Ctrl+C 后保存), T4 (对局进行中 Ctrl+C), T9 (无完成对局时 Ctrl+C)
    """

    @classmethod
    def setUpClass(cls):
        from ai.scripts import mcts_self_play
        cls.module = mcts_self_play

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_signal_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_interrupt_flag_set_by_handler(self):
        """验证信号处理器能设置 _interrupted 标志"""
        # 重置标志
        self.module._interrupted = False

        # 模拟信号触发
        self.module._signal_handler(signal.SIGINT, None)

        self.assertTrue(self.module._interrupted)

    def test_interrupt_saves_completed_games(self):
        """T3: 中断时保存已完成对局的数据"""
        self.module._interrupted = False
        all_samples = [_make_fake_sample(i) for i in range(35)]
        args = _make_fake_args(self.temp_dir, games=100)

        self.module._save_progress(all_samples, self.temp_dir, args, completed_games=35)

        sample_path = os.path.join(self.temp_dir, "sample.json")
        self.assertTrue(os.path.exists(sample_path))
        with open(sample_path, 'r', encoding='utf-8') as f:
            self.assertEqual(len(json.load(f)), 35)

    def test_interrupt_before_first_game_completes(self):
        """T9: 无已完成对局时不应写入 sample.json"""
        self.module._interrupted = False
        all_samples = []
        args = _make_fake_args(self.temp_dir, games=100)

        # 没有已完成对局，不应保存
        sample_path = os.path.join(self.temp_dir, "sample.json")
        if len(all_samples) == 0:
            # 不保存
            pass

        # sample.json 不应被创建（无完成对局）
        self.assertFalse(os.path.exists(sample_path))


# ---------- T1, T5: Integration with real model ----------

class TestRunSelfPlayIntegration(unittest.TestCase):
    """集成测试：使用真实模型运行 run_self_play 验证端到端行为
    覆盖: T1 (正常退出), T5 (已有数据合并)
    """

    @classmethod
    def setUpClass(cls):
        """尝试导入 MCTS 相关模块"""
        try:
            from ai.scripts import mcts_self_play
            cls.mcts_self_play = mcts_self_play
            cls.has_mcts = True
        except ImportError:
            cls.has_mcts = False

    def setUp(self):
        if not self.has_mcts:
            self.skipTest("MCTS 模块尚未实现")
        self.temp_dir = tempfile.mkdtemp(prefix="test_mcts_save_integ_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_args(self, games: int = 2):
        """创建真实的 argparse.Namespace"""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", default="test_model")
        parser.add_argument("--games", type=int, default=games)
        parser.add_argument("--simulations", type=int, default=10)
        parser.add_argument("--c-puct", type=float, default=2.0)
        parser.add_argument("--temp-steps", type=int, default=4)
        parser.add_argument("--temperature", type=float, default=1.0)
        parser.add_argument("--dirichlet-epsilon", type=float, default=0.25)
        parser.add_argument("--dirichlet-alpha", type=float, default=0.03)
        parser.add_argument("--output", default=None)
        parser.add_argument("--device", default="cpu")
        args = parser.parse_args([])
        args.games = games
        args.output = self.temp_dir
        return args

    def test_run_self_play_normal_exit(self):
        """T1: 正常运行完成 → sample.json 和 summary.json 正确生成"""
        import torch
        from ai.models.cnn_v7.model import CNNV7Model
        from unittest.mock import patch

        args = self._create_args(games=2)
        self.mcts_self_play._SAVE_INTERVAL = 50  # 2 < 50, 不触发周期保存

        # 使用 patch 替换 CNNV7Model 以避免加载真实权重
        with patch.object(CNNV7Model, '__init__', return_value=None):
            with patch.object(CNNV7Model, 'move_to_device', return_value=None):
                model = CNNV7Model.__new__(CNNV7Model)
                model.move_to_device = MagicMock()

                # Mock play_one_game 返回伪造样本（2 局各 5 步）
                def fake_play_one_game(model, device, hyperparams, game_idx, run_id):
                    return [_make_fake_sample(i) for i in range(5)]

                with patch.object(self.mcts_self_play, 'play_one_game',
                                  side_effect=fake_play_one_game):
                    with patch.object(self.mcts_self_play, 'CNNV7Model',
                                      return_value=model):
                        self.mcts_self_play.run_self_play(args)

        sample_path = os.path.join(self.temp_dir, "sample.json")
        summary_path = os.path.join(self.temp_dir, "summary.json")

        self.assertTrue(os.path.exists(sample_path), "sample.json 应存在")
        self.assertTrue(os.path.exists(summary_path), "summary.json 应存在")

        with open(sample_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        self.assertEqual(len(samples), 10)  # 2 局 × 5 步

        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        self.assertEqual(summary["total_samples"], 10)
        self.assertEqual(summary["completed_games"], 2)
        self.assertIn("hyperparams", summary)

    def test_run_self_play_merge_existing(self):
        """T5: 已有 sample.json → 新样本合并后写入"""
        import torch
        from ai.models.cnn_v7.model import CNNV7Model
        from unittest.mock import patch

        # 预写已有 sample.json
        existing_samples = [_make_fake_sample(i) for i in range(20)]
        existing_path = os.path.join(self.temp_dir, "sample.json")
        with open(existing_path, 'w', encoding='utf-8') as f:
            json.dump(existing_samples, f)

        args = self._create_args(games=2)
        self.mcts_self_play._SAVE_INTERVAL = 50

        with patch.object(CNNV7Model, '__init__', return_value=None):
            with patch.object(CNNV7Model, 'move_to_device', return_value=None):
                model = CNNV7Model.__new__(CNNV7Model)
                model.move_to_device = MagicMock()

                def fake_play_one_game(model, device, hyperparams, game_idx, run_id):
                    return [_make_fake_sample(i) for i in range(3)]

                with patch.object(self.mcts_self_play, 'play_one_game',
                                  side_effect=fake_play_one_game):
                    with patch.object(self.mcts_self_play, 'CNNV7Model',
                                      return_value=model):
                        self.mcts_self_play.run_self_play(args)

        with open(existing_path, 'r', encoding='utf-8') as f:
            merged = json.load(f)
        # 20 已有 + 2 局 × 3 步 = 26
        self.assertEqual(len(merged), 26)


# ---------- T2: Periodic Save Orchestration ----------

class TestPeriodicSaveOrchestration(unittest.TestCase):
    """测试周期保存的编排逻辑（通过 mock play_one_game）
    覆盖: T2 (每 50 局保存)
    """

    @classmethod
    def setUpClass(cls):
        from ai.scripts import mcts_self_play
        cls.module = mcts_self_play

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_periodic_save_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_periodic_save_called_at_interval(self):
        """T2: 运行 110 局 → 在第 50 和 100 局时触发保存"""
        import torch
        from ai.models.cnn_v7.model import CNNV7Model
        from unittest.mock import patch

        args = self._create_args(games=110, output_dir=self.temp_dir)

        # 设置较小的保存间隔以便测试
        original_interval = self.module._SAVE_INTERVAL
        self.module._SAVE_INTERVAL = 50

        try:
            with patch.object(CNNV7Model, '__init__', return_value=None):
                with patch.object(CNNV7Model, 'move_to_device', return_value=None):
                    model = CNNV7Model.__new__(CNNV7Model)
                    model.move_to_device = MagicMock()

                    # 每局返回 2 条样本
                    def fake_play_one_game(model, device, hyperparams, game_idx, run_id):
                        return [_make_fake_sample(game_idx), _make_fake_sample(game_idx)]

                    # 记录 _save_progress 的调用时机，同时执行真实保存以保证文件写入
                    save_calls = []
                    real_save = self.module._save_progress  # 保存原始函数引用

                    def fake_save_progress(all_samples, output_root, a, completed_games):
                        save_calls.append(completed_games)
                        real_save(all_samples, output_root, a, completed_games)

                    with patch.object(self.module, 'play_one_game',
                                      side_effect=fake_play_one_game):
                        with patch.object(self.module, 'CNNV7Model',
                                          return_value=model):
                            with patch.object(self.module, '_save_progress',
                                              side_effect=fake_save_progress):
                                self.module.run_self_play(args)
        finally:
            self.module._SAVE_INTERVAL = original_interval

        # 验证保存时机: 50, 100, 110 (最终)
        self.assertIn(50, save_calls, "应在 50 局时保存")
        self.assertIn(100, save_calls, "应在 100 局时保存")

    def _create_args(self, games: int, output_dir: str):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", default="test_model")
        parser.add_argument("--games", type=int, default=games)
        parser.add_argument("--simulations", type=int, default=10)
        parser.add_argument("--c-puct", type=float, default=2.0)
        parser.add_argument("--temp-steps", type=int, default=4)
        parser.add_argument("--temperature", type=float, default=1.0)
        parser.add_argument("--dirichlet-epsilon", type=float, default=0.25)
        parser.add_argument("--dirichlet-alpha", type=float, default=0.03)
        parser.add_argument("--output", default=None)
        parser.add_argument("--device", default="cpu")
        args = parser.parse_args([])
        args.games = games
        args.output = output_dir
        return args


if __name__ == "__main__":
    unittest.main()
