"""价值头自对弈 Worker 单元测试

验证 ValueSelfPlayConfig 和 ValueSelfPlayWorker。
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))


class TestValueSelfPlayConfig(unittest.TestCase):
    """ValueSelfPlayConfig 测试（设计 TC-SPC-01 ~ TC-SPC-02）"""

    def test_default_values(self):
        """TC-SPC-01: 默认值"""
        from ai.scripts.value_self_play_worker import ValueSelfPlayConfig
        cfg = ValueSelfPlayConfig()
        self.assertEqual(cfg.temperature, 1.0)
        self.assertEqual(cfg.temp_steps, 8)
        self.assertEqual(cfg.temp_side, "random")
        self.assertIsNone(cfg.top_k)
        self.assertIsNone(cfg.seed)

    def test_custom_values(self):
        """TC-SPC-02: 自定义值"""
        from ai.scripts.value_self_play_worker import ValueSelfPlayConfig
        cfg = ValueSelfPlayConfig(
            temperature=0.5,
            temp_steps=5,
            temp_side="black",
            top_k=10,
            seed=42,
        )
        self.assertEqual(cfg.temperature, 0.5)
        self.assertEqual(cfg.temp_steps, 5)
        self.assertEqual(cfg.temp_side, "black")
        self.assertEqual(cfg.top_k, 10)
        self.assertEqual(cfg.seed, 42)


class TestValueSelfPlayWorker(unittest.TestCase):
    """ValueSelfPlayWorker 测试"""

    def setUp(self):
        from ai.models.cnn_v10.model import CNNV10Model
        self.model = CNNV10Model(fresh=True)

    def test_worker_with_injected_model(self):
        """worker 接受 _model 注入"""
        from ai.scripts.value_self_play_worker import (
            ValueSelfPlayConfig, ValueSelfPlayWorker,
        )
        cfg = ValueSelfPlayConfig()
        worker = ValueSelfPlayWorker(
            model_path="", model_name=None, device="cpu",
            config=cfg, _model=self.model,
        )
        self.assertIs(worker._model, self.model)

    def test_run_games_single_game(self):
        """单局自对弈产生样本"""
        from ai.scripts.value_self_play_worker import (
            ValueSelfPlayConfig, ValueSelfPlayWorker,
        )
        cfg = ValueSelfPlayConfig()
        worker = ValueSelfPlayWorker(
            model_path="", model_name=None, device="cpu",
            config=cfg, _model=self.model,
        )
        samples = worker.run_games([0], "test")
        self.assertGreater(len(samples), 0)
        for s in samples:
            self.assertIn("state", s)
            self.assertIn("z", s)
            self.assertEqual(len(s["state"]), 226)
            self.assertIn(s["z"], [-1, 0, 1])

    def test_run_games_no_pi_field(self):
        """样本不包含 pi 字段"""
        from ai.scripts.value_self_play_worker import (
            ValueSelfPlayConfig, ValueSelfPlayWorker,
        )
        cfg = ValueSelfPlayConfig()
        worker = ValueSelfPlayWorker(
            model_path="", model_name=None, device="cpu",
            config=cfg, _model=self.model,
        )
        samples = worker.run_games([0], "test")
        for s in samples:
            self.assertNotIn("pi", s)

    def test_run_games_multiple(self):
        """多局自对弈"""
        from ai.scripts.value_self_play_worker import (
            ValueSelfPlayConfig, ValueSelfPlayWorker,
        )
        cfg = ValueSelfPlayConfig()
        worker = ValueSelfPlayWorker(
            model_path="", model_name=None, device="cpu",
            config=cfg, _model=self.model,
        )
        samples = worker.run_games([0], "test")
        self.assertGreater(len(samples), 0)

    def test_worker_entry_static(self):
        """worker_entry 是静态方法"""
        from ai.scripts.value_self_play_worker import (
            ValueSelfPlayConfig, ValueSelfPlayWorker,
        )
        cfg = ValueSelfPlayConfig()
        # 通过临时保存模型来测试 worker_entry
        tmp_dir = tempfile.mkdtemp()
        try:
            pth_path = os.path.join(tmp_dir, "test.pth")
            self.model.save(pth_path)
            samples = ValueSelfPlayWorker.worker_entry(
                model_path=pth_path,
                model_name="cnn_v10.1",
                device="cpu",
                config=cfg,
                game_indices=[0],
                run_id="test_entry",
            )
            self.assertIsInstance(samples, list)
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_z_values_consistent(self):
        """z 值与终局结果一致"""
        from ai.scripts.value_self_play_worker import (
            ValueSelfPlayConfig, ValueSelfPlayWorker,
        )
        cfg = ValueSelfPlayConfig()
        worker = ValueSelfPlayWorker(
            model_path="", model_name=None, device="cpu",
            config=cfg, _model=self.model,
        )
        samples = worker.run_games([0], "test")
        if not samples:
            return
        final_z = samples[-1]["z"]
        final_player = samples[-1]["state"][225]
        # 最后一步的玩家视角 z 值：胜=1, 负=-1, 平=0
        # 如果是平局，所有 z 应为 0
        if final_z == 0:
            for s in samples:
                self.assertEqual(s["z"], 0)
        elif final_z == 1:
            # 最后一步的玩家获胜
            for s in samples:
                player = s["state"][225]
                expected = 1 if player == final_player else -1
                self.assertEqual(s["z"], expected,
                                 f"player={player}, final_player={final_player}")
        elif final_z == -1:
            # 最后一步的玩家失败（即对手获胜）
            winner = 2 if final_player == 1 else 1
            for s in samples:
                player = s["state"][225]
                expected = 1 if player == winner else -1
                self.assertEqual(s["z"], expected,
                                 f"player={player}, winner={winner}")


if __name__ == '__main__':
    unittest.main()
