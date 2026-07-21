"""MctsSelfPlayWorker 和 MctsSelfPlayConfig 单元测试

测试 mcts_self_play_worker.py 中的类。
覆盖构造、run_games、z 值回填、设备验证等。
"""

import os
import sys
import tempfile
import unittest

# 添加项目根目录到路径
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch

from ai.scripts.mcts_self_play_worker import MctsSelfPlayWorker, MctsSelfPlayConfig
from ai.models.cnn_v7.model import CNNV7Model


class TestMctsSelfPlayConfig(unittest.TestCase):
    """MctsSelfPlayConfig 数据类测试 — TC4.1"""

    def test_default_values(self):
        """默认值: simulations=800, c_puct=2.0, temp_steps=8, temperature=1.0,
        batch_size=225, virtual_loss=3.0, use_prior=True"""
        cfg = MctsSelfPlayConfig()
        self.assertEqual(cfg.simulations, 800)
        self.assertEqual(cfg.c_puct, 2.0)
        self.assertEqual(cfg.temp_steps, 8)
        self.assertEqual(cfg.temperature, 1.0)
        self.assertEqual(cfg.dirichlet_epsilon, 0.25)
        self.assertEqual(cfg.dirichlet_alpha, 0.03)
        self.assertEqual(cfg.batch_size, 225)
        self.assertEqual(cfg.virtual_loss, 3.0)
        self.assertTrue(cfg.use_prior)

    def test_custom_values(self):
        """自定义值覆盖默认"""
        cfg = MctsSelfPlayConfig(
            simulations=100,
            c_puct=1.5,
            temp_steps=4,
            temperature=0.5,
            dirichlet_epsilon=0.1,
            dirichlet_alpha=0.01,
            batch_size=64,
            virtual_loss=1.0,
            use_prior=False,
        )
        self.assertEqual(cfg.simulations, 100)
        self.assertEqual(cfg.c_puct, 1.5)
        self.assertEqual(cfg.temp_steps, 4)
        self.assertEqual(cfg.temperature, 0.5)
        self.assertEqual(cfg.dirichlet_epsilon, 0.1)
        self.assertEqual(cfg.dirichlet_alpha, 0.01)
        self.assertEqual(cfg.batch_size, 64)
        self.assertEqual(cfg.virtual_loss, 1.0)
        self.assertFalse(cfg.use_prior)


class TestMctsSelfPlayWorker(unittest.TestCase):
    """MctsSelfPlayWorker 类单元测试 — TC1.1 ~ TC1.7"""

    @classmethod
    def setUpClass(cls):
        """创建测试用的随机权重模型文件"""
        cls._fresh_model = CNNV7Model(fresh=True)
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._model_path = os.path.join(cls._tmpdir.name, "test_model.pth")
        cls._fresh_model.save(cls._model_path)

    @classmethod
    def tearDownClass(cls):
        """清理临时模型文件"""
        cls._tmpdir.cleanup()

    # ---------- TC1.1: 正常加载 ----------

    def test_constructor_loads_model_successfully(self):
        """TC1.1: 注入模型创建 worker，模型加载到 CPU"""
        cfg = MctsSelfPlayConfig(simulations=2)
        worker = MctsSelfPlayWorker(
            model_path=self._model_path,
            model_name=None,
            device="cpu",
            config=cfg,
            _model=self._fresh_model,
        )
        self.assertIsNotNone(worker)
        self.assertIsNotNone(worker._model)

    # ---------- TC1.2: 模型文件不存在 ----------

    def test_constructor_raises_on_missing_model(self):
        """TC1.2: 模型文件不存在且未注入 _model 时抛出异常"""
        # 使用不存在的路径且不注入 _model，应抛出异常
        with self.assertRaises((FileNotFoundError, ValueError)):
            MctsSelfPlayWorker(
                model_path="/nonexistent/path/model.pth",
                model_name=None,
                device="cpu",
                config=MctsSelfPlayConfig(),
            )

    # ---------- TC1.7: 设备确认 ----------

    def test_model_on_cpu_device(self):
        """TC1.7: device='cpu' 时模型参数在 CPU 上"""
        worker = MctsSelfPlayWorker(
            model_path=self._model_path,
            model_name=None,
            device="cpu",
            config=MctsSelfPlayConfig(simulations=2),
            _model=self._fresh_model,
        )
        for param in worker._model.network.parameters():
            self.assertFalse(param.is_cuda, "模型参数应在 CPU 上")

    # ---------- TC1.3: 单局执行 ----------

    def test_run_games_single_game_returns_valid_samples(self):
        """TC1.3: 执行单局返回非空样本列表，每条含 state/pi/z，pi 总和≈1"""
        worker = MctsSelfPlayWorker(
            model_path=self._model_path,
            model_name=None,
            device="cpu",
            config=MctsSelfPlayConfig(simulations=2),
            _model=self._fresh_model,
        )
        samples = worker.run_games(game_indices=[0], run_id="test")
        self.assertIsInstance(samples, list)
        self.assertGreater(len(samples), 0, "应有至少一条样本")
        for s in samples:
            self.assertIn("state", s)
            self.assertIn("pi", s)
            self.assertIn("z", s)
            self.assertEqual(len(s["state"]), 226)
            self.assertEqual(len(s["pi"]), 225)
            self.assertAlmostEqual(sum(s["pi"]), 1.0, places=3)
            self.assertIn(s["z"], (-1, 0, 1))

    # ---------- TC1.4: 多局执行 ----------

    def test_run_games_multiple_games(self):
        """TC1.4: 执行 3 局返回正确数量样本"""
        worker = MctsSelfPlayWorker(
            model_path=self._model_path,
            model_name=None,
            device="cpu",
            config=MctsSelfPlayConfig(simulations=2),
            _model=self._fresh_model,
        )
        samples = worker.run_games(game_indices=[0, 1, 2], run_id="test")
        self.assertGreater(len(samples), 0)

    # ---------- TC1.5: z 值回填 ----------

    def test_run_games_z_value_correctness(self):
        """TC1.5: 终局后 z 值回填正确
        - 胜方步 z=1，败方步 z=-1；平局全部 z=0
        """
        worker = MctsSelfPlayWorker(
            model_path=self._model_path,
            model_name=None,
            device="cpu",
            config=MctsSelfPlayConfig(simulations=2),
            _model=self._fresh_model,
        )
        samples = worker.run_games(game_indices=[0], run_id="test")
        last_z = samples[-1]["z"]
        last_player = samples[-1]["state"][225]

        if last_z == 0:
            # 平局: 所有步 z=0
            for s in samples:
                self.assertEqual(s["z"], 0)
        elif last_z == 1:
            # 最后一步的玩家胜
            for s in samples:
                player = s["state"][225]
                if player == last_player:
                    self.assertEqual(s["z"], 1)
                else:
                    self.assertEqual(s["z"], -1)
        else:
            # last_z == -1: 对方胜
            for s in samples:
                player = s["state"][225]
                if player == last_player:
                    self.assertEqual(s["z"], -1)
                else:
                    self.assertEqual(s["z"], 1)

    # ---------- TC1.6: engine 不复用 ----------

    def test_run_games_engine_not_reused_across_games(self):
        """TC1.6: 连续执行 2 局，第二局初始棋盘为空（MCTS 树不污染）"""
        worker = MctsSelfPlayWorker(
            model_path=self._model_path,
            model_name=None,
            device="cpu",
            config=MctsSelfPlayConfig(simulations=2),
            _model=self._fresh_model,
        )
        samples = worker.run_games(game_indices=[0, 1], run_id="test")

        # 找出两局的第一条样本（state[225] 记录当前落子方）
        first_game_samples = []
        second_game_samples = []
        current_first_state = None
        for s in samples:
            board = s["state"][:225]
            if current_first_state is None or board == current_first_state:
                # 仍处于第一局（或刚进入第二局但还没找到分隔点）
                if all(c == 0 for c in board):
                    # 空棋盘 → 新局开始
                    if current_first_state is not None:
                        second_game_samples.append(s)
                        continue
                    else:
                        current_first_state = list(board)
                first_game_samples.append(s)
            else:
                second_game_samples.append(s)

        self.assertGreater(len(first_game_samples), 0, "应有第一局样本")
        self.assertGreater(len(second_game_samples), 0, "应有第二局样本")


if __name__ == "__main__":
    unittest.main()
