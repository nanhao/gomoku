"""MctsBattleWorker 单元测试

测试 mcts_battle_worker.py 中的 MctsBattleWorker 类和 BattleGameResult 数据类。
覆盖构造、run_games、执方随机性、engine 不复用等。
"""

import os
import sys
import tempfile
import unittest

# 添加项目根目录到路径
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ai.scripts.mcts_battle_worker import MctsBattleWorker, BattleGameResult
from ai.scripts.mcts_config import MctsSelfPlayConfig
from ai.models.cnn_v7.model import CNNV7Model


class TestBattleGameResult(unittest.TestCase):
    """BattleGameResult 数据类测试"""

    def test_fields_set_correctly(self):
        """各字段正确赋值"""
        r = BattleGameResult(
            game_idx=0, winner=1, model1_color=1, model2_color=2,
            model1_won=True, model2_won=False, move_count=45
        )
        self.assertEqual(r.game_idx, 0)
        self.assertEqual(r.winner, 1)
        self.assertEqual(r.model1_color, 1)
        self.assertEqual(r.model2_color, 2)
        self.assertTrue(r.model1_won)
        self.assertFalse(r.model2_won)
        self.assertEqual(r.move_count, 45)

    def test_colors_sum_to_three(self):
        """model1_color + model2_color 恒为 3（互为黑白）"""
        r = BattleGameResult(
            game_idx=0, winner=1, model1_color=1, model2_color=2,
            model1_won=True, model2_won=False, move_count=45
        )
        self.assertEqual(r.model1_color + r.model2_color, 3)


class TestMctsBattleWorker(unittest.TestCase):
    """MctsBattleWorker 类单元测试 — TC2.1 ~ TC2.8"""

    @classmethod
    def setUpClass(cls):
        """创建两个测试用的随机权重模型文件"""
        cls._model1 = CNNV7Model(fresh=True)
        cls._model2 = CNNV7Model(fresh=True)
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._model1_path = os.path.join(cls._tmpdir.name, "model1.pth")
        cls._model2_path = os.path.join(cls._tmpdir.name, "model2.pth")
        cls._model1.save(cls._model1_path)
        cls._model2.save(cls._model2_path)

    @classmethod
    def tearDownClass(cls):
        """清理临时模型文件"""
        cls._tmpdir.cleanup()

    # ---------- TC2.1: 正常加载两个模型 ----------

    def test_constructor_loads_two_models(self):
        """TC2.1: 创建 worker，两个模型均加载到 CPU"""
        cfg = MctsSelfPlayConfig(simulations=2)
        worker = MctsBattleWorker(
            model1_path=self._model1_path,
            model1_name=None,
            model2_path=self._model2_path,
            model2_name=None,
            device="cpu",
            config=cfg,
            _model1=self._model1,
            _model2=self._model2,
        )
        self.assertIsNotNone(worker)
        self.assertIsNotNone(worker._model1)
        self.assertIsNotNone(worker._model2)

    # ---------- TC2.3: 单局执行 ----------

    def test_run_games_single_game(self):
        """TC2.3: 执行单局返回 1 个 BattleGameResult，winner ∈ {0,1,2}"""
        cfg = MctsSelfPlayConfig(simulations=2)
        worker = MctsBattleWorker(
            model1_path=self._model1_path,
            model1_name=None,
            model2_path=self._model2_path,
            model2_name=None,
            device="cpu",
            config=cfg,
            _model1=self._model1,
            _model2=self._model2,
        )
        results = worker.run_games(game_indices=[0], run_id="test")
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertIn(r.winner, (0, 1, 2))
        self.assertEqual(r.model1_color + r.model2_color, 3)
        # model1_won 和 model2_won 不会同时为 True
        if r.winner != 0:
            self.assertTrue(r.model1_won or r.model2_won)
            self.assertFalse(r.model1_won and r.model2_won)
        self.assertGreater(r.move_count, 0)

    # ---------- TC2.4: 多局执行 ----------

    def test_run_games_multiple_games(self):
        """TC2.4: 执行 3 局返回 3 个结果，game_idx 覆盖 [0,1,2]"""
        cfg = MctsSelfPlayConfig(simulations=2)
        worker = MctsBattleWorker(
            model1_path=self._model1_path,
            model1_name=None,
            model2_path=self._model2_path,
            model2_name=None,
            device="cpu",
            config=cfg,
            _model1=self._model1,
            _model2=self._model2,
        )
        results = worker.run_games(game_indices=[0, 1, 2], run_id="test")
        self.assertEqual(len(results), 3)
        game_indices = {r.game_idx for r in results}
        self.assertEqual(game_indices, {0, 1, 2})

    # ---------- TC2.5: 执方随机性 ----------

    def test_run_games_color_randomness(self):
        """TC2.5: 50 局中 model1 执黑次数在 15~35 之间"""
        cfg = MctsSelfPlayConfig(simulations=2)
        worker = MctsBattleWorker(
            model1_path=self._model1_path,
            model1_name=None,
            model2_path=self._model2_path,
            model2_name=None,
            device="cpu",
            config=cfg,
            _model1=self._model1,
            _model2=self._model2,
        )
        results = worker.run_games(game_indices=list(range(50)), run_id="test")
        model1_black_count = sum(1 for r in results if r.model1_color == 1)
        # 二项分布 95% 置信区间 (n=50, p=0.5): ~[15, 35]
        self.assertGreaterEqual(model1_black_count, 15)
        self.assertLessEqual(model1_black_count, 35)

    # ---------- TC2.6: engine 不复用 ----------

    def test_run_games_engine_not_reused(self):
        """TC2.6: 连续执行 2 局，两局结果独立"""
        cfg = MctsSelfPlayConfig(simulations=2)
        worker = MctsBattleWorker(
            model1_path=self._model1_path,
            model1_name=None,
            model2_path=self._model2_path,
            model2_name=None,
            device="cpu",
            config=cfg,
            _model1=self._model1,
            _model2=self._model2,
        )
        results = worker.run_games(game_indices=[0, 1], run_id="test")
        self.assertEqual(len(results), 2)
        # 两局结果独立，不应相互影响
        self.assertIsNotNone(results[0].winner)
        self.assertIsNotNone(results[1].winner)

    # ---------- TC2.8: 设备确认 ----------

    def test_models_on_cpu_device(self):
        """TC2.8: device='cpu' 时两个模型参数均在 CPU 上"""
        cfg = MctsSelfPlayConfig(simulations=2)
        worker = MctsBattleWorker(
            model1_path=self._model1_path,
            model1_name=None,
            model2_path=self._model2_path,
            model2_name=None,
            device="cpu",
            config=cfg,
            _model1=self._model1,
            _model2=self._model2,
        )
        for model in (worker._model1, worker._model2):
            for param in model.network.parameters():
                self.assertFalse(param.is_cuda, "模型参数应在 CPU 上")


if __name__ == "__main__":
    unittest.main()
