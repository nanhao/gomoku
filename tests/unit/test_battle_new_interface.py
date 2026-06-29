"""CLI battle + mcts_self_play 适配测试（意图 35：T35-15 ~ T35-16）

测试内容：
- T35-15: CLI run_battle 直接传 ai 名称
- T35-16: mcts_self_play.py after_move(x, y) 签名适配
"""

import sys
import os
import unittest
import shutil
import tempfile
from unittest.mock import patch, MagicMock

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ---------- T35-15：CLI run_battle 直接传 ai 名称 ----------

class TestT35_15_CLIBattleWithAiNames(unittest.TestCase):
    """T35-15：CLI run_battle("random", "random", 1) 直接传 ai 名称"""

    def setUp(self):
        self.test_save_dir = tempfile.mkdtemp(prefix="test_t35_15_save_")
        self.test_ai_data_dir = tempfile.mkdtemp(prefix="test_t35_15_ai_data_")
        # Patch GomokuController 使其使用临时存档目录
        self._patcher = patch(
            'ai.game_manager.GomokuController',
            lambda save_dir=None: __import__('game').GomokuController(
                save_dir=self.test_save_dir
            )
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.test_save_dir, ignore_errors=True)
        shutil.rmtree(self.test_ai_data_dir, ignore_errors=True)

    def test_run_battle_with_ai_names_one_game(self):
        """run_battle("random", "random", 1) 返回 1 局结果"""
        import ai.battle as battle_module

        # 使用 FakeDataRecorder 避免写入真实 AI_DATA_DIR
        original_data_recorder = battle_module.DataRecorder

        class FakeDataRecorder(original_data_recorder):
            def __init__(self_, ai_name, ai_data_dir=None):
                super().__init__(ai_name, self.test_ai_data_dir)

        battle_module.DataRecorder = FakeDataRecorder
        try:
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                results = battle_module.run_battle("random", "random", 1)

            # 验证 1 局完成
            self.assertEqual(len(results), 1,
                             "run_battle 应返回 1 局结果")
            self.assertIn("winner", results[0],
                          "每局结果应含 winner 字段")

            output = f.getvalue()
            self.assertIn("第 1 局", output,
                          "输出应包含第 1 局")
            self.assertIn("=== 对战结果 ===", output,
                          "输出应包含对战结果")
        finally:
            battle_module.DataRecorder = original_data_recorder

    def test_run_battle_does_not_directly_import_get_model_or_engine(self):
        """验证 run_battle 内部不直接引用 get_model / SinglePolicyAiEngine"""
        import ai.battle as battle_module
        import inspect

        source = inspect.getsource(battle_module.run_battle)
        # run_battle 不应包含 get_model 或 SinglePolicyAiEngine 的直接调用
        # 注意：import 行可能包含这些名称，但函数体不应使用它们
        self.assertNotIn("get_model(", source.replace(" ", ""),
                         "run_battle 不应直接调用 get_model")
        self.assertNotIn("SinglePolicyAiEngine(", source.replace(" ", ""),
                         "run_battle 不应直接创建 SinglePolicyAiEngine")


# ---------- T35-16：mcts_self_play after_move(x, y) 签名适配 ----------

class TestT35_16_MctsSelfPlayAfterMove(unittest.TestCase):
    """T35-16：mcts_self_play.py after_move(x, y) 签名适配"""

    @classmethod
    def setUpClass(cls):
        try:
            from ai.mcts.searcher import MCTSSearcher
            cls.MCTSSearcher = MCTSSearcher
        except ImportError:
            cls.MCTSSearcher = None

    def setUp(self):
        if self.MCTSSearcher is None:
            self.skipTest("MCTSSearcher 尚未实现")

    def test_after_move_with_xy_from_get_move(self):
        """取 get_move 返回的 (x, y)，调用 after_move(x, y) 不抛异常，_step_count == 1"""
        from ai.engine import PolicyValueAiEngine
        from tests.unit.test_mcts_searcher import MockModel
        import numpy as np

        model = MockModel()
        searcher = self.MCTSSearcher(
            model, num_simulations=50,
            dirichlet_epsilon=0.0,
        )
        engine = PolicyValueAiEngine(searcher)

        # 空棋盘搜索
        board_2d = [[0] * 15 for _ in range(15)]
        np.random.seed(42)
        md = engine.get_move(board_2d, 1, strategy="argmax")
        x, y = md.get_position()

        # 使用 (x, y) 坐标调用 after_move（代替旧 action 调用）
        engine.after_move(x, y)

        # 验证步数递增
        self.assertEqual(engine._step_count, 1,
                         "after_move 后 _step_count 应为 1")


if __name__ == "__main__":
    unittest.main()
