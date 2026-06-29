"""PolicyValueAiEngine 单元测试（意图 34：T34-07 ~ T34-13）

测试 PolicyValueAiEngine 继承关系、构造函数、get_move 委托 MCTS、
after_move 推进根节点、终局节点 after_move 行为。
"""

import sys
import os
import unittest

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np

# 复用 test_mcts_searcher 中的 MockModel 和辅助函数
from tests.unit.test_mcts_searcher import MockModel, _make_empty_state


class TestPolicyValueInheritance(unittest.TestCase):
    """T34-07：PolicyValueAiEngine 继承 AIEngine"""

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

    def test_t34_07_is_subclass_of_ai_engine(self):
        """T34-07：PolicyValueAiEngine 是 AIEngine 的子类"""
        from ai.engine import AIEngine, PolicyValueAiEngine
        self.assertTrue(issubclass(PolicyValueAiEngine, AIEngine))

    def test_is_instance_of_ai_engine(self):
        """PolicyValueAiEngine 实例也是 AIEngine 实例"""
        from ai.engine import AIEngine, PolicyValueAiEngine

        model = MockModel()
        searcher = self.MCTSSearcher(model, num_simulations=10)
        engine = PolicyValueAiEngine(searcher)
        self.assertIsInstance(engine, AIEngine)


class TestPolicyValueConstructor(unittest.TestCase):
    """T34-08：PolicyValueAiEngine 构造函数参数验证"""

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

    def test_t34_08_default_constructor_values(self):
        """T34-08：默认构造参数正确"""
        from ai.engine import PolicyValueAiEngine

        model = MockModel()
        searcher = self.MCTSSearcher(model, num_simulations=10)
        engine = PolicyValueAiEngine(searcher)

        self.assertEqual(engine.temperature, 1.0)
        self.assertEqual(engine.temp_steps, 8)
        self.assertIsNone(engine.log_key_prefix)
        self.assertEqual(engine._step_count, 0)
        self.assertEqual(engine._search_id, 0)

    def test_custom_constructor_values(self):
        """自定义构造参数正确设置"""
        from ai.engine import PolicyValueAiEngine

        model = MockModel()
        searcher = self.MCTSSearcher(model, num_simulations=10)
        engine = PolicyValueAiEngine(
            searcher,
            temperature=0.5,
            temp_steps=12,
            log_key_prefix="run1_game0",
        )

        self.assertEqual(engine.temperature, 0.5)
        self.assertEqual(engine.temp_steps, 12)
        self.assertEqual(engine.log_key_prefix, "run1_game0")


class TestPolicyValueGetMove(unittest.TestCase):
    """T34-09 ~ T34-10：get_move 委托 MCTS 搜索"""

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

    def test_t34_09_get_move_returns_move_decision(self):
        """T34-09：get_move 返回 MoveDecision 实例"""
        from ai.engine import PolicyValueAiEngine, MoveDecision

        model = MockModel()
        searcher = self.MCTSSearcher(model, num_simulations=20)
        engine = PolicyValueAiEngine(searcher)

        board_2d = [[0] * 15 for _ in range(15)]
        md = engine.get_move(board_2d, 1, strategy="argmax")
        self.assertIsInstance(md, MoveDecision)

    def test_t34_10_get_move_contains_mcts_visit_counts(self):
        """T34-10：get_move 返回的 MoveDecision 包含 MCTS 访问计数"""
        from ai.engine import PolicyValueAiEngine

        model = MockModel()
        searcher = self.MCTSSearcher(model, num_simulations=20)
        engine = PolicyValueAiEngine(searcher)

        board_2d = [[0] * 15 for _ in range(15)]
        md = engine.get_move(board_2d, 1, strategy="argmax")

        visit_counts = md.get_mcts_visit_counts()
        self.assertIsNotNone(visit_counts)
        self.assertEqual(len(visit_counts), 225)


class TestPolicyValueAfterMove(unittest.TestCase):
    """T34-11 ~ T34-13：after_move 推进根节点"""

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

    def test_t34_11_after_move_advances_root(self):
        """T34-11：after_move 成功推进根节点"""
        from ai.engine import PolicyValueAiEngine

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
        action = x * 15 + y

        # 推进不应抛异常
        engine.after_move(action)

        # 步数应递增
        self.assertEqual(engine._step_count, 1)

    def test_t34_12_after_move_invalid_action_raises(self):
        """T34-12：推进不存在的子节点抛 RuntimeError"""
        from ai.engine import PolicyValueAiEngine

        model = MockModel()
        searcher = self.MCTSSearcher(
            model, num_simulations=10,
            dirichlet_epsilon=0.0,
        )
        engine = PolicyValueAiEngine(searcher)

        # 空棋盘搜索——但 action 225 不在搜索树中
        board_2d = [[0] * 15 for _ in range(15)]
        np.random.seed(42)
        engine.get_move(board_2d, 1, strategy="argmax")

        with self.assertRaises(RuntimeError):
            engine.after_move(999)  # 无效索引

    def test_t34_13_after_move_terminal_node(self):
        """T34-13：终局节点 after_move 不抛异常（必胜落子后的推进）"""
        from ai.engine import PolicyValueAiEngine

        # 构造必胜棋局：黑方 (7,0)-(7,3) 四子，下一步 (7,4) 获胜
        board_flat = [0] * 225
        for y in range(4):
            board_flat[7 * 15 + y] = 1

        win_action = 7 * 15 + 4

        # 偏向必胜落子的策略
        policy = [0.001] * 225
        policy[win_action] = 10.0
        total = sum(policy)
        policy = [p / total for p in policy]

        model = MockModel(fixed_policy=policy, fixed_value=0.0)
        searcher = self.MCTSSearcher(
            model, num_simulations=50,
            dirichlet_epsilon=0.0,
        )
        engine = PolicyValueAiEngine(searcher, temperature=0.0)  # 贪婪

        # 将 flat board 转为 15×15 二维矩阵
        board_2d = [[board_flat[i * 15 + j] for j in range(15)] for i in range(15)]

        np.random.seed(42)
        md = engine.get_move(board_2d, 1, strategy="argmax")
        x, y = md.get_position()
        action = x * 15 + y

        self.assertEqual(action, win_action,
                         f"Expected winning action {win_action}, got {action}")

        # 终局节点推进不应抛异常
        try:
            engine.after_move(action)
        except RuntimeError as e:
            self.fail(f"after_move({action}) on terminal node should not raise: {e}")


if __name__ == "__main__":
    unittest.main()
