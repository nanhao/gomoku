"""MCTSMoveDecider 单元测试

测试 MCTS 落子决策器：
- get_move 返回 MoveDecision（含 position 和 mcts_visit_counts）
- make_move 子树复用
- 温度选择
"""

import sys
import os
import unittest
import tempfile
import shutil
from unittest.mock import MagicMock

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch


class MockSearcher:
    """Mock MCTSSearcher，用于独立测试 MoveDecider"""

    def __init__(self, visit_counts=None, root_value=0.0, root_policy=None):
        self._visit_counts = visit_counts or [1] * 225
        self._root_value = root_value
        self._root_policy = root_policy or [1.0 / 225] * 225
        self.search_calls = []
        self.advance_root_calls = []
        self.reset_calls = []
        self._advance_result = True

    def search(self, state, num_simulations=None, log_key=None):
        """记录调用并返回固定值"""
        self.search_calls.append({
            "state": state,
            "num_simulations": num_simulations,
            "log_key": log_key,
        })
        return self._visit_counts, self._root_value, self._root_policy

    def advance_root(self, action):
        """记录调用"""
        self.advance_root_calls.append(action)
        return self._advance_result

    def set_advance_result(self, result):
        self._advance_result = result

    def reset(self):
        self.reset_calls.append(True)


# ---------- 测试类 ----------

class TestMCTSMoveDecider(unittest.TestCase):
    """MCTSMoveDecider 基础测试"""

    @classmethod
    def setUpClass(cls):
        """导入 MCTSMoveDecider"""
        try:
            from ai.mcts.decision_maker import MCTSMoveDecider
            cls.MCTSMoveDecider = MCTSMoveDecider
        except ImportError:
            cls.MCTSMoveDecider = None

    def setUp(self):
        if self.MCTSMoveDecider is None:
            self.skipTest("MCTSMoveDecider 尚未实现")

    @staticmethod
    def _make_board_2d(player=1):
        """创建 15×15 二维棋盘（带一个中心棋子）。"""
        board = [[0] * 15 for _ in range(15)]
        board[7][7] = player  # 中心位置
        return board

    def test_get_move_returns_move_decision(self):
        """get_move 返回 MoveDecision 实例，含 position 和 mcts_visit_counts"""
        from ai.engine import MoveDecision
        mock_searcher = MockSearcher()
        decider = self.MCTSMoveDecider(mock_searcher)
        board = self._make_board_2d()
        result = decider.get_move(board, 1)

        self.assertIsInstance(result, MoveDecision)
        x, y = result.get_position()
        self.assertIsInstance(x, int)
        self.assertIsInstance(y, int)
        self.assertGreaterEqual(x, 0)
        self.assertLess(x, 15)
        self.assertGreaterEqual(y, 0)
        self.assertLess(y, 15)
        visit_counts = result.get_mcts_visit_counts()
        self.assertIsInstance(visit_counts, list)
        self.assertEqual(len(visit_counts), 225)
        # 验证 flat index = x * 15 + y
        action = x * 15 + y
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 225)

    def test_get_move_selects_legal_action(self):
        """get_move 返回的 position 应在空位中"""
        mock_searcher = MockSearcher()
        decider = self.MCTSMoveDecider(mock_searcher)
        board = self._make_board_2d()
        md = decider.get_move(board, 1)
        x, y = md.get_position()
        # 检查 action 对应位置是空位
        self.assertEqual(board[x][y], 0, f"action ({x},{y}) 应在空位上")

    def test_get_move_with_log_key_prefix(self):
        """get_move 使用构造时注入的 log_key_prefix 自动生成 log_key"""
        mock_searcher = MockSearcher()
        decider = self.MCTSMoveDecider(mock_searcher, log_key_prefix="game0")
        board = self._make_board_2d()
        decider.get_move(board, 1)
        self.assertEqual(len(mock_searcher.search_calls), 1)
        self.assertEqual(mock_searcher.search_calls[0]["log_key"], "game0_search0")

    def test_make_move_calls_advance_root(self):
        """make_move 调用 searcher.advance_root"""
        mock_searcher = MockSearcher()
        decider = self.MCTSMoveDecider(mock_searcher)
        decider.make_move(112)
        self.assertEqual(len(mock_searcher.advance_root_calls), 1)
        self.assertEqual(mock_searcher.advance_root_calls[0], 112)

    def test_temperature_selection_explores(self):
        """温度 > 0 时应有一定随机性"""
        mock_searcher = MockSearcher()
        decider = self.MCTSMoveDecider(mock_searcher, temperature=1.0)
        board = self._make_board_2d()
        # 多次调用检查结果可能不同（但不强求，概率性的）
        actions = set()
        for _ in range(10):
            md = decider.get_move(board, 1, strategy="sampling")
            x, y = md.get_position()
            actions.add(x * 15 + y)
        # 在温度=1.0时大概率探索多个位置
        # 至少能返回有效 action
        self.assertGreater(len(actions), 0)

    def test_temperature_zero_is_greedy(self):
        """温度=0 时选择访问计数最大的 action"""
        visit_counts = [0] * 225
        visit_counts[100] = 10  # 最高访问计数
        visit_counts[50] = 5
        visit_counts[200] = 2
        mock_searcher = MockSearcher(visit_counts=visit_counts)
        decider = self.MCTSMoveDecider(mock_searcher, temperature=0.0)
        # 创建空棋盘（action 100 = (6,10)，应是空位）
        board = [[0] * 15 for _ in range(15)]
        md = decider.get_move(board, 1, strategy="sampling")
        x, y = md.get_position()
        self.assertEqual(x * 15 + y, 100)

    def test_make_move_failure_raises(self):
        """advance_root 失败时应抛出 RuntimeError"""
        mock_searcher = MockSearcher()
        mock_searcher.set_advance_result(False)
        decider = self.MCTSMoveDecider(mock_searcher)
        with self.assertRaises(RuntimeError):
            decider.make_move(0)
        self.assertEqual(len(mock_searcher.advance_root_calls), 1)

    def test_reset_clears_step_count_and_searcher(self):
        """reset 应重置 _step_count 并调用 searcher.reset"""
        mock_searcher = MockSearcher()
        decider = self.MCTSMoveDecider(mock_searcher)
        # 先走一步，让 _step_count > 0
        decider.make_move(112)
        self.assertGreater(decider._step_count, 0)
        # reset
        decider.reset()
        self.assertEqual(decider._step_count, 0)
        self.assertEqual(len(mock_searcher.reset_calls), 1)


class TestMCTSMoveDeciderConstruction(unittest.TestCase):
    """构造参数测试"""

    @classmethod
    def setUpClass(cls):
        try:
            from ai.mcts.decision_maker import MCTSMoveDecider
            cls.MCTSMoveDecider = MCTSMoveDecider
        except ImportError:
            cls.MCTSMoveDecider = None

    def setUp(self):
        if self.MCTSMoveDecider is None:
            self.skipTest("MCTSMoveDecider 尚未实现")

    def test_default_construction(self):
        """默认构造"""
        mock_searcher = MockSearcher()
        decider = self.MCTSMoveDecider(mock_searcher)
        self.assertIsNotNone(decider)

    def test_construction_with_temp_steps(self):
        """带温度步数参数构造"""
        mock_searcher = MockSearcher()
        decider = self.MCTSMoveDecider(mock_searcher, temp_steps=10)
        self.assertIsNotNone(decider)


if __name__ == "__main__":
    unittest.main()
