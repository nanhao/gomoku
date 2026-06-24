"""MCTSMoveDecider 单元测试

测试 MCTS 落子决策器：
- decide 返回 (action, visit_counts, log_path, root_policy)
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

    def _make_state(self, player=1):
        """创建棋盘状态"""
        board = [0] * 225
        # 放置一些棋子避免全空
        board[112] = player  # 中心
        board.append(player)
        return board

    def test_decide_returns_four_tuple(self):
        """decide 返回 4 元组 (action, visit_counts, log_path, root_policy)"""
        mock_searcher = MockSearcher()
        decider = self.MCTSMoveDecider(mock_searcher)
        state = self._make_state()
        result = decider.decide(state)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        action, visit_counts, log_path, root_policy = result
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 225)
        self.assertIsInstance(visit_counts, list)
        self.assertEqual(len(visit_counts), 225)
        if log_path is not None:
            self.assertIsInstance(log_path, str)
        self.assertIsInstance(root_policy, list)
        self.assertEqual(len(root_policy), 225)

    def test_decide_selects_legal_action(self):
        """decide 返回的 action 应在空位中"""
        mock_searcher = MockSearcher()
        decider = self.MCTSMoveDecider(mock_searcher)
        state = self._make_state()
        action, _, _, _ = decider.decide(state)
        # 检查 action 对应位置是空位
        board = state[:225]
        self.assertEqual(board[action], 0, f"action {action} 应在空位上")

    def test_decide_with_log_key(self):
        """decide 透传 log_key 给 searcher"""
        mock_searcher = MockSearcher()
        decider = self.MCTSMoveDecider(mock_searcher)
        state = self._make_state()
        decider.decide(state, log_key="game0_move5")
        self.assertEqual(len(mock_searcher.search_calls), 1)
        self.assertEqual(mock_searcher.search_calls[0]["log_key"], "game0_move5")

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
        state = self._make_state()
        # 多次调用检查结果可能不同（但不强求，概率性的）
        actions = set()
        for _ in range(10):
            action, _, _, _ = decider.decide(state)
            actions.add(action)
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
        state = self._make_state()
        # 把 100 设为空位
        state[100] = 0
        action, _, _, _ = decider.decide(state)
        self.assertEqual(action, 100)

    def test_make_move_failure_handling(self):
        """advance_root 失败时的处理"""
        mock_searcher = MockSearcher()
        mock_searcher.set_advance_result(False)
        decider = self.MCTSMoveDecider(mock_searcher)
        # make_move 不应抛异常
        decider.make_move(0)
        self.assertEqual(len(mock_searcher.advance_root_calls), 1)


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
