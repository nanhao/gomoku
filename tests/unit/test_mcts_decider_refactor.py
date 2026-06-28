"""MCTSMoveDecider 重构单元测试

对应设计文档 design/33/main_docs/test_design.md T3 和 T4：
- T3：MCTSMoveDecider 构造函数新增 log_key_prefix
- T4：MCTSMoveDecider get_move() 删除 state/log_key、新增 board/current_player/strategy
"""

import sys
import os
import unittest

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.engine import MoveDecision


# ---------------------------------------------------------------------------
# Mock Searcher：记录 search 调用参数，返回固定访问计数
# ---------------------------------------------------------------------------

class MockSearcher:
    """Mock MCTSSearcher，用于独立测试 MCTSMoveDecider。

    search_calls 记录每次 search() 调用的参数，便于验证：
    - state 是否正确（226 维向量）
    - log_key 是否正确（按前缀 + search_id 拼接）
    """

    def __init__(self, visit_counts=None, root_value=0.0, root_policy=None):
        self._visit_counts = visit_counts or [1] * 225
        self._root_value = root_value
        self._root_policy = root_policy or [1.0 / 225] * 225
        self.search_calls = []
        self.advance_root_calls = []
        self.reset_calls = []
        self._advance_result = True

    def search(self, state, num_simulations=None, log_key=None):
        """记录调用参数并返回固定值。"""
        self.search_calls.append({
            "state": state,
            "num_simulations": num_simulations,
            "log_key": log_key,
        })
        return self._visit_counts, self._root_value, self._root_policy

    def advance_root(self, action):
        self.advance_root_calls.append(action)
        return self._advance_result

    def set_advance_result(self, result):
        self._advance_result = result

    def reset(self):
        self.reset_calls.append(True)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _make_board_2d():
    """返回 15×15 的全零二维棋盘。"""
    return [[0] * 15 for _ in range(15)]


def _make_board_with_piece(x, y, player):
    """返回带一个棋子的 15×15 二维棋盘。"""
    board = [[0] * 15 for _ in range(15)]
    board[x][y] = player
    return board


# ---------------------------------------------------------------------------
# T3：MCTSMoveDecider 构造函数新增 log_key_prefix
# ---------------------------------------------------------------------------

class TestMCTSMoveDeciderLogKeyPrefix(unittest.TestCase):
    """T3：log_key_prefix 构造函数参数和 log_key 自动生成。"""

    @classmethod
    def setUpClass(cls):
        """导入 MCTSMoveDecider。"""
        try:
            from ai.mcts.decision_maker import MCTSMoveDecider
            cls.MCTSMoveDecider = MCTSMoveDecider
        except ImportError:
            cls.MCTSMoveDecider = None

    def setUp(self):
        if self.MCTSMoveDecider is None:
            self.skipTest("MCTSMoveDecider 尚未实现")

    # ---------- T3-01：log_key_prefix=None 不输出日志 ----------

    def test_no_log_key_prefix_produces_none_log_key(self):
        """T3-01：不传 log_key_prefix 时，search 收到 log_key=None。"""
        searcher = MockSearcher()
        decider = self.MCTSMoveDecider(searcher)
        board = _make_board_2d()
        decider.get_move(board, 1)
        self.assertEqual(len(searcher.search_calls), 1)
        self.assertIsNone(searcher.search_calls[0]["log_key"],
                          "log_key_prefix=None 时 log_key 应为 None")

    # ---------- T3-02：传入 log_key_prefix 时自动拼接 search_id ----------

    def test_log_key_prefix_auto_generates_search0(self):
        """T3-02：第一次 get_move 生成 log_key="runA_game0_search0"。"""
        searcher = MockSearcher()
        decider = self.MCTSMoveDecider(searcher, log_key_prefix="runA_game0")
        board = _make_board_2d()
        decider.get_move(board, 1)
        self.assertEqual(searcher.search_calls[0]["log_key"], "runA_game0_search0")

    # ---------- T3-03：search_id 随每次 get_move 递增 ----------

    def test_search_id_increments_each_call(self):
        """T3-03：连续 3 次 get_move，search_id 依次为 0, 1, 2。"""
        searcher = MockSearcher()
        decider = self.MCTSMoveDecider(searcher, log_key_prefix="runA_game0")
        board = _make_board_2d()

        decider.get_move(board, 1)
        decider.get_move(board, 1)
        decider.get_move(board, 1)

        self.assertEqual(len(searcher.search_calls), 3)
        self.assertEqual(searcher.search_calls[0]["log_key"], "runA_game0_search0")
        self.assertEqual(searcher.search_calls[1]["log_key"], "runA_game0_search1")
        self.assertEqual(searcher.search_calls[2]["log_key"], "runA_game0_search2")

    # ---------- T3-04：reset() 重置 search_id 为 0 ----------

    def test_reset_resets_search_id_to_zero(self):
        """T3-04：调用 2 次 get_move → reset() → 再 1 次 get_move，
        第三次 log_key 应为 search0。"""
        searcher = MockSearcher()
        decider = self.MCTSMoveDecider(searcher, log_key_prefix="runA_game0")
        board = _make_board_2d()

        decider.get_move(board, 1)  # search0
        decider.get_move(board, 1)  # search1
        decider.reset()
        decider.get_move(board, 1)  # search0 (重置后)

        self.assertEqual(len(searcher.search_calls), 3)
        self.assertEqual(searcher.search_calls[2]["log_key"], "runA_game0_search0")

    # ---------- T3-05：log_key_prefix 为 None 时 search_id 仍递增但不传 log_key ----------

    def test_no_prefix_search_id_still_increments(self):
        """T3-05：不传 log_key_prefix，调用 3 次 get_move，不抛异常且始终 log_key=None。"""
        searcher = MockSearcher()
        decider = self.MCTSMoveDecider(searcher)  # 无 log_key_prefix
        board = _make_board_2d()

        decider.get_move(board, 1)
        decider.make_move(112)  # 模拟走子
        decider.get_move(board, 1)
        decider.get_move(board, 1)

        # 不应抛异常
        self.assertEqual(len(searcher.search_calls), 3)
        for call in searcher.search_calls:
            self.assertIsNone(call["log_key"],
                              "log_key_prefix=None 时所有调用 log_key 应为 None")


# ---------------------------------------------------------------------------
# T4：MCTSMoveDecider get_move() 删除 state/log_key、新增 board/current_player/strategy
# ---------------------------------------------------------------------------

class TestMCTSMoveDeciderGetMoveSignature(unittest.TestCase):
    """T4：get_move 新签名 (board, current_player, strategy)。"""

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

    # ---------- T4-01：get_move 不再接受 state 参数 ----------

    def test_get_move_rejects_state_parameter(self):
        """T4-01：调用 get_move(state=[0]*226) 应抛出 TypeError。"""
        searcher = MockSearcher()
        decider = self.MCTSMoveDecider(searcher)
        with self.assertRaises(TypeError):
            decider.get_move(state=[0] * 226)

    # ---------- T4-02：get_move 不再接受 log_key 参数 ----------

    def test_get_move_rejects_log_key_parameter(self):
        """T4-02：调用 get_move(board, player, log_key="test") 应抛出 TypeError。"""
        searcher = MockSearcher()
        decider = self.MCTSMoveDecider(searcher)
        board = _make_board_2d()
        with self.assertRaises(TypeError):
            decider.get_move(board, 1, log_key="test")

    # ---------- T4-03：strategy="argmax"（默认）— 贪婪选择 ----------

    def test_strategy_argmax_selects_highest_visit_counts(self):
        """T4-03：argmax 策略选择访问计数最大的空位。"""
        visit_counts = [0] * 225
        visit_counts[100] = 50  # 最高访问计数
        visit_counts[50] = 10
        visit_counts[200] = 5
        searcher = MockSearcher(visit_counts=visit_counts)
        decider = self.MCTSMoveDecider(searcher)
        board = _make_board_2d()

        result = decider.get_move(board, 1)  # 默认 strategy="argmax"
        x, y = result.get_position()
        action = x * 15 + y
        self.assertEqual(action, 100, f"argmax 应选最高访问计数的 action 100，实际 {action}")

        self.assertIsInstance(result, MoveDecision)
        self.assertEqual(result.get_mcts_visit_counts(), visit_counts)

    # ---------- T4-04：strategy="sampling" — 温度采样 ----------

    def test_strategy_sampling_uses_temperature(self):
        """T4-04：sampling 策略按访问计数比例采样。"""
        visit_counts = [0] * 225
        visit_counts[0] = 80
        visit_counts[1] = 15
        visit_counts[2] = 5
        searcher = MockSearcher(visit_counts=visit_counts)
        decider = self.MCTSMoveDecider(searcher, temperature=1.0)
        board = _make_board_2d()

        # 多次采样后，高频动作应更常出现
        action_counts = {}
        for _ in range(100):
            result = decider.get_move(board, 1, strategy="sampling")
            action = result.get_position()[0] * 15 + result.get_position()[1]
            action_counts[action] = action_counts.get(action, 0) + 1

        # action 0 访问计数最高，应最频繁被选中
        self.assertIn(0, action_counts)
        self.assertGreater(action_counts[0], action_counts.get(1, 0),
                           "action 0 访问计数最高，应最频繁")
        self.assertGreater(action_counts[0], action_counts.get(2, 0))

    # ---------- T4-05：strategy="sampling" 且 temperature=0 — 退化为贪婪 ----------

    def test_strategy_sampling_zero_temp_equals_greedy(self):
        """T4-05：temperature=0.0 时 sampling 等价 argmax。"""
        visit_counts = [0] * 225
        visit_counts[100] = 10
        visit_counts[50] = 5
        visit_counts[200] = 2
        searcher = MockSearcher(visit_counts=visit_counts)
        decider = self.MCTSMoveDecider(searcher, temperature=0.0)
        board = _make_board_2d()

        for _ in range(30):
            result = decider.get_move(board, 1, strategy="sampling")
            action = result.get_position()[0] * 15 + result.get_position()[1]
            self.assertEqual(action, 100, "T=0 时应等价 argmax，始终选 100")

    # ---------- T4-06：strategy 非法值 ----------

    def test_invalid_strategy_raises_value_error(self):
        """T4-06：strategy="invalid" 应抛出 ValueError。"""
        searcher = MockSearcher()
        decider = self.MCTSMoveDecider(searcher)
        board = _make_board_2d()
        with self.assertRaises(ValueError):
            decider.get_move(board, 1, strategy="invalid")

    # ---------- T4-07：make_move 后 search_id 正确递增 ----------

    def test_make_move_does_not_reset_search_id(self):
        """T4-07：make_move 不重置 search_id，仅在 reset() 时重置。"""
        searcher = MockSearcher()
        decider = self.MCTSMoveDecider(searcher, log_key_prefix="runB_game1")
        board = _make_board_2d()

        decider.get_move(board, 1)  # search0
        self.assertEqual(searcher.search_calls[0]["log_key"], "runB_game1_search0")

        decider.make_move(112)  # 走子，不重置 search_id
        decider.get_move(board, 1)  # search1
        self.assertEqual(searcher.search_calls[1]["log_key"], "runB_game1_search1")

    # ---------- T4-08：内部 board→state 转换正确 ----------

    def test_board_to_state_conversion(self):
        """T4-08：board=15×15 矩阵 + current_player=2 → searcher 收到 226 维向量。"""
        searcher = MockSearcher()
        decider = self.MCTSMoveDecider(searcher)
        # 构造非空棋盘：在 (7,7) 放黑子 player=1，(3,5) 放白子 player=2
        board = _make_board_2d()
        board[7][7] = 1
        board[3][5] = 2

        decider.get_move(board, 2)

        state = searcher.search_calls[0]["state"]
        self.assertEqual(len(state), 226, "state 应为 226 维向量")
        # 前 225 维是棋盘按行展开
        # (7,7) → 索引 7*15+7 = 112
        self.assertEqual(state[112], 1, "board[7][7]=1 应在 state[112]")
        # (3,5) → 索引 3*15+5 = 50
        self.assertEqual(state[50], 2, "board[3][5]=2 应在 state[50]")
        # 第 226 维是 current_player
        self.assertEqual(state[225], 2, "状态末位应为 current_player=2")

        # 验证其他空位为 0
        self.assertEqual(state[0], 0)


if __name__ == "__main__":
    unittest.main()
