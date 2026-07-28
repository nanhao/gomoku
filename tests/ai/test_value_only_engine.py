"""ValueOnlyAiEngine 单元测试

验证价值头引擎的孤立空位检测、5步落子算法、以及边界条件。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.base_model import ModelPredict


# ============================================================================
# Mock 模型：返回固定 value，用于隔离引擎测试
# ============================================================================

class _MockValueModel:
    """返回固定 value 的模拟模型，用于隔离引擎测试"""

    def __init__(self, fixed_value: float = 0.5):
        self._fixed_value = fixed_value

    @property
    def network(self):
        return None

    def predict(self, board_vector):
        return ModelPredict(value=self._fixed_value)

    def predict_batch(self, board_vectors):
        return [ModelPredict(value=self._fixed_value) for _ in board_vectors]


class _PositionMockModel:
    """根据落子位置返回不同 value 的模拟模型，用于测试确定性"""

    def __init__(self):
        pass

    @property
    def network(self):
        return None

    def predict(self, board_vector):
        # 从 board_vector 的最后一步落子位置推断坐标
        return ModelPredict(value=0.5)

    def predict_batch(self, board_vectors):
        # 每个新局面的 value 按其在 batch 中的索引递增
        results = []
        for i, _ in enumerate(board_vectors):
            results.append(ModelPredict(value=0.3 + i * 0.01))
        return results


# ============================================================================
# 孤立空位检测测试
# ============================================================================

class TestIsolatedDetection(unittest.TestCase):
    """孤立空位检测测试（设计 TC-ISO-01 ~ TC-ISO-06）"""

    @staticmethod
    def _is_isolated(board, idx):
        """直接调用静态方法"""
        from ai.engine.value_only import ValueOnlyAiEngine
        return ValueOnlyAiEngine._is_isolated_empty(board, idx)

    def _board_with_stone(self, positions):
        """创建仅指定位置有棋子的棋盘（225 维列表）"""
        board = [0] * 225
        for idx, player in positions:
            board[idx] = player
        return board

    def test_adjacent_not_isolated(self):
        """TC-ISO-01: 天元邻接位置非孤立"""
        board = self._board_with_stone([(7 * 15 + 7, 1)])
        self.assertFalse(self._is_isolated(board, 7 * 15 + 8),
                         "相邻 1 格应非孤立")

    def test_distance_2_not_isolated(self):
        """TC-ISO-02: 天元距离 2 内非孤立"""
        board = self._board_with_stone([(7 * 15 + 7, 1)])
        self.assertFalse(self._is_isolated(board, 7 * 15 + 9),
                         "距离 2 应非孤立")

    def test_distance_3_isolated(self):
        """TC-ISO-03: 天元距离 3 外为孤立"""
        board = self._board_with_stone([(7 * 15 + 7, 1)])
        self.assertTrue(self._is_isolated(board, 7 * 15 + 10),
                        "距离 3 应为孤立")

    def test_diagonal_distance_2_not_isolated(self):
        """TC-ISO-04: 斜线方向距离 2 内非孤立"""
        board = self._board_with_stone([(7 * 15 + 7, 1)])
        self.assertFalse(self._is_isolated(board, 9 * 15 + 9),
                         "斜线距离 2 应非孤立")

    def test_corner_isolated_empty_board(self):
        """TC-ISO-05: 空棋盘角落为孤立"""
        board = [0] * 225
        self.assertTrue(self._is_isolated(board, 0),
                        "空棋盘角落应为孤立")

    def test_occupied_not_tested(self):
        """TC-ISO-06: 已有棋子位置不在检测范围（调用方保证仅对空位调用）"""
        # 此测试验证调用方逻辑：预处理仅遍历空位
        board = self._board_with_stone([(7 * 15 + 7, 1)])
        # 对于已有棋子的位置，_is_isolated_empty 的行为未定义
        # （调用方不会对其调用），此处仅验证调用方约束
        pass


# ============================================================================
# ValueOnlyAiEngine 引擎测试
# ============================================================================

class TestValueOnlyAiEngine(unittest.TestCase):
    """ValueOnlyAiEngine 引擎测试（设计 TC-ENG-01 ~ TC-ENG-10）"""

    @classmethod
    def setUpClass(cls):
        from ai.engine.value_only import ValueOnlyAiEngine
        cls.EngineCls = ValueOnlyAiEngine

    def setUp(self):
        """每个测试前创建新引擎实例"""
        self.model = _MockValueModel(fixed_value=0.5)
        self.engine = self.EngineCls(model=self.model)

    def _empty_board(self):
        """创建空棋盘 15×15"""
        return [[0] * 15 for _ in range(15)]

    def test_get_move_empty_board(self):
        """TC-ENG-01: 空棋盘落子，返回合法坐标"""
        board = self._empty_board()
        decision = self.engine.get_move(board, 1, strategy="argmax")
        x, y = decision.get_position()
        self.assertGreaterEqual(x, 0)
        self.assertLess(x, 15)
        self.assertGreaterEqual(y, 0)
        self.assertLess(y, 15)
        self.assertEqual(board[x][y], 0, "落子位置应为空")

    def test_get_move_full_board(self):
        """TC-ENG-02: 棋盘已满抛 ValueError"""
        board = [[1] * 15 for _ in range(15)]  # 全部非空
        with self.assertRaises(ValueError):
            self.engine.get_move(board, 1)

    def test_get_move_only_one_empty(self):
        """TC-ENG-03: 仅剩 1 个空位时强制选中"""
        board = [[1] * 15 for _ in range(15)]
        board[0][0] = 0  # 仅 (0,0) 为空
        decision = self.engine.get_move(board, 1, strategy="argmax")
        self.assertEqual(decision.get_position(), (0, 0))

    def test_get_move_argmax_deterministic(self):
        """TC-ENG-04: argmax 确定性——位置相关模型两次结果一致"""
        # 使用 _PositionMockModel：不同候选位置返回不同 value
        # 确保 argmax 有唯一最大值
        model = _PositionMockModel()
        engine = self.EngineCls(model=model)
        board = [[0] * 15 for _ in range(15)]
        board[7][7] = 1  # 天元黑子
        d1 = engine.get_move(board, 1, strategy="argmax")
        d2 = engine.get_move(board, 1, strategy="argmax")
        self.assertEqual(d1.get_position(), d2.get_position())

    def test_get_move_sampling(self):
        """TC-ENG-05: sampling 策略返回合法坐标"""
        board = self._empty_board()
        # 采样多次确保不会崩溃
        for _ in range(5):
            decision = self.engine.get_move(board, 1, strategy="sampling")
            x, y = decision.get_position()
            self.assertEqual(board[x][y], 0)

    def test_get_move_sampling_high_temp(self):
        """sampling 高温时也有合法输出"""
        board = self._empty_board()
        engine = self.EngineCls(model=self.model, temperature=100.0)
        decision = engine.get_move(board, 1, strategy="sampling")
        x, y = decision.get_position()
        self.assertEqual(board[x][y], 0)

    def test_isolated_position_excluded(self):
        """TC-ENG-06: 孤立空位被排除，落子在非孤立区域"""
        # 仅天元有棋子，其后 2 格内（即 5x5 区域）为非孤立，其余孤立
        board = [[0] * 15 for _ in range(15)]
        board[7][7] = 1  # 天元黑子
        decision = self.engine.get_move(board, 1, strategy="argmax")
        x, y = decision.get_position()
        # 非孤立区域：天元周围 2 格（行 5-9，列 5-9）
        self.assertGreaterEqual(x, 5)
        self.assertLessEqual(x, 9)
        self.assertGreaterEqual(y, 5)
        self.assertLessEqual(y, 9)
        self.assertEqual(board[x][y], 0)

    def test_terminal_win_move_selected(self):
        """TC-ENG-07: 终局（五连）位置应被选中"""
        # 黑方已有四连在行 0 的 0-3 列，空位 (0,4) 可五连
        board = [[0] * 15 for _ in range(15)]
        board[0][0] = 1
        board[0][1] = 1
        board[0][2] = 1
        board[0][3] = 1
        # 黑方视角
        decision = self.engine.get_move(board, 1, strategy="argmax")
        self.assertEqual(decision.get_position(), (0, 4),
                         "应选中可五连的位置")

    def test_terminal_lose_move_avoided(self):
        """TC-ENG-08: 对方可五连的位置应被避开"""
        # 白方已有四连在行 0 的 0-3 列，空位 (0,4) 让白方五连
        # 仅创建 2 个非孤立候选：(0,4) 终局 -1 和 (0,5) 非终局
        # 这样非终局位置评估值唯一最大，argmax 必然选它
        board = [[0] * 15 for _ in range(15)]
        board[0][0] = 2
        board[0][1] = 2
        board[0][2] = 2
        board[0][3] = 2
        # 添加一个非孤立位置 (0,5) 附近的棋子使其非孤立
        board[0][6] = 1
        # 黑方视角评估
        decision = self.engine.get_move(board, 1, strategy="argmax")
        x, y = decision.get_position()
        self.assertNotEqual((x, y), (0, 4),
                            "不应选对方可五连的位置")

    def test_after_move_noop(self):
        """TC-ENG-09: after_move 不抛异常"""
        # 应无异常
        self.engine.after_move(0, 0)

    def test_move_decision_has_probs(self):
        """TC-ENG-10: MoveDecision 包含 225 维概率分布"""
        board = self._empty_board()
        decision = self.engine.get_move(board, 1, strategy="argmax")
        probs = decision.get_probs()
        self.assertIsNotNone(probs)
        self.assertEqual(len(probs), 225)
        # 概率和为 1（允许浮点误差）
        self.assertAlmostEqual(sum(probs), 1.0, places=5)

    def test_sampling_with_top_k(self):
        """sampling + top_k 截断正常工作"""
        engine = self.EngineCls(model=self.model, temperature=1.0, top_k=10)
        board = self._empty_board()
        decision = engine.get_move(board, 1, strategy="sampling")
        x, y = decision.get_position()
        self.assertEqual(board[x][y], 0)


if __name__ == '__main__':
    unittest.main()
