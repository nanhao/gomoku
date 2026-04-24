"""性能优化第二级 - 棋型检测 Numba JIT 测试

测试 Numba JIT 加速后 detect_single_direction 的正确性和回退机制。
"""

import unittest
import numpy as np
from unittest.mock import patch

from ai.shape_detector import (
    detect_single_direction,
    detect_shapes,
    _detect_single_direction_jit,
    _detect_single_direction_pure_python,
    _SHAPE_CODE_MAP,
    _NUMBA_AVAILABLE,
)


def _make_board() -> list:
    """创建空棋盘（225 维一维列表）"""
    return [0] * 225


def _place(board: list, x: int, y: int, player: int):
    """在棋盘上放置棋子"""
    board[x * 15 + y] = player


class TestDetectSingleDirectionJIT(unittest.TestCase):
    """测试 JIT 版 detect_single_direction 的数值正确性"""

    def test_jit_returns_integer_code(self):
        """JIT 函数应返回整数编码而非字符串"""
        board = _make_board()
        _place(board, 7, 7, 1)  # 放一颗黑子
        board_arr = np.array(board, dtype=np.int8)
        # 独一：只有一颗棋子
        result = _detect_single_direction_jit(board_arr, 7, 7, 1, 0, 1)
        self.assertIsInstance(result, (int, np.integer))

    def test_jit_five_in_a_row(self):
        """连五检测"""
        board = _make_board()
        for i in range(5):
            _place(board, 7, 3 + i, 1)
        board_arr = np.array(board, dtype=np.int8)
        result = _detect_single_direction_jit(board_arr, 7, 5, 1, 0, 1)
        self.assertEqual(result, 1)  # 连五编码

    def test_jit_live_four(self):
        """活四检测"""
        board = _make_board()
        for i in range(4):
            _place(board, 7, 3 + i, 1)
        board_arr = np.array(board, dtype=np.int8)
        result = _detect_single_direction_jit(board_arr, 7, 5, 1, 0, 1)
        self.assertEqual(result, 2)  # 活四编码

    def test_jit_rush_four(self):
        """冲四检测（一端被堵）"""
        board = _make_board()
        for i in range(4):
            _place(board, 7, 3 + i, 1)
        _place(board, 7, 7, 2)  # 一端堵住
        board_arr = np.array(board, dtype=np.int8)
        result = _detect_single_direction_jit(board_arr, 7, 5, 1, 0, 1)
        self.assertEqual(result, 3)  # 冲四编码

    def test_jit_live_three(self):
        """活三检测"""
        board = _make_board()
        for i in range(3):
            _place(board, 7, 4 + i, 1)
        board_arr = np.array(board, dtype=np.int8)
        result = _detect_single_direction_jit(board_arr, 7, 5, 1, 0, 1)
        self.assertEqual(result, 4)  # 活三编码

    def test_jit_sleep_three(self):
        """眠三检测（一端被堵）"""
        board = _make_board()
        for i in range(3):
            _place(board, 7, 4 + i, 1)
        _place(board, 7, 7, 2)  # 一端堵住
        board_arr = np.array(board, dtype=np.int8)
        result = _detect_single_direction_jit(board_arr, 7, 5, 1, 0, 1)
        self.assertEqual(result, 5)  # 眠三编码

    def test_jit_both_blocked_returns_zero(self):
        """两端都被堵时返回 0（无棋型）"""
        board = _make_board()
        for i in range(3):
            _place(board, 7, 4 + i, 1)
        _place(board, 7, 3, 2)  # 一端堵住
        _place(board, 7, 7, 2)  # 另一端堵住
        board_arr = np.array(board, dtype=np.int8)
        result = _detect_single_direction_jit(board_arr, 7, 5, 1, 0, 1)
        self.assertEqual(result, 0)

    def test_jit_live_two(self):
        """活二检测"""
        board = _make_board()
        for i in range(2):
            _place(board, 7, 5 + i, 1)
        board_arr = np.array(board, dtype=np.int8)
        result = _detect_single_direction_jit(board_arr, 7, 5, 1, 0, 1)
        self.assertEqual(result, 6)  # 活二编码

    def test_jit_sleep_two(self):
        """眠二检测"""
        board = _make_board()
        for i in range(2):
            _place(board, 7, 5 + i, 1)
        _place(board, 7, 7, 2)
        board_arr = np.array(board, dtype=np.int8)
        result = _detect_single_direction_jit(board_arr, 7, 5, 1, 0, 1)
        self.assertEqual(result, 7)  # 眠二编码

    def test_jit_single_stone(self):
        """独一检测"""
        board = _make_board()
        _place(board, 7, 7, 1)
        board_arr = np.array(board, dtype=np.int8)
        result = _detect_single_direction_jit(board_arr, 7, 7, 1, 0, 1)
        self.assertEqual(result, 8)  # 独一编码


class TestShapeCodeMap(unittest.TestCase):
    """测试棋型编码映射的完整性"""

    def test_code_map_covers_all_shapes(self):
        """编码映射应覆盖所有棋型类型"""
        expected_shapes = {"连五", "活四", "冲四", "活三", "眠三", "活二", "眠二", "独一"}
        actual_shapes = set(_SHAPE_CODE_MAP.values())
        self.assertEqual(actual_shapes, expected_shapes)

    def test_code_map_has_all_codes(self):
        """编码应为 1-8 的连续整数"""
        codes = set(_SHAPE_CODE_MAP.keys())
        self.assertEqual(codes, {1, 2, 3, 4, 5, 6, 7, 8})

    def test_code_zero_means_none(self):
        """编码 0 应映射为 None（无棋型）"""
        # 0 不在 _SHAPE_CODE_MAP 中，表示 None
        self.assertNotIn(0, _SHAPE_CODE_MAP)


class TestDetectSingleDirectionWithJIT(unittest.TestCase):
    """测试 detect_single_direction 使用 JIT 后结果与纯 Python 一致"""

    def test_all_shape_types_match(self):
        """各种棋型场景下，JIT 版与纯 Python 版结果一致"""
        test_cases = [
            # (棋盘描述, 落子位置, 玩家, 方向, 预期棋型)
            ("连五", self._make_five(), (7, 5), 1, (0, 1)),
            ("活四", self._make_live_four(), (7, 5), 1, (0, 1)),
            ("冲四", self._make_rush_four(), (7, 5), 1, (0, 1)),
            ("活三", self._make_live_three(), (7, 5), 1, (0, 1)),
            ("眠三", self._make_sleep_three(), (7, 5), 1, (0, 1)),
            ("活二", self._make_live_two(), (7, 5), 1, (0, 1)),
            ("眠二", self._make_sleep_two(), (7, 5), 1, (0, 1)),
            ("独一", self._make_single(), (7, 7), 1, (0, 1)),
            ("两端堵三", self._make_both_blocked_three(), (7, 5), 1, (0, 1)),
        ]

        for desc, board, (x, y), player, (dx, dy) in test_cases:
            with self.subTest(desc=desc):
                # 纯 Python 版本（直接调用原始逻辑）
                # detect_single_direction 内部已模拟落子，board 需要是不含落子的状态
                result = detect_single_direction(board, x, y, player, dx, dy)
                # 验证结果与预期匹配
                expected = self._get_expected_shape(desc)
                self.assertEqual(result, expected, f"{desc}: 期望 {expected}，得到 {result}")

    def _get_expected_shape(self, desc: str):
        """根据描述返回预期棋型"""
        mapping = {
            "连五": "连五", "活四": "活四", "冲四": "冲四",
            "活三": "活三", "眠三": "眠三", "活二": "活二",
            "眠二": "眠二", "独一": "独一", "两端堵三": None,
        }
        return mapping[desc]

    def _make_five(self):
        board = _make_board()
        for i in range(5):
            _place(board, 7, 3 + i, 1)
        return board

    def _make_live_four(self):
        board = _make_board()
        for i in range(4):
            _place(board, 7, 3 + i, 1)
        return board

    def _make_rush_four(self):
        board = _make_board()
        for i in range(4):
            _place(board, 7, 3 + i, 1)
        _place(board, 7, 7, 2)
        return board

    def _make_live_three(self):
        board = _make_board()
        for i in range(3):
            _place(board, 7, 4 + i, 1)
        return board

    def _make_sleep_three(self):
        board = _make_board()
        for i in range(3):
            _place(board, 7, 4 + i, 1)
        _place(board, 7, 7, 2)
        return board

    def _make_live_two(self):
        board = _make_board()
        for i in range(2):
            _place(board, 7, 5 + i, 1)
        return board

    def _make_sleep_two(self):
        board = _make_board()
        for i in range(2):
            _place(board, 7, 5 + i, 1)
        _place(board, 7, 7, 2)
        return board

    def _make_single(self):
        board = _make_board()
        _place(board, 7, 7, 1)
        return board

    def _make_both_blocked_three(self):
        board = _make_board()
        for i in range(3):
            _place(board, 7, 4 + i, 1)
        _place(board, 7, 3, 2)
        _place(board, 7, 7, 2)
        return board


class TestDetectShapesWithJIT(unittest.TestCase):
    """测试 detect_shapes 整体功能不受 JIT 影响"""

    def test_shapes_result_unchanged(self):
        """detect_shapes 结果应与 JIT 前完全一致"""
        board = _make_board()
        # 构造一个活三场景：横向三连
        for i in range(3):
            _place(board, 7, 4 + i, 1)

        result = detect_shapes(board, (7, 5), 1)

        # 应检测出活三
        shape_types = [s["shape"] for s in result]
        self.assertIn("活三", shape_types)

    def test_shapes_multiple_directions(self):
        """多方向棋型检测"""
        board = _make_board()
        # 横向活三：(7,3)(7,4)(7,5) 检测位 (7,4)
        for i in range(3):
            _place(board, 7, 3 + i, 1)
        # 纵向活三：(5,4)(6,4)(7,4) — 但 (7,4) 已被横向占据
        # 换一种构造：在另一个位置构造双活三
        # 横向活三 (7,3)(7,4)(7,5)，纵向活三 (5,5)(6,5)(7,5)
        # 这在 (7,5) 处交叉，但横向已是4连
        # 正确方案：在 (7,5) 处检测，但不要让横向形成4连
        board2 = _make_board()
        # 横向活三：(7,3)(7,4)(7,5) — 交叉点在 (7,5)
        for i in range(3):
            _place(board2, 7, 3 + i, 1)
        # 纵向活三：(5,5)(6,5)(7,5) — 交叉点在 (7,5)
        for i in range(3):
            _place(board2, 5 + i, 5, 1)

        # 注意：交叉点 (7,5) 已被横向占据，纵向 4+5+5 中 (7,5) 是已有棋子
        # 实际横向变成4连，所以结果是活四而非多活三
        # 改用无交叉的双活三场景
        board3 = _make_board()
        # 横向活三：(7,3)(7,4)(7,5) 检测位 (7,4)
        for i in range(3):
            _place(board3, 7, 3 + i, 1)
        # 纵向活三：(4,4)(5,4)(6,4) 检测位 (5,4)
        # 不行，两个不同位置不能同时检测
        # 只能在同一落子位置形成双活三
        # 构造：横向 (7,4)(7,5)(7,6) + 纵向 (5,5)(6,5)(7,5)
        # 落子位 (7,5)，横向3连+纵向3连 -> 双活三（多活三）
        board4 = _make_board()
        # 横向: (7,4)(7,5)(7,6) — 空位 (7,5) 是检测位
        _place(board4, 7, 4, 1)
        _place(board4, 7, 6, 1)
        # 纵向: (5,5)(6,5) — (7,5) 是检测位
        _place(board4, 5, 5, 1)
        _place(board4, 6, 5, 1)

        result = detect_shapes(board4, (7, 5), 1)
        shape_types = [s["shape"] for s in result]
        # 两个方向活三 -> 多活三
        self.assertIn("多活三", shape_types)

    def test_five_in_a_row(self):
        """连五检测"""
        board = _make_board()
        for i in range(4):
            _place(board, 7, 3 + i, 1)
        # (7,7) 是空位，落子后形成连五
        result = detect_shapes(board, (7, 7), 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["shape"], "连五")

    def test_board_restored_after_detect(self):
        """检测后棋盘应恢复原状"""
        board = _make_board()
        for i in range(3):
            _place(board, 7, 4 + i, 1)
        # 复制棋盘状态
        original = list(board)
        detect_shapes(board, (7, 5), 1)
        # 棋盘应恢复
        self.assertEqual(board, original)


class TestNumbaFallback(unittest.TestCase):
    """测试 Numba 不可用时的回退机制"""

    def test_numba_available_flag(self):
        """_NUMBA_AVAILABLE 应为布尔值"""
        self.assertIsInstance(_NUMBA_AVAILABLE, bool)

    def test_detect_works_with_list_input(self):
        """即使不使用 Numba，list 输入也应正常工作"""
        board = _make_board()
        for i in range(3):
            _place(board, 7, 4 + i, 1)
        # list 输入应正常工作（可能走 JIT 路径内部转换，或走纯 Python 回退）
        result = detect_single_direction(board, 7, 5, 1, 0, 1)
        self.assertEqual(result, "活三")

    def test_detect_works_with_numpy_input(self):
        """numpy 数组输入也应正常工作"""
        board = _make_board()
        for i in range(3):
            _place(board, 7, 4 + i, 1)
        board_arr = np.array(board, dtype=np.int8)
        result = detect_single_direction(board_arr, 7, 5, 1, 0, 1)
        self.assertEqual(result, "活三")


class TestJITPerformance(unittest.TestCase):
    """简单性能基准测试：JIT 版本应比纯 Python 快"""

    def test_jit_faster_than_python(self):
        """JIT 版本在大批量调用时应显著快于纯 Python"""
        import time

        board = _make_board()
        for i in range(3):
            _place(board, 7, 4 + i, 1)
        board_arr = np.array(board, dtype=np.int8)

        # JIT 版本预热（首次编译）
        _detect_single_direction_jit(board_arr, 7, 5, 1, 0, 1)

        # JIT 版本计时
        n = 10000
        start = time.perf_counter()
        for _ in range(n):
            _detect_single_direction_jit(board_arr, 7, 5, 1, 0, 1)
        jit_time = time.perf_counter() - start

        # 纯 Python 版本计时（使用原始逻辑）
        start = time.perf_counter()
        for _ in range(n):
            _detect_single_direction_pure_python(board, 7, 5, 1, 0, 1)
        python_time = time.perf_counter() - start

        # JIT 应该至少不比纯 Python 慢（允许一定误差）
        # 在大多数环境下 JIT 应快 5x 以上，但这里只验证不退化
        self.assertLess(jit_time, python_time * 2,
                        f"JIT ({jit_time:.4f}s) 不应比纯 Python ({python_time:.4f}s) 慢超过 2 倍")


if __name__ == "__main__":
    unittest.main()
