"""游戏工具函数测试 — check_winner, is_board_full

测试从 GomokuGame 提取的独立纯函数。
"""

import sys
import os
import unittest

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from game.game import check_winner, is_board_full


class TestCheckWinner(unittest.TestCase):
    """check_winner 函数测试"""

    def _board_from_positions(self, positions_black, positions_white):
        """辅助方法：从棋子位置列表创建 225 维展平棋盘"""
        board = [0] * 225
        for x, y in positions_black:
            board[x * 15 + y] = 1
        for x, y in positions_white:
            board[x * 15 + y] = 2
        return board

    def test_empty_board_no_winner(self):
        """空棋盘应返回无胜者"""
        board = [0] * 225
        result = check_winner(board, 112)  # 中心位置(7,7)
        self.assertEqual(result, 0)

    def test_horizontal_win_black(self):
        """水平五连黑胜"""
        board = self._board_from_positions(
            [(7, 3), (7, 4), (7, 5), (7, 6), (7, 7)], []
        )
        result = check_winner(board, 7 * 15 + 7)
        self.assertEqual(result, 1)

    def test_horizontal_win_white(self):
        """水平五连白胜"""
        board = self._board_from_positions(
            [], [(3, 5), (3, 6), (3, 7), (3, 8), (3, 9)]
        )
        result = check_winner(board, 3 * 15 + 7)
        self.assertEqual(result, 2)

    def test_vertical_win(self):
        """垂直五连胜"""
        board = self._board_from_positions(
            [(2, 4), (3, 4), (4, 4), (5, 4), (6, 4)], []
        )
        result = check_winner(board, 4 * 15 + 4)
        self.assertEqual(result, 1)

    def test_diagonal_main_win(self):
        """主对角线五连胜"""
        board = self._board_from_positions(
            [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7)], []
        )
        result = check_winner(board, 5 * 15 + 5)
        self.assertEqual(result, 1)

    def test_diagonal_anti_win(self):
        """副对角线五连胜"""
        board = self._board_from_positions(
            [(3, 7), (4, 6), (5, 5), (6, 4), (7, 3)], []
        )
        result = check_winner(board, 5 * 15 + 5)
        self.assertEqual(result, 1)

    def test_four_in_row_no_win(self):
        """只有四连不应判定为胜"""
        board = self._board_from_positions(
            [(7, 3), (7, 4), (7, 5), (7, 6)], []
        )
        result = check_winner(board, 7 * 15 + 6)
        self.assertEqual(result, 0)

    def test_six_in_row_win(self):
        """六连也应判定为胜"""
        board = self._board_from_positions(
            [(7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7)], []
        )
        result = check_winner(board, 7 * 15 + 4)
        self.assertEqual(result, 1)

    def test_interrupted_row_no_win(self):
        """被对方棋子打断的不应判胜"""
        board = self._board_from_positions(
            [(7, 2), (7, 3), (7, 4), (7, 5)],
            [(7, 6)]
        )
        # 黑棋从(7,2)到(7,5)只有4个，(7,6)被白棋占据，不满足五连
        result = check_winner(board, 7 * 15 + 4)
        self.assertEqual(result, 0)

    def test_empty_position_no_winner(self):
        """last_move 为空位时应返回 0"""
        board = self._board_from_positions(
            [(7, 3), (7, 4), (7, 5), (7, 6), (7, 7)], []
        )
        # 检查一个空位
        result = check_winner(board, 0)
        self.assertEqual(result, 0)

    def test_win_at_board_edge(self):
        """在棋盘边缘的五连胜"""
        board = self._board_from_positions(
            [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)], []
        )
        result = check_winner(board, 0 * 15 + 2)
        self.assertEqual(result, 1)

    def test_last_move_is_middle_of_five(self):
        """last_move 是五连中间位置的判定"""
        board = self._board_from_positions(
            [(7, 3), (7, 4), (7, 5), (7, 6), (7, 7)], []
        )
        # 中间位置(7, 5)也应检测出五连
        result = check_winner(board, 7 * 15 + 5)
        self.assertEqual(result, 1)

    def test_board_with_scattered_pieces(self):
        """散落棋子不应误判"""
        board = self._board_from_positions(
            [(0, 0), (2, 2), (4, 4), (6, 6), (8, 8)], []
        )
        result = check_winner(board, 4 * 15 + 4)
        self.assertEqual(result, 0)


class TestIsBoardFull(unittest.TestCase):
    """is_board_full 函数测试"""

    def test_empty_board_not_full(self):
        """空棋盘不应为满"""
        board = [0] * 225
        self.assertFalse(is_board_full(board))

    def test_partial_board_not_full(self):
        """部分落子棋盘不应为满"""
        board = [0] * 225
        board[0] = 1
        board[112] = 2
        self.assertFalse(is_board_full(board))

    def test_full_board(self):
        """全满棋盘"""
        board = [1] * 225  # 所有位置都有棋子
        self.assertTrue(is_board_full(board))

    def test_almost_full_board(self):
        """只差一个空位的棋盘"""
        board = [1] * 225
        board[112] = 0  # 一个空位
        self.assertFalse(is_board_full(board))

    def test_full_with_mixed_pieces(self):
        """混合棋子的满棋盘"""
        board = [1 if i % 2 == 0 else 2 for i in range(225)]
        self.assertTrue(is_board_full(board))


if __name__ == "__main__":
    unittest.main()
