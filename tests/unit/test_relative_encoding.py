"""相对视角编码测试 — board_vector_to_cnn_tensor_relative

测试相对视角编码的正确性，与绝对视角编码对比验证。
"""

import sys
import os
import unittest

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from ai.base_model import board_vector_to_cnn_tensor_relative, board_vector_to_cnn_tensor


class TestRelativeEncoding(unittest.TestCase):
    """相对视角编码测试"""

    def _make_board_vector(self, black_positions, white_positions, current_player):
        """创建 226 维 board_vector"""
        board = [0] * 225
        for x, y in black_positions:
            board[x * 15 + y] = 1
        for x, y in white_positions:
            board[x * 15 + y] = 2
        board.append(current_player)
        return board

    def test_output_shape(self):
        """输出应为 (1, 3, 15, 15)"""
        board_vector = self._make_board_vector([(7, 7)], [(0, 0)], 1)
        device = torch.device("cpu")
        tensor = board_vector_to_cnn_tensor_relative(board_vector, device)
        self.assertEqual(tensor.shape, (1, 3, 15, 15))

    def test_dtype_float32(self):
        """输出 dtype 应为 float32"""
        board_vector = self._make_board_vector([(7, 7)], [(0, 0)], 1)
        device = torch.device("cpu")
        tensor = board_vector_to_cnn_tensor_relative(board_vector, device)
        self.assertEqual(tensor.dtype, torch.float32)

    def test_my_pieces_channel_black(self):
        """当当前方为黑时，通道0应标记黑棋为1.0"""
        board_vector = self._make_board_vector([(7, 7)], [(0, 0)], 1)
        device = torch.device("cpu")
        tensor = board_vector_to_cnn_tensor_relative(board_vector, device)

        # 通道0: 我方棋子 → 黑棋位置(7,7)应为1.0
        self.assertAlmostEqual(float(tensor[0, 0, 7, 7]), 1.0, places=5)
        # 白棋位置(0,0)在通道0应为0.0
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), 0.0, places=5)

    def test_my_pieces_channel_white(self):
        """当当前方为白时，通道0应标记白棋为1.0"""
        board_vector = self._make_board_vector([(7, 7)], [(0, 0)], 2)
        device = torch.device("cpu")
        tensor = board_vector_to_cnn_tensor_relative(board_vector, device)

        # 通道0: 我方棋子 → 白棋位置(0,0)应为1.0
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), 1.0, places=5)
        # 黑棋位置(7,7)在通道0应为0.0
        self.assertAlmostEqual(float(tensor[0, 0, 7, 7]), 0.0, places=5)

    def test_opponent_pieces_channel_black(self):
        """当当前方为黑时，通道1应标记白棋为1.0"""
        board_vector = self._make_board_vector([(7, 7)], [(0, 0)], 1)
        device = torch.device("cpu")
        tensor = board_vector_to_cnn_tensor_relative(board_vector, device)

        # 通道1: 对方棋子 → 白棋位置(0,0)应为1.0
        self.assertAlmostEqual(float(tensor[0, 1, 0, 0]), 1.0, places=5)
        # 黑棋位置(7,7)在通道1应为0.0
        self.assertAlmostEqual(float(tensor[0, 1, 7, 7]), 0.0, places=5)

    def test_opponent_pieces_channel_white(self):
        """当当前方为白时，通道1应标记黑棋为1.0"""
        board_vector = self._make_board_vector([(7, 7)], [(0, 0)], 2)
        device = torch.device("cpu")
        tensor = board_vector_to_cnn_tensor_relative(board_vector, device)

        # 通道1: 对方棋子 → 黑棋位置(7,7)应为1.0
        self.assertAlmostEqual(float(tensor[0, 1, 7, 7]), 1.0, places=5)

    def test_player_color_channel_black(self):
        """当当前方为黑时，通道2应为全1.0"""
        board_vector = self._make_board_vector([], [], 1)
        device = torch.device("cpu")
        tensor = board_vector_to_cnn_tensor_relative(board_vector, device)

        # 通道2: 我方颜色 → 黑方全1.0
        self.assertAlmostEqual(float(tensor[0, 2, 0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(tensor[0, 2, 7, 7]), 1.0, places=5)

    def test_player_color_channel_white(self):
        """当当前方为白时，通道2应为全0.0"""
        board_vector = self._make_board_vector([], [], 2)
        device = torch.device("cpu")
        tensor = board_vector_to_cnn_tensor_relative(board_vector, device)

        # 通道2: 我方颜色 → 白方全0.0
        self.assertAlmostEqual(float(tensor[0, 2, 0, 0]), 0.0, places=5)

    def test_empty_positions_are_zero(self):
        """空位在所有通道中都是0.0（除通道2可能是颜色指示器）"""
        board_vector = self._make_board_vector([(0, 0)], [(14, 14)], 1)
        device = torch.device("cpu")
        tensor = board_vector_to_cnn_tensor_relative(board_vector, device)

        # 空位(7,7)：通道0和通道1都为0.0
        self.assertAlmostEqual(float(tensor[0, 0, 7, 7]), 0.0, places=5)
        self.assertAlmostEqual(float(tensor[0, 1, 7, 7]), 0.0, places=5)

    def test_vs_absolute_encoding_same_player_black(self):
        """当当前方=黑时，相对编码与绝对编码应一致（通道0=黑，通道1=白）"""
        board_vector = self._make_board_vector(
            [(3, 3), (7, 7)], [(5, 5), (10, 10)], 1
        )
        device = torch.device("cpu")
        rel = board_vector_to_cnn_tensor_relative(board_vector, device)
        abs_tensor = board_vector_to_cnn_tensor(board_vector, device)

        self.assertTrue(torch.allclose(rel, abs_tensor))

    def test_vs_absolute_encoding_flipped_player_white(self):
        """当当前方=白时，相对编码的通道0/1应与绝对编码交换"""
        board_vector = self._make_board_vector(
            [(3, 3), (7, 7)], [(5, 5), (10, 10)], 2
        )
        device = torch.device("cpu")
        rel = board_vector_to_cnn_tensor_relative(board_vector, device)
        abs_tensor = board_vector_to_cnn_tensor(board_vector, device)

        # 相对编码的通道0（我方=白）应为绝对编码的通道1（白）
        self.assertTrue(torch.allclose(rel[:, 0:1, :, :], abs_tensor[:, 1:2, :, :]))
        # 相对编码的通道1（对方=黑）应为绝对编码的通道0（黑）
        self.assertTrue(torch.allclose(rel[:, 1:2, :, :], abs_tensor[:, 0:1, :, :]))


if __name__ == "__main__":
    unittest.main()
