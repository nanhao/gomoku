"""Random 模型测试 - 接口契约验证"""
import unittest
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.models.random.model import RandomModel


class TestRandomModelContract(unittest.TestCase):
    """Random 模型接口契约测试"""

    def setUp(self):
        self.model = RandomModel()

    def test_predict_prob_sum_normal_board(self):
        """正常棋盘：概率之和应为 1"""
        board_vector = [0] * 225 + [1]  # 空棋盘，黑方回合
        result = self.model.predict(board_vector)
        self.assertAlmostEqual(sum(result), 1.0, places=5)

    def test_predict_prob_sum_full_board(self):
        """满棋盘：概率之和应为 1（修复空棋盘返回全0的bug）"""
        board_vector = [1] * 225 + [1]  # 满棋盘（全黑子）
        result = self.model.predict(board_vector)
        # 修复后空棋盘也返回均匀分布
        self.assertAlmostEqual(sum(result), 1.0, places=5,
                               msg="满棋盘时概率之和也应为1")

    def test_predict_full_board_uniform(self):
        """满棋盘：所有位置概率应相等"""
        board_vector = [1] * 225 + [1]
        result = self.model.predict(board_vector)
        expected_prob = 1.0 / 225
        for prob in result:
            self.assertAlmostEqual(prob, expected_prob, places=5)

    def test_predict_empty_positions_zero_prob(self):
        """已落子位置概率应为 0"""
        board = [0] * 225
        board[7 * 15 + 7] = 1  # (7,7) 有黑子
        board_vector = board + [1]
        result = self.model.predict(board_vector)
        
        # (7,7) 位置概率为 0
        self.assertEqual(result[7 * 15 + 7], 0.0)
        # 其他空位概率大于 0
        self.assertGreater(result[0], 0.0)

    def test_name_property(self):
        """模型名称应为 random"""
        self.assertEqual(self.model.name, "random")


if __name__ == '__main__':
    unittest.main()
