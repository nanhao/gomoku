"""is_sample_game 函数单元测试

覆盖设计文档 test_cases.md B-01 ~ B-10：
判定单局对弈结果是否为样本局。
逻辑：winner ∈ {1,2} 且 should_keep_game(result) 为 True。
"""

import unittest
from unittest.mock import patch

from ai.create_samples import is_sample_game


class TestIsSampleGameNoTemp(unittest.TestCase):
    """无温度采样场景 (temp_side_this_game=None)"""

    def test_black_win_no_temp(self):
        """B-01: 黑胜+无温度采样 → True"""
        result = {"winner": 1, "temp_side_this_game": None}
        self.assertTrue(is_sample_game(result))

    def test_white_win_no_temp(self):
        """B-02: 白胜+无温度采样 → True"""
        result = {"winner": 2, "temp_side_this_game": None}
        self.assertTrue(is_sample_game(result))

    def test_draw_no_temp(self):
        """B-03: 平局+无温度采样 → False"""
        result = {"winner": 3, "temp_side_this_game": None}
        self.assertFalse(is_sample_game(result))

    def test_unfinished_no_temp(self):
        """B-04: 未结束+无温度采样 → False"""
        result = {"winner": 0, "temp_side_this_game": None}
        self.assertFalse(is_sample_game(result))


class TestIsSampleGameSingleSide(unittest.TestCase):
    """单方采样场景"""

    def test_sample_side_black_wins(self):
        """B-05: 采样方黑胜+temp_side=black → True"""
        result = {"winner": 1, "temp_side_this_game": "black"}
        self.assertTrue(is_sample_game(result))

    def test_greedy_side_white_wins(self):
        """B-06: 贪婪方白胜+temp_side=black → False"""
        result = {"winner": 2, "temp_side_this_game": "black"}
        self.assertFalse(is_sample_game(result))

    def test_sample_side_white_wins(self):
        """B-07: 采样方白胜+temp_side=white → True"""
        result = {"winner": 2, "temp_side_this_game": "white"}
        self.assertTrue(is_sample_game(result))

    def test_greedy_side_black_wins(self):
        """B-08: 贪婪方黑胜+temp_side=white → False"""
        result = {"winner": 1, "temp_side_this_game": "white"}
        self.assertFalse(is_sample_game(result))


class TestIsSampleGameBothMode(unittest.TestCase):
    """both 模式场景"""

    def test_both_mode_black_wins(self):
        """B-09: both模式+黑胜 → True"""
        result = {"winner": 1, "temp_side_this_game": "both"}
        self.assertTrue(is_sample_game(result))

    def test_both_mode_draw(self):
        """B-10: both模式+平局 → False"""
        result = {"winner": 3, "temp_side_this_game": "both"}
        self.assertFalse(is_sample_game(result))


if __name__ == "__main__":
    unittest.main()
