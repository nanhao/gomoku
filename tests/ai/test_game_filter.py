"""game_filter 模块单元测试

覆盖 should_keep_game 函数的各种场景：
- 单方采样且采样方获胜 → 保留
- 单方采样且贪婪方获胜 → 丢弃
- 单方采样且平局 → 丢弃
- 双方采样 → 不过滤
- 无温度采样 → 不过滤
- 旧格式 record.json 兼容
- 未结束对局 → 丢弃
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.game_filter import should_keep_game


class TestShouldKeepGameBlack(unittest.TestCase):
    """temp_side_this_game="black" 场景"""

    def test_black_wins(self):
        """GF-01: 采样方黑胜 → 保留"""
        self.assertTrue(should_keep_game({"temp_side_this_game": "black", "winner": 1}))

    def test_white_wins(self):
        """GF-02: 贪婪方白胜 → 丢弃"""
        self.assertFalse(should_keep_game({"temp_side_this_game": "black", "winner": 2}))

    def test_draw(self):
        """GF-03: 平局 → 丢弃"""
        self.assertFalse(should_keep_game({"temp_side_this_game": "black", "winner": 3}))


class TestShouldKeepGameWhite(unittest.TestCase):
    """temp_side_this_game="white" 场景"""

    def test_white_wins(self):
        """GF-04: 采样方白胜 → 保留"""
        self.assertTrue(should_keep_game({"temp_side_this_game": "white", "winner": 2}))

    def test_black_wins(self):
        """GF-05: 贪婪方黑胜 → 丢弃"""
        self.assertFalse(should_keep_game({"temp_side_this_game": "white", "winner": 1}))

    def test_draw(self):
        """GF-06: 平局 → 丢弃"""
        self.assertFalse(should_keep_game({"temp_side_this_game": "white", "winner": 3}))


class TestShouldKeepGameNonSingleSide(unittest.TestCase):
    """非单方采样场景：不过滤"""

    def test_both_side_black_wins(self):
        """GF-07: 双方采样 + 黑胜 → 不过滤"""
        self.assertTrue(should_keep_game({"temp_side_this_game": "both", "winner": 1}))

    def test_none_no_sampling(self):
        """GF-08: 无温度采样 → 不过滤"""
        self.assertTrue(should_keep_game({"temp_side_this_game": None, "winner": 1}))

    def test_old_format_compatibility(self):
        """GF-09: 旧格式 record.json（无 temp_side_this_game 字段）→ 不过滤"""
        self.assertTrue(should_keep_game({"winner": 1}))

    def test_old_format_white_wins(self):
        """GF-11: 旧格式 + 白胜 → 不过滤"""
        self.assertTrue(should_keep_game({"winner": 2}))


class TestShouldKeepGameEdgeCases(unittest.TestCase):
    """边界条件"""

    def test_unfinished_game(self):
        """GF-10: 未结束（winner=0）+ 单方采样 → 丢弃"""
        self.assertFalse(should_keep_game({"temp_side_this_game": "black", "winner": 0}))

    def test_both_side_draw(self):
        """双方采样 + 平局 → 不过滤"""
        self.assertTrue(should_keep_game({"temp_side_this_game": "both", "winner": 3}))

    def test_none_draw(self):
        """无温度采样 + 平局 → 不过滤"""
        self.assertTrue(should_keep_game({"temp_side_this_game": None, "winner": 3}))

    def test_empty_record(self):
        """空 record → 不过滤（get() 返回默认值 None/0）"""
        self.assertTrue(should_keep_game({}))


if __name__ == "__main__":
    unittest.main()