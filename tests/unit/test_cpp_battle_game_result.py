"""CppBattleGameResult 数据类单元测试

测试 C++ 对战结果数据类的字段正确性和业务逻辑。
"""

import sys
import os

# 确保项目根在 sys.path 中
_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import pytest
from cpp.python.cpp_battle_game_result import CppBattleGameResult


class TestCppBattleGameResult:

    # ---- TC1.8 字段合法性（部分：数据类构造） ----

    def test_construct_black_wins_model1_black(self):
        """model1 执黑并获胜"""
        r = CppBattleGameResult(
            game_index=0,
            winner=1,
            model1_color=1,
            model2_color=2,
            model1_won=True,
            model2_won=False,
            move_count=45,
        )
        assert r.game_index == 0
        assert r.winner == 1
        assert r.model1_color == 1
        assert r.model2_color == 2
        assert r.model1_won is True
        assert r.model2_won is False
        assert r.move_count == 45

    def test_construct_white_wins_model2_white(self):
        """model2 执白并获胜"""
        r = CppBattleGameResult(
            game_index=3,
            winner=2,
            model1_color=1,
            model2_color=2,
            model1_won=False,
            model2_won=True,
            move_count=52,
        )
        assert r.game_index == 3
        assert r.winner == 2
        assert r.model2_won is True
        assert r.model1_won is False

    def test_construct_draw(self):
        """平局：双方均未获胜"""
        r = CppBattleGameResult(
            game_index=7,
            winner=0,
            model1_color=1,
            model2_color=2,
            model1_won=False,
            model2_won=False,
            move_count=225,
        )
        assert r.winner == 0
        assert r.model1_won is False
        assert r.model2_won is False
        assert r.move_count == 225

    def test_construct_model1_white_model2_black(self):
        """执方互换：model1 执白，model2 执黑"""
        r = CppBattleGameResult(
            game_index=5,
            winner=1,
            model1_color=2,
            model2_color=1,
            model1_won=False,
            model2_won=True,
            move_count=38,
        )
        assert r.model1_color == 2
        assert r.model2_color == 1
        # model2 执黑，黑胜 → model2_won
        assert r.model2_won is True

    def test_model1_color_and_model2_color_mutually_exclusive(self):
        """执方互斥：model1_color + model2_color == 3"""
        for m1c in (1, 2):
            m2c = 3 - m1c
            r = CppBattleGameResult(
                game_index=0,
                winner=0,
                model1_color=m1c,
                model2_color=m2c,
                model1_won=False,
                model2_won=False,
                move_count=10,
            )
            assert r.model1_color + r.model2_color == 3

    def test_model1_won_and_model2_won_cannot_both_be_true(self):
        """model1_won 和 model2_won 最多一个为 True（平局时均为 False）"""
        # 黑胜，model1 执黑
        r1 = CppBattleGameResult(
            game_index=0, winner=1,
            model1_color=1, model2_color=2,
            model1_won=True, model2_won=False, move_count=20,
        )
        assert not (r1.model1_won and r1.model2_won)

        # 平局
        r2 = CppBattleGameResult(
            game_index=1, winner=0,
            model1_color=1, model2_color=2,
            model1_won=False, model2_won=False, move_count=225,
        )
        assert not r2.model1_won
        assert not r2.model2_won

    def test_move_count_minimum(self):
        """五子棋最少 5 步获胜"""
        r = CppBattleGameResult(
            game_index=0,
            winner=1,
            model1_color=1,
            model2_color=2,
            model1_won=True,
            model2_won=False,
            move_count=5,
        )
        assert r.move_count == 5
