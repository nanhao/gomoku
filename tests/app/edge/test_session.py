"""GameSession 边缘测试

覆盖非主流程的分支和边界条件：
- 落子在已占用位置
- 落子坐标越界
- pvp 模式调用 do_ai_move 抛异常
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from game import GomokuController
from game.save import GomokuSave
from app import GameSession


def _make_controller():
    test_save_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_test_saves"
    )
    os.makedirs(test_save_dir, exist_ok=True)
    c = GomokuController()
    c.save = GomokuSave(test_save_dir)
    return c


class TestGameSessionEdge:
    """GameSession 边缘测试"""

    def setup_method(self):
        self.controller = _make_controller()
        self.controller.new_game(f"edge_{id(self)}")
        self.session = GameSession("pvp", self.controller)

    def test_make_move_invalid_occupied(self):
        """落子在已占用位置应返回 status=error"""
        self.session.make_player_move(7, 7)
        result = self.session.make_player_move(7, 7)
        assert result["status"] == "error"
        assert "error" in result

    def test_make_move_out_of_bounds(self):
        """落子坐标越界应返回 status=error"""
        result = self.session.make_player_move(15, 15)
        assert result["status"] == "error"
        assert "error" in result

    def test_pvp_do_ai_move_raises_runtime_error(self):
        """pvp 模式调用 do_ai_move 应抛出 RuntimeError"""
        try:
            self.session.do_ai_move()
            assert False, "应抛出 RuntimeError"
        except RuntimeError:
            pass

    def test_make_move_no_game_started(self):
        """未创建游戏时落子应返回 error"""
        controller = _make_controller()  # 未调用 new_game
        session = GameSession("pvp", controller)
        result = session.make_player_move(7, 7)
        assert result["status"] == "error"

    def test_make_move_after_game_over_returns_error(self):
        """游戏结束后落子应返回 error"""
        # 构造黑棋连五
        moves = [
            (7, 5), (0, 0),
            (7, 6), (0, 1),
            (7, 7), (0, 2),
            (7, 8), (0, 3),
            (7, 9),  # 黑棋连五，游戏结束
        ]
        for x, y in moves:
            self.session.make_player_move(x, y)

        # 游戏已结束，再次落子应报错
        result = self.session.make_player_move(0, 4)
        assert result["status"] == "error"
