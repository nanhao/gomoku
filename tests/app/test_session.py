"""GameSession 边缘测试

覆盖非主流程的分支和边界条件：
- 落子在已占用位置
- 落子坐标越界
- pvp 模式调用 do_ai_move 抛异常
"""

import sys
import os

import pytest

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
        """pvp 模式调用 do_ai_move 应抛出 RuntimeError 且消息含 "pvp 模式没有 AI" """
        with pytest.raises(RuntimeError, match=r"pvp 模式没有 AI"):
            self.session.do_ai_move()

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


class TestGameSessionStartGameEdge:
    """start_game 边缘测试"""

    def setup_method(self):
        self.controller = _make_controller()

    def test_start_game_with_duplicate_name_returns_error(self):
        """存档名称已存在时应返回 error"""
        # 先创建一个存档
        self.controller.new_game("duplicate_name_test")

        # 再尝试用相同名称创建
        session = GameSession("pvp", self.controller)
        result = session.start_game("duplicate_name_test")

        assert result["status"] == "error"
        assert "error" in result

    def test_do_ai_move_when_game_not_started_raises_error(self):
        """游戏未创建时调用 do_ai_move 应返回 error"""
        session = GameSession("pve", self.controller, ai_name="random", ai_player=1)
        # 注意：不调用 start_game，直接调用 do_ai_move
        result = session.do_ai_move()
        assert result["status"] == "error"

    def test_do_ai_move_returns_winner_when_game_over(self):
        """玩家落子导致游戏结束时，应返回 winner"""
        import time
        # 创建一个即将结束的游戏局面（AI 执白）
        session = GameSession("pve", self.controller, ai_name="random", ai_player=2)
        unique_name = f"test_ai_wins_{time.time()}"
        result = session.start_game(unique_name)
        assert result["status"] == "success", f"启动游戏失败: {result.get('error', '')}"

        # 构造黑棋四连，下一步黑方获胜
        # 黑棋：(7,5)(7,6)(7,7)(7,8)  下一步 (7,9) 获胜
        moves = [
            (7, 5), (0, 0),  # 黑, 白
            (7, 6), (0, 1),  # 黑, 白
            (7, 7), (0, 2),  # 黑, 白
            (7, 8), (0, 3),  # 黑, 白
        ]
        for x, y in moves:
            self.controller.make_move(x, y)

        # 现在轮到黑方，玩家落子获胜
        result = session.make_player_move(7, 9)
        assert result["status"] == "success", f"落子失败: {result.get('error', '')}"
        assert result["game_over"] is True
        assert result["winner"] == 1  # 黑方胜

    def test_start_game_twice_returns_error(self):
        """重复调用 start_game 应返回 error（存档已存在）"""
        import time
        unique_name = f"test_double_start_{time.time()}"
        session = GameSession("pvp", self.controller)
        result1 = session.start_game(unique_name)
        assert result1["status"] == "success"

        # 再次调用应失败
        result2 = session.start_game(unique_name)
        assert result2["status"] == "error"
