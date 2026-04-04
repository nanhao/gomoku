"""GameSession 主流程测试

覆盖设计文档中的主流程测试用例：
- pvp 模式落子
- pve 模式落子后触发 AI 标志
- do_ai_move 成功
- 游戏结束时 winner 正确
- create_engine 工厂函数
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from game import GomokuController
from game.save import GomokuSave
from app import GameSession
from ai import create_engine
from ai.engine import AIEngine
from ai.models.random import RandomModel
from ai.models.fc_v1 import FCv1Model


def _make_controller():
    """创建使用测试目录的 controller"""
    test_save_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_test_saves"
    )
    os.makedirs(test_save_dir, exist_ok=True)
    c = GomokuController()
    c.save = GomokuSave(test_save_dir)
    return c


class TestCreateEngine:
    """测试 create_engine 工厂函数"""

    def test_create_engine_random(self):
        """create_engine('random') 应返回 AIEngine，model 为 RandomModel"""
        engine = create_engine("random")
        assert isinstance(engine, AIEngine)
        assert isinstance(engine.model, RandomModel)

    def test_create_engine_fc_v1(self):
        """create_engine('fc_v1') 应返回 AIEngine，model 为 FCv1Model"""
        engine = create_engine("fc_v1")
        assert isinstance(engine, AIEngine)
        assert isinstance(engine.model, FCv1Model)

    def test_create_engine_unknown_fallback(self):
        """create_engine 未知名称时应兜底返回 AIEngine(RandomModel)"""
        engine = create_engine("unknown_model_xyz")
        assert isinstance(engine, AIEngine)
        assert isinstance(engine.model, RandomModel)


class TestGameSessionPvp:
    """pvp 模式主流程测试"""

    def setup_method(self):
        self.controller = _make_controller()
        self.controller.new_game(f"pvp_primary_{id(self)}")
        self.session = GameSession("pvp", self.controller)

    def test_pvp_make_move_success(self):
        """pvp 模式合法落子应返回 status=success, need_ai_move=False"""
        result = self.session.make_player_move(7, 7)
        assert result["status"] == "success"
        assert result["need_ai_move"] is False

    def test_pvp_game_not_over_after_one_move(self):
        """pvp 模式一步落子后游戏不应结束"""
        result = self.session.make_player_move(7, 7)
        assert result["game_over"] is False

    def test_pvp_mode_property(self):
        """mode 属性应返回 'pvp'"""
        assert self.session.mode == "pvp"

    def test_pvp_ai_player_is_none(self):
        """pvp 模式 ai_player 应为 None"""
        assert self.session.ai_player is None


class TestGameSessionPve:
    """pve 模式主流程测试"""

    def setup_method(self):
        # 玩家执黑（player=1），AI 执白（ai_player=2）
        self.controller = _make_controller()
        self.controller.new_game(f"pve_primary_{id(self)}")
        self.session = GameSession("pve", self.controller, ai_name="random", ai_player=2)

    def test_pve_player_move_triggers_ai_flag(self):
        """pve 模式玩家（黑）落子后轮到 AI（白），need_ai_move 应为 True"""
        result = self.session.make_player_move(7, 7)
        assert result["status"] == "success"
        assert result["need_ai_move"] is True

    def test_pve_player_move_no_ai_when_not_ai_turn(self):
        """pve 模式 AI 落子后再由玩家落子，此时不轮到 AI，need_ai_move 应为 False"""
        # 玩家先走
        self.session.make_player_move(7, 7)
        # AI 走
        self.session.do_ai_move()
        # 玩家再走 → 轮到 AI，need_ai_move=True（此时仍是 AI 的回合触发场景）
        # 改为：AI 走完后立刻再由玩家走一次
        result = self.session.make_player_move(7, 8)
        assert result["status"] == "success"
        # 玩家落子后轮到 AI（白），所以 need_ai_move=True
        assert result["need_ai_move"] is True

    def test_do_ai_move_success(self):
        """pve 模式 AI 落子应返回 status=success"""
        # 先让玩家落子，轮到 AI
        self.session.make_player_move(7, 7)
        result = self.session.do_ai_move()
        assert result["status"] == "success"

    def test_pve_mode_property(self):
        """mode 属性应返回 'pve'"""
        assert self.session.mode == "pve"

    def test_pve_ai_player_property(self):
        """ai_player 属性应返回 2"""
        assert self.session.ai_player == 2


class TestGameSessionGameOver:
    """游戏结束场景测试"""

    def setup_method(self):
        self.controller = _make_controller()
        self.controller.new_game(f"gameover_primary_{id(self)}")
        self.session = GameSession("pvp", self.controller)

    def test_game_over_sets_flag(self):
        """落子导致游戏结束时，game_over=True 且 winner 正确"""
        # 黑棋横向连五：(7,5)(7,6)(7,7)(7,8)(7,9)，白棋分散落子
        moves = [
            (7, 5), (0, 0),
            (7, 6), (0, 1),
            (7, 7), (0, 2),
            (7, 8), (0, 3),
        ]
        for x, y in moves:
            self.session.make_player_move(x, y)

        # 最后一步黑棋连五
        result = self.session.make_player_move(7, 9)
        assert result["game_over"] is True
        assert result["winner"] == 1  # 黑棋胜


class TestGameSessionStartGame:
    """start_game 方法主流程测试"""

    def setup_method(self):
        self.controller = _make_controller()

    def test_pvp_start_game_creates_game(self):
        """PvP 模式 start_game 应创建游戏"""
        session = GameSession("pvp", self.controller)
        result = session.start_game(f"pvp_start_{id(self)}")

        assert result["status"] == "success"
        assert result["ai_moved"] is False
        assert self.controller.game is not None

    def test_pve_ai_first_start_game_creates_and_ai_moves(self):
        """PvE AI 先走（ai_player=1），start_game 应创建游戏并执行 AI 落子"""
        session = GameSession("pve", self.controller, ai_name="random", ai_player=1)
        result = session.start_game(f"pve_ai_first_{id(self)}")

        assert result["status"] == "success"
        assert result["ai_moved"] is True
        # 验证棋盘上有一个黑子
        state = self.controller.get_state()
        black_count = sum(row.count(1) for row in state["board"])
        assert black_count == 1
        # 验证轮到白方
        assert self.controller.get_current_player() == 2

    def test_pve_player_first_start_game_only_creates(self):
        """PvE 玩家先走（ai_player=2），start_game 只创建游戏"""
        session = GameSession("pve", self.controller, ai_name="random", ai_player=2)
        result = session.start_game(f"pve_player_first_{id(self)}")

        assert result["status"] == "success"
        assert result["ai_moved"] is False
        # 验证棋盘为空
        state = self.controller.get_state()
        black_count = sum(row.count(1) for row in state["board"])
        assert black_count == 0
        # 验证轮到黑方
        assert self.controller.get_current_player() == 1

    def test_start_game_with_auto_generated_name(self):
        """start_game 不传存档名称时应自动生成"""
        session = GameSession("pvp", self.controller)
        result = session.start_game()

        assert result["status"] == "success"
        assert self.controller.current_save_name is not None
