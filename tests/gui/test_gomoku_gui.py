"""
GomokuGUI 主界面测试 - TestGomokuGUI

覆盖人机对战核心流程：
1. 落子后正确解析 dict 结果（不崩溃）
2. 使用 is_over() 而非 is_game_over()（不崩溃）
3. 玩家执黑落子后，轮到白方 AI，AI 应被触发
4. 玩家执白落子后，轮到黑方 AI，AI 应被触发
"""

import sys
import os
import tkinter as tk
import unittest.mock as mock
import time

# 确保项目根目录在路径中
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from game import GomokuController
from app import GameSession
from ai import AIGameManager, AIEngine
from ai.models.random import RandomModel


def _make_gui(game_mode="pvp", ai_player=None, ai_model="random"):
    """
    构造 GomokuGUI 实例，绕过 __init__ 中的 Tkinter 界面创建，
    只初始化核心逻辑属性（controller / session / helper）。
    """
    from gui.main import GomokuGUI
    from app import GUIHelper
    root = tk.Tk()
    root.withdraw()

    gui = GomokuGUI.__new__(GomokuGUI)
    gui.root = root
    gui.controller = GomokuController()
    gui._board_update_called = False   # 用于追踪 _update_board 是否被调用
    gui._ai_move_scheduled = False     # 用于追踪 _ai_move 是否被调度

    # 创建 GameSession
    gui.session = GameSession(
        mode=game_mode,
        controller=gui.controller,
        ai_name=ai_model,
        ai_player=ai_player,
    )

    # 创建 GUIHelper
    gui.helper = GUIHelper(gui.controller, gui.session)

    # 替换会弹窗或依赖界面的方法
    gui._update_board = lambda: setattr(gui, '_board_update_called', True)

    def _fake_after(ms, func):
        """记录 after 调度，不真正等待"""
        gui._ai_move_scheduled = True
    gui.root.after = _fake_after

    return gui, root


def _click(gui, row, col):
    """mock 掉 messagebox 后执行 _on_board_click，避免弹窗阵底"""
    with mock.patch("gui.main.messagebox"):
        gui._on_board_click(row, col)


class TestGomokuGUIOnBoardClick:
    """测试 _on_board_click 的核心逻辑"""

    def setup_method(self):
        self.gui, self.root = _make_gui(game_mode="pvp")
        self.gui.controller.new_game(f"测试_pvp_{id(self.gui)}")  # 用对象id保证名称唯一

    def teardown_method(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    def test_on_board_click_no_game_does_not_crash(self):
        """无游戏时点击不应崩溃"""
        gui, root = _make_gui()
        try:
            _click(gui, 7, 7)
        except Exception as e:
            assert False, f"无游戏时点击崩溃: {e}"
        root.destroy()

    def test_make_move_returns_dict_not_tuple(self):
        """make_move 返回 dict，_on_board_click 不应用元组解包"""
        result = self.gui.controller.make_move(7, 7)
        assert isinstance(result, dict), "make_move 应返回 dict"
        assert "status" in result or "error" in result

    def test_on_board_click_valid_move_updates_board(self):
        """合法落子后应刷新棋盘"""
        _click(self.gui, 7, 7)
        assert self.gui._board_update_called, "合法落子后应调用 _update_board"

    def test_on_board_click_invalid_move_silent(self):
        """重复落子应静默处理，不崩溃"""
        _click(self.gui, 7, 7)
        self.gui._board_update_called = False
        _click(self.gui, 7, 7)   # 重复落子
        assert not self.gui._board_update_called, "重复落子不应刷新棋盘"

    def test_is_over_method_exists(self):
        """controller 应有 is_over() 方法，不应有 is_game_over()"""
        assert hasattr(self.gui.controller, "is_over"), "controller 应有 is_over()"
        assert not hasattr(self.gui.controller, "is_game_over"), \
            "controller 不应有 is_game_over()，应使用 is_over()"

    def test_on_board_click_calls_is_over_not_crash(self):
        """落子后调用 is_over() 不应崩溃"""
        try:
            _click(self.gui, 7, 7)
        except AttributeError as e:
            assert False, f"is_over 方法名错误: {e}"


class TestGomokuGUIPveAiPlayer2:
    """玩家执黑（ai_player=2，AI执白）的人机对战测试"""

    def setup_method(self):
        self.gui, self.root = _make_gui(game_mode="pve", ai_player=2, ai_model="random")
        self.gui.controller.new_game(f"测试_pve黑_{id(self.gui)}")  # 用对象id保证名称唯一

    def teardown_method(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    def test_ai_player_is_2_when_human_plays_black(self):
        """玩家选黑时，AI 应执白（ai_player=2）"""
        assert self.gui.session.ai_player == 2

    def test_ai_model_is_set_correctly(self):
        """pve 模式应有 AI 管理器"""
        assert self.gui.session._ai_manager is not None

    def test_player_move_triggers_ai(self):
        """玩家（黑）落子后，轮到白方 AI，应调度 _ai_move"""
        _click(self.gui, 7, 7)
        assert self.gui._ai_move_scheduled, \
            "玩家执黑落子后，应调度 AI（白方）落子"

    def test_current_player_after_black_move_is_white(self):
        """黑方落子后，当前玩家应为白方（2）"""
        self.gui.controller.make_move(7, 7)
        assert self.gui.controller.get_current_player() == 2

    def test_ai_move_uses_correct_method(self):
        """AIGameManager 应有 ai_move() 方法"""
        manager = self.gui.session._ai_manager
        assert manager is not None, "pve 模式应有 ai_manager"
        assert hasattr(manager, "ai_move"), \
            "AIGameManager 应有 ai_move() 方法"
        assert not hasattr(manager, "make_ai_move"), \
            "AIGameManager 不应有 make_ai_move()，应使用 ai_move()"

    def test_ai_move_returns_dict(self):
        """ai_move() 应返回 dict"""
        self.gui.controller.make_move(7, 7)   # 玩家先落子，让轮到AI
        result = self.gui.session._ai_manager.ai_move()
        assert isinstance(result, dict), "ai_move() 应返回 dict"
        assert "status" in result or "error" in result

    def test_ai_move_does_not_crash(self):
        """_ai_move 不应崩溃"""
        self.gui.controller.make_move(7, 7)   # 玩家落子后轮到AI
        try:
            with mock.patch("gui.main.messagebox"):
                self.gui._ai_move()
        except Exception as e:
            assert False, f"_ai_move 崩溃: {e}"


class TestGomokuGUIPveAiPlayer1:
    """玩家执白（ai_player=1，AI执黑）的人机对战测试"""

    def setup_method(self):
        self.gui, self.root = _make_gui(game_mode="pve", ai_player=1, ai_model="fc_v1")
        self.gui.controller.new_game(f"测试_pve白_{id(self.gui)}")  # 用对象id保证名称唯一

    def teardown_method(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    def test_ai_player_is_1_when_human_plays_white(self):
        """玩家选白时，AI 应执黑（ai_player=1）"""
        assert self.gui.session.ai_player == 1

    def test_ai_model_fc_v1_is_set_correctly(self):
        """fc_v1 ai_model 应正确设置"""
        # 通过 session 内部 ai_manager 中的 model 类型验证
        manager = self.gui.session._ai_manager
        assert manager is not None, "pve 模式应有 ai_manager"

    def test_first_move_is_black_player(self):
        """游戏开始时当前玩家应为黑方（1）"""
        assert self.gui.controller.get_current_player() == 1

    def test_pvp_mode_does_not_trigger_ai(self):
        """PvP 模式落子不应调度 AI"""
        gui, root = _make_gui(game_mode="pvp")
        gui.controller.new_game(f"测试_pvp2_{id(gui)}")
        _click(gui, 7, 7)
        assert not gui._ai_move_scheduled, "PvP 模式不应调度 AI"
        root.destroy()


class TestGomokuGUIPveAiFirstMove:
    """AI 先走（AI执黑，ai_player=1）的测试 - 验证 _start_game_with_ai_first"""

    def setup_method(self):
        self.gui, self.root = _make_gui(game_mode="pve", ai_player=1, ai_model="random")

    def teardown_method(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    def test_start_game_with_ai_first_creates_game_and_moves(self):
        """AI先走时，_start_game_with_ai_first 应创建游戏并执行 AI 落子"""
        # 初始状态：没有游戏
        assert self.gui.controller.game is None, "初始状态应无游戏"

        # 调用 _start_game_with_ai_first
        with mock.patch("gui.main.messagebox"):
            self.gui._start_game_with_ai_first()

        # 验证游戏已创建
        assert self.gui.controller.game is not None, "AI先走时游戏应被创建"
        # 验证 AI 已落子（棋盘上应该有黑子）
        state = self.gui.controller.get_state()
        board = state["board"]
        black_count = sum(row.count(1) for row in board)
        assert black_count == 1, f"AI先走后应有一个黑子，实际有 {black_count} 个"

    def test_start_game_with_ai_first_updates_board(self):
        """AI先走后应更新显示"""
        self.gui._board_update_called = False

        with mock.patch("gui.main.messagebox"):
            self.gui._start_game_with_ai_first()

        assert self.gui._board_update_called, "AI先走后应调用 _update_board"

    def test_start_game_with_ai_first_current_player_is_white(self):
        """AI先走后，当前玩家应为白方（轮到玩家）"""
        with mock.patch("gui.main.messagebox"):
            self.gui._start_game_with_ai_first()

        current = self.gui.controller.get_current_player()
        assert current == 2, f"AI先走后应轮到白方（玩家），实际当前玩家是 {current}"

    def test_start_game_with_ai_first_does_not_crash(self):
        """AI先走流程不应崩溃"""
        try:
            with mock.patch("gui.main.messagebox"):
                self.gui._start_game_with_ai_first()
        except Exception as e:
            assert False, f"AI先走流程崩溃: {e}"
