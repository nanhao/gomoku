"""GUI 主程序集成逻辑测试

测试 GomokuGUI 的菜单调用链、方法调用顺序、状态转换。
使用 mock 替换 tkinter 相关调用，只测试业务逻辑和调用链。
"""

import sys
import os
import tempfile
import shutil
import pytest
from unittest.mock import MagicMock, patch, PropertyMock, call

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from game import GomokuController
from game.save import GomokuSave
from app import GameSession, GUIHelper
from app.exceptions import ModelLoadError
from gui.display_updater import GameDisplayUpdater
from gui.main import GomokuGUI


def _make_gui_with_mock(game_mode="pvp", ai_player=None, ai_model="random"):
    """构造 GomokuGUI 实例，绕过 Tkinter 界面创建，只初始化核心逻辑属性"""
    gui = GomokuGUI.__new__(GomokuGUI)

    # 创建 mock root
    gui.root = MagicMock()
    gui.root.after = MagicMock()

    # 创建临时目录的 controller
    temp_dir = tempfile.mkdtemp(prefix="test_gui_main_")
    save_dir = os.path.join(temp_dir, "saves")
    gui.controller = GomokuController()
    gui.controller.save = GomokuSave(save_dir)
    gui._temp_dir = temp_dir

    # 创建 session
    gui.session = GameSession(
        mode=game_mode,
        controller=gui.controller,
        ai_name=ai_model,
        ai_player=ai_player,
    )

    # 创建 helper
    gui.helper = GUIHelper(gui.controller, gui.session)

    # 创建 mock 的显示更新器和界面组件
    gui._display_updater = MagicMock()
    gui.board = MagicMock()
    gui.save_name_label = MagicMock()
    gui.status_label = MagicMock()
    gui.game_status_label = MagicMock()
    gui.history_listbox = MagicMock()

    return gui


def _cleanup_gui(gui):
    """清理 GUI 测试资源"""
    if hasattr(gui, '_temp_dir'):
        shutil.rmtree(gui._temp_dir, ignore_errors=True)


class TestOnBoardClick:
    """测试 _on_board_click 的调用链和状态转换"""

    def setup_method(self):
        self.gui = _make_gui_with_mock(game_mode="pvp")

    def teardown_method(self):
        _cleanup_gui(self.gui)

    def test_no_game_shows_message(self):
        """无游戏时点击应弹出提示"""
        with patch("gui.main.messagebox") as mock_msg:
            self.gui._on_board_click(7, 7)

        mock_msg.showinfo.assert_called_once()
        assert "提示" in mock_msg.showinfo.call_args[0]

    def test_valid_move_calls_helper_then_updates(self):
        """合法落子应先调用 helper.handle_player_move，再更新显示"""
        self.gui.controller.new_game("click_test")

        with patch.object(self.gui.helper, 'handle_player_move') as mock_handle:
            mock_handle.return_value = {
                "status": "success",
                "game_over": False,
                "need_ai_move": False,
            }
            with patch("gui.main.messagebox"):
                self.gui._on_board_click(7, 7)

            # 验证 helper 被调用
            mock_handle.assert_called_once_with(7, 7)
            # 验证显示更新
            self.gui._display_updater.update.assert_called_once()

    def test_game_over_shows_game_over(self):
        """落子导致游戏结束应调用 _show_game_over"""
        self.gui.controller.new_game("click_over_test")

        with patch.object(self.gui.helper, 'handle_player_move') as mock_handle:
            mock_handle.return_value = {
                "status": "success",
                "game_over": True,
                "winner": 1,
                "need_ai_move": False,
            }
            with patch.object(self.gui, '_show_game_over') as mock_over:
                with patch("gui.main.messagebox"):
                    self.gui._on_board_click(7, 7)

                mock_over.assert_called_once_with(1)

    def test_need_ai_move_schedules_ai(self):
        """需要 AI 落子时应调度 _ai_move"""
        self.gui = _make_gui_with_mock(game_mode="pve", ai_player=2, ai_model="random")
        self.gui.controller.new_game("click_ai_test")

        with patch.object(self.gui.helper, 'handle_player_move') as mock_handle:
            mock_handle.return_value = {
                "status": "success",
                "game_over": False,
                "need_ai_move": True,
            }
            with patch("gui.main.messagebox"):
                self.gui._on_board_click(7, 7)

            # 验证 root.after 被调用以调度 AI
            self.gui.root.after.assert_called_once()
            args = self.gui.root.after.call_args[0]
            assert args[0] == 500  # 延迟 500ms

    def test_invalid_move_shows_message(self):
        """落子失败应弹出提示"""
        self.gui.controller.new_game("click_invalid_test")

        with patch.object(self.gui.helper, 'handle_player_move') as mock_handle:
            mock_handle.return_value = {
                "status": "error",
                "error": "该位置已被占用",
            }
            with patch("gui.main.messagebox") as mock_msg:
                self.gui._on_board_click(7, 7)

            mock_msg.showinfo.assert_called_once()


class TestStartGameWithAiFirst:
    """测试 _start_game_with_ai_first 流程"""

    def setup_method(self):
        self.gui = _make_gui_with_mock(game_mode="pve", ai_player=1, ai_model="random")

    def teardown_method(self):
        _cleanup_gui(self.gui)

    def test_calls_helper_start_game_with_ai_first(self):
        """应调用 helper.start_game_with_ai_first"""
        with patch.object(self.gui.helper, 'start_game_with_ai_first') as mock_start:
            mock_start.return_value = {
                "status": "success",
                "game_over": False,
            }
            with patch("gui.main.messagebox"):
                self.gui._start_game_with_ai_first()

            mock_start.assert_called_once()

    def test_updates_display_after_success(self):
        """成功后应更新显示"""
        with patch.object(self.gui.helper, 'start_game_with_ai_first') as mock_start:
            mock_start.return_value = {
                "status": "success",
                "game_over": False,
            }
            with patch("gui.main.messagebox"):
                self.gui._start_game_with_ai_first()

            self.gui._display_updater.update.assert_called_once()

    def test_error_shows_error_message(self):
        """失败时应弹出错误消息"""
        with patch.object(self.gui.helper, 'start_game_with_ai_first') as mock_start:
            mock_start.return_value = {
                "status": "error",
                "error": "启动失败",
            }
            with patch("gui.main.messagebox") as mock_msg:
                self.gui._start_game_with_ai_first()

            mock_msg.showerror.assert_called_once()

    def test_game_over_shows_result(self):
        """AI 先走导致游戏结束应显示结果"""
        with patch.object(self.gui.helper, 'start_game_with_ai_first') as mock_start:
            mock_start.return_value = {
                "status": "success",
                "game_over": True,
                "winner": 1,
            }
            with patch.object(self.gui, '_show_game_over') as mock_over:
                with patch("gui.main.messagebox"):
                    self.gui._start_game_with_ai_first()

                mock_over.assert_called_once_with(1)


class TestAiMove:
    """测试 _ai_move 流程"""

    def setup_method(self):
        self.gui = _make_gui_with_mock(game_mode="pve", ai_player=2, ai_model="random")
        self.gui.controller.new_game("ai_move_test")

    def teardown_method(self):
        _cleanup_gui(self.gui)

    def test_ai_move_success_updates_display(self):
        """AI 落子成功应更新显示"""
        with patch.object(self.gui.helper, 'handle_ai_move') as mock_ai:
            mock_ai.return_value = {
                "status": "success",
                "game_over": False,
            }
            with patch("gui.main.messagebox"):
                self.gui._ai_move()

            self.gui._display_updater.update.assert_called_once()

    def test_ai_move_game_over_shows_result(self):
        """AI 落子导致游戏结束应显示结果"""
        with patch.object(self.gui.helper, 'handle_ai_move') as mock_ai:
            mock_ai.return_value = {
                "status": "success",
                "game_over": True,
                "winner": 2,
            }
            with patch.object(self.gui, '_show_game_over') as mock_over:
                with patch("gui.main.messagebox"):
                    self.gui._ai_move()

                mock_over.assert_called_once_with(2)

    def test_ai_move_error_shows_error(self):
        """AI 落子失败应弹出错误"""
        with patch.object(self.gui.helper, 'handle_ai_move') as mock_ai:
            mock_ai.return_value = {
                "status": "error",
                "error": "AI 落子失败",
            }
            with patch("gui.main.messagebox") as mock_msg:
                self.gui._ai_move()

            mock_msg.showerror.assert_called_once()


class TestUpdateDisplay:
    """测试 _update_display 调用链"""

    def setup_method(self):
        self.gui = _make_gui_with_mock()

    def teardown_method(self):
        _cleanup_gui(self.gui)

    def test_update_display_calls_helper_and_updater(self):
        """_update_display 应先获取显示状态，再调用 updater"""
        with patch.object(self.gui.helper, 'get_display_state') as mock_get:
            mock_get.return_value = {"board": [], "status": "playing"}
            self.gui._update_display()

            mock_get.assert_called_once()
            self.gui._display_updater.update.assert_called_once_with({"board": [], "status": "playing"})

    def test_update_display_with_real_updater_drives_ui_components(self):
        """端到端：_update_display 接入真实 GameDisplayUpdater 后，

        真实 UI 组件（mock）会收到符合预期的绘制/文本指令。
        用于打破"只验证 _display_updater.update 被调用"的反模式：
        原 mock updater 即使内部完全坏掉也能通过，真实 updater 会暴露契约。
        """
        self.gui.controller.new_game("real_updater_test")
        self.gui.controller.make_move(7, 7)  # 落一手，让 history 有内容

        # 替换为真实 GameDisplayUpdater（组件层仍用 MagicMock，但 updater 真实）
        real_updater = GameDisplayUpdater(
            board=self.gui.board,
            save_name_label=self.gui.save_name_label,
            status_label=self.gui.status_label,
            game_status_label=self.gui.game_status_label,
            history_listbox=self.gui.history_listbox,
            helper=self.gui.helper,
            color_text="#000000",
            color_bear="#654321",
        )
        self.gui._display_updater = real_updater

        self.gui._update_display()

        # 端到端契约：真实 updater 应调用各个 UI 组件
        self.gui.board.draw_board.assert_called_once()
        self.gui.board.draw_pieces.assert_called_once()
        # 存档名称标签应被设置包含存档名
        save_calls = self.gui.save_name_label.config.call_args_list
        assert any("real_updater_test" in str(c) for c in save_calls), (
            f"save_name_label 应被更新为含存档名，实际调用: {save_calls}"
        )
        # 历史记录应被重置并插入 1 条（7,7）
        self.gui.history_listbox.delete.assert_called()
        insert_calls = self.gui.history_listbox.insert.call_args_list
        assert len(insert_calls) == 1, f"history_listbox.insert 应被调 1 次，实际 {len(insert_calls)}"
        assert "(7, 7)" in str(insert_calls[0])


class TestNewGame:
    """测试 _new_game 菜单调用链"""

    def setup_method(self):
        self.gui = _make_gui_with_mock()

    def teardown_method(self):
        _cleanup_gui(self.gui)

    def test_cancel_does_nothing(self):
        """用户取消新建游戏时不做任何操作"""
        with patch("gui.main.simpledialog.askstring", return_value=None):
            self.gui._new_game()

        # controller 不应有游戏
        assert self.gui.controller.game is None

    def test_create_game_updates_display(self):
        """新建游戏成功后应更新显示"""
        with patch("gui.main.simpledialog.askstring", return_value="new_game_test"):
            with patch("gui.main.messagebox"):
                with patch.object(self.gui, '_update_display') as mock_update:
                    self.gui._new_game()

                    mock_update.assert_called_once()


class TestSaveGame:
    """测试 _save_game 菜单调用链"""

    def setup_method(self):
        self.gui = _make_gui_with_mock()

    def teardown_method(self):
        _cleanup_gui(self.gui)

    def test_no_game_shows_warning(self):
        """没有游戏时保存应提示"""
        with patch("gui.main.messagebox") as mock_msg:
            self.gui._save_game()

        mock_msg.showwarning.assert_called_once()

    def test_with_game_shows_info(self):
        """有游戏时保存应提示已自动保存"""
        self.gui.controller.new_game("save_test")
        with patch("gui.main.messagebox") as mock_msg:
            self.gui._save_game()

        mock_msg.showinfo.assert_called_once()
        assert "保存" in mock_msg.showinfo.call_args[0][1] or "保存" in str(mock_msg.showinfo.call_args)


class TestModelLoadErrorHandling:
    """测试 ModelLoadError 在 GUI 层的处理"""

    def test_game_session_raises_model_load_error(self):
        """GameSession 构造函数在 pve 模式下应能抛出 ModelLoadError"""
        controller = GomokuController()
        with pytest.raises(ModelLoadError):
            GameSession(
                mode="pve",
                controller=controller,
                ai_name="unknown_model",
                ai_player=2,
            )

    def test_main_catches_model_load_error_pve(self):
        """main() 中创建 GomokuGUI 时 ModelLoadError 应被捕获并弹窗"""
        with patch("gui.main.tk.Tk") as MockTk, \
             patch("gui.main.GameModeSelector") as MockSelector, \
             patch("gui.main.messagebox") as mock_msg, \
             patch("gui.main.create_pve_config") as mock_config:

            # 模拟 Tk 窗口（selector_root 和 game root 共用此 mock）
            mock_tk_root = MagicMock()
            MockTk.return_value = mock_tk_root

            # 模拟用户选择 pve 模式
            mock_selector = MagicMock()
            mock_selector.result = {
                "mode": "pve",
                "player_color": "black",
                "ai": "unknown_model",
            }
            MockSelector.return_value = mock_selector

            mock_config.return_value = {"ai_player": 2, "ai_model": "unknown_model"}

            # GomokuGUI 抛出 ModelLoadError
            with patch("gui.main.GomokuGUI", side_effect=ModelLoadError("模型加载失败：未知模型")):
                from gui.main import main
                main()

            # 验证弹窗提示
            mock_msg.showerror.assert_called_once_with(
                "模型加载失败", "模型加载失败：未知模型"
            )

    def test_main_catches_model_load_error_ai_vs_ai(self):
        """main() 中 AI 内战模式 ModelLoadError 应被捕获并弹窗"""
        with patch("gui.main.tk.Tk") as MockTk, \
             patch("gui.main.GameModeSelector") as MockSelector, \
             patch("gui.main.run_ai_vs_ai_mode") as mock_run, \
             patch("gui.main.messagebox") as mock_msg:

            # 模拟 selector_root Tk 窗口
            MockTk.return_value = MagicMock()

            # 模拟用户选择 AI 内战模式
            mock_selector = MagicMock()
            mock_selector.result = {
                "mode": "ai_vs_ai",
                "num_games": 1,
                "ai1": "unknown_model",
                "ai2": "random",
            }
            MockSelector.return_value = mock_selector

            # run_ai_vs_ai_mode 抛出 ModelLoadError
            mock_run.side_effect = ModelLoadError("模型加载失败：未知模型")

            from gui.main import main
            main()

            # 验证弹窗提示
            mock_msg.showerror.assert_called_once_with(
                "模型加载失败", "模型加载失败：未知模型"
            )
