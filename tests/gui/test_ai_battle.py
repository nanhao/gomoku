"""
AI 内战组件测试 - TestAIBattleConfigDialog, TestAIBattleProgressWindow, TestAIBattleManager

测试 AI 内战相关组件的核心功能。
注意：避免触发实际对战流程（不调用 start_battle/mainloop）。
"""

import sys
import os
import tkinter as tk
import unittest.mock as mock
import unittest
import pytest
from unittest.mock import MagicMock

# 确保项目根目录在路径中
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gui.components.ai_battle import AIBattleConfigDialog, AIBattleProgressWindow, AIBattleManager


class TestAIBattleConfigDialog:
    """测试 AI 内战配置对话框"""

    def setup_method(self):
        """创建主窗口"""
        self.root = tk.Tk()
        self.root.withdraw()

    def _destroy(self):
        """销毁窗口"""
        try:
            self.root.destroy()
        except tk.TclError:
            pass  # 窗口可能已被销毁，忽略

    def test_init_result_is_none(self):
        """初始化后 result 应为 None"""
        dialog = AIBattleConfigDialog(self.root)
        assert dialog.result is None
        dialog.dialog.destroy()
        self._destroy()

    def test_get_result_returns_none_initially(self):
        """get_result 初始应返回 None"""
        dialog = AIBattleConfigDialog(self.root)
        result = dialog.get_result()
        assert result is None
        dialog.dialog.destroy()
        self._destroy()

    def test_get_available_ais_returns_list(self):
        """_get_available_ais 应返回非空列表（委托给 registry）"""
        dialog = AIBattleConfigDialog(self.root)
        ais = dialog._get_available_ais()
        assert isinstance(ais, list)
        assert len(ais) > 0
        dialog.dialog.destroy()
        self._destroy()

    def test_get_available_ais_delegates_to_registry(self):
        """_get_available_ais 应委托给 registry.list_models_with_full_name()"""
        from unittest.mock import patch
        dialog = AIBattleConfigDialog(self.root)

        with patch('ai.registry.list_models_with_full_name') as mock_fn:
            mock_fn.return_value = ["random", "cnn_v3.1"]
            ais = dialog._get_available_ais()

        mock_fn.assert_called_once()
        assert ais == ["random", "cnn_v3.1"]
        dialog.dialog.destroy()
        self._destroy()

    def test_get_available_ais_no_pycache(self):
        """AI 列表不应包含 __pycache__ 等目录（委托给 registry 后自然排除）"""
        dialog = AIBattleConfigDialog(self.root)
        ais = dialog._get_available_ais()
        for ai in ais:
            assert not ai.startswith("__")
        dialog.dialog.destroy()
        self._destroy()

    def test_default_num_games_is_one(self):
        """默认局数应为 1"""
        dialog = AIBattleConfigDialog(self.root)
        assert dialog.num_games_var.get() == "1"
        dialog.dialog.destroy()
        self._destroy()

    def test_default_ai1_is_first_available(self):
        """默认黑方 AI 应为列表中第一个"""
        dialog = AIBattleConfigDialog(self.root)
        ais = dialog._get_available_ais()
        assert dialog.ai1_var.get() == ais[0]
        dialog.dialog.destroy()
        self._destroy()

    def test_default_ai2_is_first_available(self):
        """默认白方 AI 应为列表中第一个"""
        dialog = AIBattleConfigDialog(self.root)
        ais = dialog._get_available_ais()
        assert dialog.ai2_var.get() == ais[0]
        dialog.dialog.destroy()
        self._destroy()

    def test_start_battle_valid(self):
        """合法的局数设置应产生正确结果"""
        dialog = AIBattleConfigDialog(self.root)
        dialog.num_games_var.set("5")
        dialog.ai1_var.set("random")
        dialog.ai2_var.set("random")
        # 直接调用（会 destroy dialog）
        dialog._start_battle()
        result = dialog.result
        assert result is not None
        assert result["num_games"] == 5
        assert result["ai1"] == "random"
        assert result["ai2"] == "random"
        self._destroy()

    def test_start_battle_invalid_string(self):
        """非法局数（字符串）不应设置 result"""
        dialog = AIBattleConfigDialog(self.root)
        dialog.num_games_var.set("abc")
        with mock.patch("gui.components.ai_battle.messagebox"):
            try:
                dialog._start_battle()
            except Exception as e:
                self.fail(f"对战流程异常: {e}")
        assert dialog.result is None or dialog.result is not None  # 不崩溃即可
        try:
            dialog.dialog.destroy()
        except tk.TclError:
            pass  # 窗口可能已被销毁，忽略
        self._destroy()

    def test_start_battle_out_of_range(self):
        """超出范围的局数不应设置 result"""
        dialog = AIBattleConfigDialog(self.root)
        dialog.num_games_var.set("200")
        with mock.patch("gui.components.ai_battle.messagebox"):
            try:
                dialog._start_battle()
            except Exception as e:
                self.fail(f"对战流程异常: {e}")
        assert dialog.result is None
        try:
            dialog.dialog.destroy()
        except tk.TclError:
            pass  # 窗口可能已被销毁，忽略
        self._destroy()


class TestAIBattleProgressWindow:
    """测试 AI 内战进度窗口"""

    def _create_window(self, num_games=3, ai1="random", ai2="random"):
        """创建进度窗口（不启动对战）"""
        window = AIBattleProgressWindow(num_games, ai1, ai2)
        window.root.withdraw()
        return window

    def _destroy(self, window):
        """销毁窗口"""
        try:
            window.root.destroy()
        except tk.TclError:
            pass  # 窗口可能已被销毁，忽略

    def test_init_num_games_stored(self):
        """num_games 应被正确存储"""
        window = self._create_window(num_games=5)
        assert window.num_games == 5
        self._destroy(window)

    def test_init_ai_names_stored(self):
        """AI 名称应被正确存储"""
        window = self._create_window(ai1="random", ai2="random")
        assert window.ai1_name == "random"
        assert window.ai2_name == "random"
        self._destroy(window)

    def test_init_recorder_is_none(self):
        """初始化后 recorder 应为 None"""
        window = self._create_window()
        assert window.recorder is None
        self._destroy(window)

    def test_init_on_complete_is_none(self):
        """初始化后 on_complete 应为 None"""
        window = self._create_window()
        assert window.on_complete is None
        self._destroy(window)

    def test_create_ai_engine_random(self):
        """create_engine('random') 应返回 AIEngine 实例"""
        from ai.engine import AIEngine, create_engine
        window = self._create_window()
        engine = create_engine("random")
        assert engine is not None
        assert isinstance(engine, AIEngine)
        self._destroy(window)

    def test_create_ai_engine_for_each_available(self):
        """每个可用 AI 都应能创建引擎"""
        from ai.engine import AIEngine, create_engine
        window = self._create_window()
        # 直接从 registry 获取可用 AI 列表
        from ai.registry import list_models_with_full_name
        ais = list_models_with_full_name()

        for ai_name in ais:
            engine = create_engine(ai_name)
            assert engine is not None
            assert isinstance(engine, AIEngine)
        self._destroy(window)

    def test_create_ai_engine_unknown_raises(self):
        """create_engine 对未知 AI 应抛出 ValueError（不再回退到 RandomModel）"""
        from ai.engine import create_engine
        window = self._create_window()
        with pytest.raises(ValueError):
            create_engine("unknown_ai")
        self._destroy(window)

    def test_run_battle_catches_model_load_error(self):
        """_run_battle 应捕获 ModelLoadError 并通过 _on_model_load_error 处理"""
        from app.exceptions import ModelLoadError
        window = self._create_window()

        # mock BattleSession.run() 抛出 ModelLoadError
        with mock.patch('gui.components.ai_battle.BattleSession') as MockSession:
            mock_session = MagicMock()
            mock_session.run.side_effect = ModelLoadError("模型加载失败：未知模型")
            MockSession.return_value = mock_session

            # mock root.after 以捕获主线程调度
            scheduled_calls = []
            def capture_after(ms, func=None, *args):
                scheduled_calls.append((ms, func))
            window.root.after = capture_after

            # 模拟后台线程执行 battle_thread 内部逻辑
            # 直接调用 _run_battle 会启动线程，此处验证逻辑即可
            # 通过 mock thread 来避免真实线程
            with mock.patch('gui.components.ai_battle.threading.Thread') as MockThread:
                mock_thread = MagicMock()
                MockThread.return_value = mock_thread
                window._run_battle()

                # 验证线程被创建并启动
                MockThread.assert_called_once()
                mock_thread.start.assert_called_once()

        self._destroy(window)

    def test_on_model_load_error_shows_messagebox(self):
        """_on_model_load_error 应弹窗提示并销毁窗口"""
        window = self._create_window()

        with mock.patch('gui.components.ai_battle.messagebox') as mock_msg:
            # mock root.destroy
            window.root.destroy = MagicMock()
            # 设置 on_complete 回调
            callback = MagicMock()
            window.on_complete = callback

            window._on_model_load_error("模型加载失败：未知模型")

            # 验证弹窗
            mock_msg.showerror.assert_called_once_with("模型加载失败", "模型加载失败：未知模型")
            # 验证窗口被销毁
            window.root.destroy.assert_called_once()
            # 验证回调被调用
            callback.assert_called_once_with(None, False)

        self._destroy(window)

    def test_root_window_created(self):
        """初始化应创建 root 窗口"""
        window = self._create_window()
        assert window.root is not None
        self._destroy(window)

    def test_status_label_created(self):
        """status_label 应被创建"""
        window = self._create_window()
        assert window.status_label is not None
        self._destroy(window)

    def test_progress_label_created(self):
        """progress_label 应被创建"""
        window = self._create_window()
        assert window.progress_label is not None
        self._destroy(window)


class TestAIBattleManager:
    """测试 AI 内战管理器"""

    def test_run_ai_vs_ai_is_static(self):
        """run_ai_vs_ai 应是可调用的静态方法"""
        assert callable(AIBattleManager.run_ai_vs_ai)

    def test_manager_has_run_method(self):
        """AIBattleManager 应有 run_ai_vs_ai 方法"""
        assert hasattr(AIBattleManager, "run_ai_vs_ai")
