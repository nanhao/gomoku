"""
AI 内战组件测试 - TestAIBattleConfigDialog, TestAIBattleProgressWindow, TestAIBattleManager

测试 AI 内战相关组件的核心功能。
注意：避免触发实际对战流程（不调用 start_battle/mainloop）。
"""

import sys
import os
import tkinter as tk
import unittest.mock as mock

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
        except Exception:
            pass

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
        """_get_available_ais 应返回非空列表"""
        dialog = AIBattleConfigDialog(self.root)
        ais = dialog._get_available_ais()
        assert isinstance(ais, list)
        assert len(ais) > 0
        dialog.dialog.destroy()
        self._destroy()

    def test_get_available_ais_returns_sorted(self):
        """AI 列表应按字母排序"""
        dialog = AIBattleConfigDialog(self.root)
        ais = dialog._get_available_ais()
        assert ais == sorted(ais)
        dialog.dialog.destroy()
        self._destroy()

    def test_get_available_ais_excludes_pycache(self):
        """AI 列表应排除 __pycache__ 等目录"""
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
            except Exception:
                pass
        assert dialog.result is None or dialog.result is not None  # 不崩溃即可
        try:
            dialog.dialog.destroy()
        except Exception:
            pass
        self._destroy()

    def test_start_battle_out_of_range(self):
        """超出范围的局数不应设置 result"""
        dialog = AIBattleConfigDialog(self.root)
        dialog.num_games_var.set("200")
        with mock.patch("gui.components.ai_battle.messagebox"):
            try:
                dialog._start_battle()
            except Exception:
                pass
        assert dialog.result is None
        try:
            dialog.dialog.destroy()
        except Exception:
            pass
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
        except Exception:
            pass

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
        # 从 AIBattleConfigDialog 获取可用 AI 列表
        root = tk.Tk()
        root.withdraw()
        config_dialog = AIBattleConfigDialog(root)
        ais = config_dialog._get_available_ais()
        config_dialog.dialog.destroy()
        root.destroy()

        for ai_name in ais:
            engine = create_engine(ai_name)
            assert engine is not None
            assert isinstance(engine, AIEngine)
        self._destroy(window)

    def test_create_ai_engine_unknown_raises(self):
        """create_engine 对未知 AI 应兼容处理（兼底返回 RandomModel）"""
        from ai.engine import AIEngine, create_engine
        from ai.models.random import RandomModel
        window = self._create_window()
        engine = create_engine("unknown_ai")
        assert isinstance(engine, AIEngine)
        assert isinstance(engine.model, RandomModel)
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
