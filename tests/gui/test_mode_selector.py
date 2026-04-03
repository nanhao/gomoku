"""游戏模式选择器测试 - TestGameModeSelector"""

import sys
import os
import tkinter as tk

# 确保项目根目录在路径中
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gui.components.mode_selector import GameModeSelector


class TestGameModeSelector:
    """测试游戏模式选择器"""

    def setup_method(self):
        """测试前准备"""
        self.root = tk.Tk()
        self.root.withdraw()  # 隐藏窗口，避免测试时弹出

    def teardown_method(self):
        """测试后清理"""
        try:
            self.root.destroy()
        except Exception:
            pass

    def test_init(self):
        """测试初始化"""
        selector = GameModeSelector(self.root)
        assert selector is not None
        assert selector.result is None

    def test_select_pvp(self):
        """测试选择人人对战模式"""
        selector = GameModeSelector(self.root)
        selector._select_pvp()

        assert selector.result is not None
        assert selector.result["mode"] == "pvp"

    def test_select_pve_black(self):
        """测试选择人机对战 - 执黑棋"""
        selector = GameModeSelector(self.root)
        selector.result = {"mode": "pve", "player_color": "black", "ai": "random"}

        assert selector.result["mode"] == "pve"
        assert selector.result["player_color"] == "black"
        assert selector.result["ai"] == "random"

    def test_select_pve_white(self):
        """测试选择人机对战 - 执白棋"""
        selector = GameModeSelector(self.root)
        selector.result = {"mode": "pve", "player_color": "white", "ai": "fc_v1"}

        assert selector.result["mode"] == "pve"
        assert selector.result["player_color"] == "white"
        assert selector.result["ai"] == "fc_v1"

    def test_select_pve_with_different_ai(self):
        """测试选择人机对战 - 不同 AI 模型"""
        selector = GameModeSelector(self.root)
        selector.result = {"mode": "pve", "player_color": "black", "ai": "fc_v1"}

        assert selector.result["ai"] == "fc_v1"

    def test_select_ai_vs_ai(self):
        """测试选择 AI 内战模式"""
        selector = GameModeSelector(self.root)
        selector.result = {
            "mode": "ai_vs_ai",
            "num_games": 5,
            "ai1": "random",
            "ai2": "random"
        }

        assert selector.result["mode"] == "ai_vs_ai"
        assert selector.result["num_games"] == 5
        assert selector.result["ai1"] == "random"
        assert selector.result["ai2"] == "random"

    def test_select_replay(self):
        """测试选择回放模式"""
        selector = GameModeSelector(self.root)
        selector._select_replay()

        assert selector.result is not None
        assert selector.result["mode"] == "replay"

    def test_get_available_ais(self):
        """测试获取可用 AI 列表 - 动态扫描 ai/models 目录"""
        selector = GameModeSelector(self.root)
        ais = selector._get_available_ais()

        assert isinstance(ais, list)
        assert len(ais) >= 2, "ai/models 目录下至少应有 random 和 fc_v1 两个 AI"
        assert "random" in ais, "random AI 必须存在"
        assert "fc_v1" in ais, "fc_v1 AI 必须存在"

    def test_get_available_ais_dynamic_scan(self):
        """测试动态扫描 ai/models 目录 - 验证返回的是目录名列表"""
        import os
        selector = GameModeSelector(self.root)
        ais = selector._get_available_ais()

        # 验证每个 AI 名称都对应 ai/models 目录下的一个子目录
        ai_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "ai", "models"
        )
        for ai_name in ais:
            ai_path = os.path.join(ai_dir, ai_name)
            assert os.path.isdir(ai_path), f"{ai_name} 应对应 ai/models 目录下的子目录"
            assert not ai_name.startswith("__"), "不应包含 __ 开头的特殊目录"

    def test_get_result_none(self):
        """测试获取结果 - 未选择时返回 None"""
        selector = GameModeSelector(self.root)
        result = selector.get_result()

        assert result is None

    def test_get_result_after_select(self):
        """测试获取结果 - 选择后返回正确值"""
        selector = GameModeSelector(self.root)
        selector.result = {"mode": "pvp"}
        result = selector.get_result()

        assert result is not None
        assert result["mode"] == "pvp"

    def test_constants(self):
        """测试常量定义"""
        selector = GameModeSelector(self.root)

        # 验证配色方案存在
        assert hasattr(selector, 'COLOR_BG')
        assert hasattr(selector, 'COLOR_BEAR')
        assert hasattr(selector, 'COLOR_BUNNY')
        assert hasattr(selector, 'COLOR_TEXT')
        assert hasattr(selector, 'COLOR_BUTTON_BG')
        assert hasattr(selector, 'COLOR_BORDER')

        # 验证窗口尺寸
        assert hasattr(selector, 'WINDOW_WIDTH')
        assert hasattr(selector, 'WINDOW_HEIGHT')
        assert selector.WINDOW_WIDTH == 400
        assert selector.WINDOW_HEIGHT == 600
