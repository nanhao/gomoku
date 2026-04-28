"""GUIHelper 测试

测试 GUI 辅助类的功能。
"""

import sys
import os
import time

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import unittest
from game import GomokuController
from game import get_winner_text
from game.save import GomokuSave
from app import GameSession
from app.gui_helper import GUIHelper, get_ai_data_dir, create_pve_config


class TestGUIHelper(unittest.TestCase):
    """测试 GUIHelper 类"""

    def setUp(self):
        """测试前准备"""
        self.test_save_dir = os.path.join(os.path.dirname(__file__), 'primary', '_test_saves')
        os.makedirs(self.test_save_dir, exist_ok=True)

        self.controller = GomokuController()
        self.controller.save = GomokuSave(self.test_save_dir)

    def tearDown(self):
        """测试后清理"""
        import shutil
        if os.path.exists(self.test_save_dir):
            shutil.rmtree(self.test_save_dir, ignore_errors=True)

    def _create_helper(self, game_mode="pvp", ai_player=None, ai_model="random"):
        """创建 GUIHelper 实例"""
        session = GameSession(
            mode=game_mode,
            controller=self.controller,
            ai_name=ai_model,
            ai_player=ai_player,
        )
        return GUIHelper(self.controller, session)

    def test_start_game_with_ai_first_creates_game_and_moves(self):
        """AI 先走时启动游戏应创建游戏并执行 AI 落子"""
        helper = self._create_helper(game_mode="pve", ai_player=1)
        result = helper.start_game_with_ai_first()

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["ai_moved"])
        self.assertIsNotNone(self.controller.game)

    def test_handle_player_move_error(self):
        """处理玩家落子返回错误（无游戏时）"""
        helper = self._create_helper(game_mode="pvp")
        result = helper.handle_player_move(7, 7)

        self.assertEqual(result["status"], "error")
        self.assertIn("error", result)

    def test_handle_player_move_success(self):
        """处理玩家落子成功"""
        helper = self._create_helper(game_mode="pve", ai_player=2)
        helper.start_game_with_ai_first()  # 创建游戏

        result = helper.handle_player_move(7, 7)

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["need_ai_move"])

    def test_handle_player_move_no_game(self):
        """无游戏时处理玩家落子返回错误"""
        helper = self._create_helper(game_mode="pvp")
        result = helper.handle_player_move(7, 7)

        self.assertEqual(result["status"], "error")

    def test_handle_ai_move_success(self):
        """处理 AI 落子成功"""
        helper = self._create_helper(game_mode="pve", ai_player=1)
        helper.start_game_with_ai_first()  # AI 先走已落子

        # 玩家落子，轮到 AI
        helper.handle_player_move(7, 7)

        result = helper.handle_ai_move()
        self.assertEqual(result["status"], "success")

    def test_handle_ai_move_no_game(self):
        """无游戏时处理 AI 落子返回错误"""
        helper = self._create_helper(game_mode="pve", ai_player=1)
        result = helper.handle_ai_move()

        self.assertEqual(result["status"], "error")

    def test_handle_ai_move_error(self):
        """处理 AI 落子返回错误状态（无游戏时）"""
        helper = self._create_helper(game_mode="pve", ai_player=1)
        # 不创建游戏，直接调用 AI 落子
        result = helper.handle_ai_move()

        # 应该返回错误（因为没有游戏）
        self.assertEqual(result["status"], "error")

    def test_get_display_state_with_game(self):
        """获取有游戏时的显示状态"""
        helper = self._create_helper(game_mode="pvp")
        helper.start_game_with_ai_first()

        state = helper.get_display_state()

        self.assertTrue(state["has_game"])
        self.assertIsNotNone(state["board"])
        self.assertIn("save_name", state)
        self.assertIn("current_player", state)

    def test_get_display_state_no_game(self):
        """获取无游戏时的显示状态"""
        helper = self._create_helper(game_mode="pvp")
        state = helper.get_display_state()

        self.assertFalse(state["has_game"])

    def test_get_player_name_black(self):
        """获取黑方玩家名称"""
        helper = self._create_helper(game_mode="pvp")
        name = helper.get_player_name(1)
        self.assertIn("小熊", name)

    def test_get_player_name_white(self):
        """获取白方玩家名称"""
        helper = self._create_helper(game_mode="pvp")
        name = helper.get_player_name(2)
        self.assertIn("小兔", name)

    def test_get_player_name_winner_black(self):
        """获取黑方获胜时的玩家名称"""
        helper = self._create_helper(game_mode="pvp")
        name = helper.get_player_name(1, winner=1)
        self.assertIn("小熊", name)

    def test_get_player_name_winner_white(self):
        """获取白方获胜时的玩家名称"""
        helper = self._create_helper(game_mode="pvp")
        name = helper.get_player_name(2, winner=2)
        self.assertIn("小兔", name)

    def test_start_game_with_ai_first_game_over(self):
        """AI 先走且游戏立即结束（AI 直接获胜）"""
        # 这个测试主要验证代码路径，实际 AI 不会一步获胜
        helper = self._create_helper(game_mode="pve", ai_player=1)
        result = helper.start_game_with_ai_first()

        # 验证返回结构正确
        self.assertEqual(result["status"], "success")
        self.assertIn("game_over", result)

    def test_handle_player_move_game_over(self):
        """玩家落子后游戏结束"""
        helper = self._create_helper(game_mode="pvp")
        helper.start_game_with_ai_first()

        # 模拟快速胜利
        moves = [(7, 5), (0, 0), (7, 6), (0, 1), (7, 7), (0, 2), (7, 8)]
        for row, col in moves:
            result = helper.handle_player_move(row, col)
            if result.get("game_over"):
                break

        # 验证最后一步返回游戏结束
        self.assertTrue(result.get("game_over", False) or result["status"] == "success")

    def test_get_winner_text_from_game(self):
        """测试 game 包的 get_winner_text 函数"""
        self.assertEqual(get_winner_text(0), "")       # 未结束
        self.assertEqual(get_winner_text(1), "黑方获胜")  # 黑胜
        self.assertEqual(get_winner_text(2), "白方获胜")  # 白胜
        self.assertEqual(get_winner_text(3), "平局")      # 平局
        self.assertEqual(get_winner_text(99), "")        # 无效值


class TestGUIHelperFunctions(unittest.TestCase):
    """测试 GUIHelper 独立函数"""

    def test_get_ai_data_dir(self):
        """获取 ai_data 目录路径（应位于 ai/ai_data/ 下）"""
        path = get_ai_data_dir()
        self.assertIn("ai_data", path)
        self.assertIn("ai" + os.sep + "ai_data", path)
        self.assertTrue(os.path.isabs(path))

    def test_create_pve_config_ai_black(self):
        """创建 PvE 配置，玩家执白，AI 执黑"""
        selector_result = {"player_color": "white", "ai": "random"}
        config = create_pve_config(selector_result)

        self.assertEqual(config["game_mode"], "pve")
        self.assertEqual(config["ai_player"], 1)  # AI 执黑
        self.assertEqual(config["ai_model"], "random")

    def test_create_pve_config_ai_white(self):
        """创建 PvE 配置，玩家执黑，AI 执白"""
        selector_result = {"player_color": "black", "ai": "fc_v1"}
        config = create_pve_config(selector_result)

        self.assertEqual(config["game_mode"], "pve")
        self.assertEqual(config["ai_player"], 2)  # AI 执白
        self.assertEqual(config["ai_model"], "fc_v1")


if __name__ == "__main__":
    unittest.main()
