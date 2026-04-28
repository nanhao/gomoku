"""GameDisplayUpdater 单元测试

测试 gui/display_updater.py 中的 GameDisplayUpdater 类：
- update: 根据 display_state 字典刷新界面
- update_history: 更新历史记录列表
- show_game_over: 显示游戏结束弹窗

测试策略：
1. 创建 tkinter root 窗口（withdraw），构建所有需要的 mock/真实组件
2. Mock helper 的 get_player_name 方法
3. Mock board 的 draw_board 和 draw_pieces 方法
4. 测试各种显示状态下的 label 更新
5. 测试历史记录更新
6. 测试游戏结束弹窗（get_winner_text 已从 game 包导入，无需 mock）
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# 确保项目根目录在路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import tkinter as tk

from gui.display_updater import GameDisplayUpdater


class TestGameDisplayUpdater(unittest.TestCase):
    """测试 GameDisplayUpdater 的各种显示更新逻辑"""

    @classmethod
    def setUpClass(cls):
        """测试类级别：只创建一次 tk.Tk()，避免 Windows 下反复创建/销毁导致 Tcl 错误"""
        cls.root = tk.Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        """测试类级别：销毁 root 窗口"""
        try:
            cls.root.destroy()
        except tk.TclError:
            pass

    def setUp(self):
        """测试前准备：创建 tkinter 组件和 mock 对象"""
        # 创建真实的 tkinter 组件（复用类级别 root）
        self.save_name_label = tk.Label(self.root)
        self.status_label = tk.Label(self.root)
        self.game_status_label = tk.Label(self.root)
        self.history_listbox = tk.Listbox(self.root)

        # Mock board（避免真实绘制）
        self.mock_board = MagicMock()

        # Mock helper
        self.mock_helper = MagicMock()
        # 配置 get_player_name 默认返回值
        self.mock_helper.get_player_name.side_effect = self._mock_get_player_name
        # get_winner_text 已从 game 包导入，无需 mock，使用真实函数

        # 颜色常量
        self.color_text = "#2C2C2C"
        self.color_bear = "#8B4513"

        # 创建被测对象
        self.updater = GameDisplayUpdater(
            board=self.mock_board,
            save_name_label=self.save_name_label,
            status_label=self.status_label,
            game_status_label=self.game_status_label,
            history_listbox=self.history_listbox,
            helper=self.mock_helper,
            color_text=self.color_text,
            color_bear=self.color_bear,
        )

    def tearDown(self):
        """测试后清理：销毁本次创建的 tkinter 组件"""
        try:
            self.save_name_label.destroy()
            self.status_label.destroy()
            self.game_status_label.destroy()
            self.history_listbox.destroy()
        except tk.TclError:
            pass

    # ==================== 辅助方法 ====================

    @staticmethod
    def _mock_get_player_name(player, winner=None):
        """模拟 get_player_name 返回值"""
        if winner:
            if winner == 1:
                return "🐻 黑方"
            else:
                return "🐰 白方"
        if player == 1:
            return "🐻 黑方"
        return "🐰 白方"

    def _make_display_state(self, has_game=True, save_name="测试存档",
                            board_state=None, last_move=None,
                            winner=0, current_player=1, history=None):
        """构建测试用的 display_state 字典"""
        if board_state is None:
            board_state = [[0] * 15 for _ in range(15)]
        if history is None:
            history = []
        return {
            "has_game": has_game,
            "save_name": save_name,
            "board": board_state,
            "last_move": last_move,
            "winner": winner,
            "current_player": current_player,
            "history": history,
        }

    # ==================== update 方法测试 ====================

    def test_update_无游戏状态(self):
        """测试 has_game=False 时，显示未开始状态"""
        state = self._make_display_state(has_game=False)
        self.updater.update(state)

        # 验证存档名称标签
        self.assertEqual(self.save_name_label.cget("text"), "🌲 当前存档: 无")
        # 验证当前玩家标签
        self.assertEqual(self.status_label.cget("text"), "🎯 当前玩家: -")
        # 验证游戏状态标签
        self.assertEqual(self.game_status_label.cget("text"), "🌳 游戏状态: 未开始")
        self.assertEqual(self.game_status_label.cget("fg"), "gray")
        # 验证历史记录被清空
        self.assertEqual(self.history_listbox.size(), 0)

    def test_update_无游戏时棋盘不绘制棋子(self):
        """测试 has_game=False 时，不调用 draw_pieces"""
        state = self._make_display_state(has_game=False)
        self.updater.update(state)

        # draw_board 应被调用
        self.mock_board.draw_board.assert_called_once()
        # draw_pieces 不应被调用
        self.mock_board.draw_pieces.assert_not_called()

    def test_update_有游戏当前玩家回合(self):
        """测试 has_game=True, winner=0 时，显示当前玩家回合"""
        state = self._make_display_state(winner=0, current_player=1)
        self.updater.update(state)

        # 验证存档名称
        self.assertIn("测试存档", self.save_name_label.cget("text"))
        # 验证当前玩家
        self.assertIn("黑方", self.status_label.cget("text"))
        # 验证游戏状态显示回合信息
        self.assertIn("回合", self.game_status_label.cget("text"))

    def test_update_有游戏白方回合(self):
        """测试 has_game=True, winner=0, current_player=2 时，显示白方回合"""
        state = self._make_display_state(winner=0, current_player=2)
        self.updater.update(state)

        # 验证当前玩家显示白方
        self.assertIn("白方", self.status_label.cget("text"))

    def test_update_平局状态(self):
        """测试 has_game=True, winner=3 时，显示平局"""
        state = self._make_display_state(winner=3)
        self.updater.update(state)

        # 验证游戏状态显示平局
        self.assertEqual(self.game_status_label.cget("text"), "🌳 游戏状态: 平局")
        self.assertEqual(self.game_status_label.cget("fg"), "#8B4513")
        # 验证当前玩家为空
        self.assertEqual(self.status_label.cget("text"), "🎯 当前玩家: -")

    def test_update_黑方获胜(self):
        """测试 has_game=True, winner=1 时，显示黑方获胜"""
        state = self._make_display_state(winner=1)
        self.updater.update(state)

        # 验证获胜方显示
        self.assertIn("获胜方", self.status_label.cget("text"))
        self.assertIn("黑方", self.status_label.cget("text"))
        # 验证游戏状态显示获胜信息（含动物 emoji）
        self.assertIn("获胜", self.game_status_label.cget("text"))
        self.assertIn("🐻", self.game_status_label.cget("text"))

    def test_update_白方获胜(self):
        """测试 has_game=True, winner=2 时，显示白方获胜"""
        state = self._make_display_state(winner=2)
        self.updater.update(state)

        # 验证获胜方显示
        self.assertIn("获胜方", self.status_label.cget("text"))
        self.assertIn("白方", self.status_label.cget("text"))
        # 验证游戏状态显示获胜信息（含动物 emoji）
        self.assertIn("获胜", self.game_status_label.cget("text"))
        self.assertIn("🐰", self.game_status_label.cget("text"))

    def test_update_有游戏时调用draw_pieces(self):
        """测试 has_game=True 时，调用 draw_board 和 draw_pieces"""
        board_state = [[0] * 15 for _ in range(15)]
        last_move = (7, 7)
        state = self._make_display_state(
            board_state=board_state, last_move=last_move
        )
        self.updater.update(state)

        # 验证 draw_board 被调用
        self.mock_board.draw_board.assert_called_once()
        # 验证 draw_pieces 被调用且参数正确
        self.mock_board.draw_pieces.assert_called_once_with(board_state, last_move)

    def test_update_有游戏时更新存档名称(self):
        """测试 has_game=True 时，存档名称被正确更新"""
        state = self._make_display_state(save_name="我的存档")
        self.updater.update(state)

        self.assertIn("我的存档", self.save_name_label.cget("text"))

    def test_update_获胜时status_label颜色(self):
        """测试获胜时 status_label 前景色为主题色"""
        state = self._make_display_state(winner=1)
        self.updater.update(state)

        self.assertEqual(self.status_label.cget("fg"), self.color_bear)

    def test_update_当前玩家回合时status_label颜色(self):
        """测试当前玩家回合时 status_label 前景色为文本色"""
        state = self._make_display_state(winner=0, current_player=1)
        self.updater.update(state)

        self.assertEqual(self.status_label.cget("fg"), self.color_text)

    # ==================== update_history 方法测试 ====================

    def test_update_history_空历史(self):
        """测试空历史记录，listbox 应为空"""
        self.updater.update_history([])

        self.assertEqual(self.history_listbox.size(), 0)

    def test_update_history_单条记录(self):
        """测试单条历史记录"""
        history = [{"player": 1, "x": 7, "y": 7}]
        self.updater.update_history(history)

        # 验证 listbox 有 1 条记录
        self.assertEqual(self.history_listbox.size(), 1)
        # 验证记录内容
        entry = self.history_listbox.get(0)
        self.assertIn("黑", entry)
        self.assertIn("7", entry)

    def test_update_history_多条记录(self):
        """测试多条历史记录"""
        history = [
            {"player": 1, "x": 7, "y": 7},
            {"player": 2, "x": 8, "y": 8},
            {"player": 1, "x": 6, "y": 6},
        ]
        self.updater.update_history(history)

        # 验证 listbox 有 3 条记录
        self.assertEqual(self.history_listbox.size(), 3)
        # 验证第一条是黑方
        self.assertIn("黑", self.history_listbox.get(0))
        # 验证第二条是白方
        self.assertIn("白", self.history_listbox.get(1))

    def test_update_history_记录序号格式(self):
        """测试历史记录的序号格式（3 位右对齐）"""
        history = [{"player": 1, "x": 0, "y": 0}]
        self.updater.update_history(history)

        entry = self.history_listbox.get(0)
        # 序号应为 "  1."（3 位右对齐）
        self.assertTrue(entry.startswith("  1."))

    def test_update_history_先清空再插入(self):
        """测试更新历史时先清空旧记录"""
        # 先插入一些旧记录
        self.history_listbox.insert(tk.END, "旧记录")
        self.assertEqual(self.history_listbox.size(), 1)

        # 更新为新历史
        history = [{"player": 1, "x": 7, "y": 7}]
        self.updater.update_history(history)

        # 旧记录应被清除
        self.assertEqual(self.history_listbox.size(), 1)
        self.assertIn("黑", self.history_listbox.get(0))

    # ==================== show_game_over 方法测试 ====================

    @patch('gui.display_updater.messagebox.showinfo')
    def test_show_game_over_黑方获胜(self, mock_showinfo):
        """测试黑方获胜时显示弹窗"""
        self.updater.show_game_over(1)

        # 验证 messagebox.showinfo 被调用
        mock_showinfo.assert_called_once()
        # 验证弹窗标题
        args, kwargs = mock_showinfo.call_args
        self.assertEqual(args[0], "游戏结束")
        # 验证弹窗内容包含获胜信息
        self.assertIn("黑方获胜", args[1])

    @patch('gui.display_updater.messagebox.showinfo')
    def test_show_game_over_白方获胜(self, mock_showinfo):
        """测试白方获胜时显示弹窗"""
        self.updater.show_game_over(2)

        # 验证 messagebox.showinfo 被调用
        mock_showinfo.assert_called_once()
        args, kwargs = mock_showinfo.call_args
        self.assertIn("白方获胜", args[1])

    @patch('gui.display_updater.messagebox.showinfo')
    def test_show_game_over_平局(self, mock_showinfo):
        """测试平局时显示弹窗"""
        self.updater.show_game_over(3)

        # 验证 messagebox.showinfo 被调用
        mock_showinfo.assert_called_once()
        # 验证弹窗内容包含"平局"
        args, kwargs = mock_showinfo.call_args
        self.assertIn("平局", args[1])

    @patch('gui.display_updater.messagebox.showinfo')
    def test_show_game_over_弹窗内容格式(self, mock_showinfo):
        """测试弹窗内容格式：'winner_text！'"""
        self.updater.show_game_over(1)

        args, kwargs = mock_showinfo.call_args
        # 内容应以 "！" 结尾
        self.assertTrue(args[1].endswith("！"))

    # ==================== 初始化测试 ====================

    def test_init_属性设置(self):
        """测试初始化时各属性正确设置"""
        self.assertIs(self.updater.board, self.mock_board)
        self.assertIs(self.updater.save_name_label, self.save_name_label)
        self.assertIs(self.updater.status_label, self.status_label)
        self.assertIs(self.updater.game_status_label, self.game_status_label)
        self.assertIs(self.updater.history_listbox, self.history_listbox)
        self.assertIs(self.updater.helper, self.mock_helper)
        self.assertEqual(self.updater.COLOR_TEXT, self.color_text)
        self.assertEqual(self.updater.COLOR_BEAR, self.color_bear)


if __name__ == '__main__':
    unittest.main()
