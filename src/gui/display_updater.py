"""游戏状态渲染器

负责根据 display_state 字典刷新棋盘、标签、历史记录列表。
"""

import tkinter as tk
from tkinter import messagebox

from game import get_winner_text
from gui.components.constants import ANIMALS


class GameDisplayUpdater:
    """游戏显示更新器"""
    
    def __init__(self, board, save_name_label, status_label,
                 game_status_label, history_listbox, helper,
                 color_text, color_bear):
        """初始化
        
        Args:
            board: GameBoard 实例
            save_name_label: 存档名称标签
            status_label: 当前玩家状态标签
            game_status_label: 游戏状态标签
            history_listbox: 历史记录列表框
            helper: GUIHelper 实例
            color_text: 文本颜色
            color_bear: 主题色（熊/黑方颜色）
        """
        self.board = board
        self.save_name_label = save_name_label
        self.status_label = status_label
        self.game_status_label = game_status_label
        self.history_listbox = history_listbox
        self.helper = helper
        self.COLOR_TEXT = color_text
        self.COLOR_BEAR = color_bear
    
    def update(self, display_state: dict):
        """根据显示状态更新界面
        
        Args:
            display_state: 由 helper.get_display_state() 返回的字典
        """
        # 重绘棋盘
        self.board.draw_board()
        
        if not display_state["has_game"]:
            self.save_name_label.config(text="🌲 当前存档: 无")
            self.status_label.config(text="🎯 当前玩家: -")
            self.game_status_label.config(text="🌳 游戏状态: 未开始", fg="gray")
            self.history_listbox.delete(0, tk.END)
            return
        
        # 更新存档名称
        self.save_name_label.config(text=f"🌲 当前存档: {display_state['save_name']}")
        
        # 绘制棋子
        self.board.draw_pieces(display_state["board"], display_state["last_move"])
        
        # 更新玩家信息（动物主题）
        winner = display_state["winner"]
        current_player = display_state["current_player"]
        
        if winner == 0:
            player_name = self.helper.get_player_name(current_player)
            status_text = f"🌲 {player_name.split()[0]}的回合"
            self.status_label.config(text=f"🎯 当前玩家: {player_name}", fg=self.COLOR_TEXT)
            self.game_status_label.config(text=status_text, fg=self.COLOR_BEAR)
        elif winner == 3:
            self.status_label.config(text="🎯 当前玩家: -")
            self.game_status_label.config(text=f"🌳 游戏状态: {get_winner_text(winner)}", fg="#8B4513")
        else:
            winner_name = self.helper.get_player_name(winner, winner=winner)
            win_emoji = ANIMALS["black"] if winner == 1 else ANIMALS["white"]
            self.status_label.config(text=f"🏆 获胜方: {winner_name}", fg=self.COLOR_BEAR)
            self.game_status_label.config(text=f"{win_emoji} {winner_name.split()[0]}获胜！", fg=self.COLOR_BEAR)
        
        # 更新历史记录
        self.update_history(display_state["history"])
    
    def update_history(self, history: list):
        """更新历史记录显示
        
        Args:
            history: 历史记录列表
        """
        self.history_listbox.delete(0, tk.END)
        
        for i, move in enumerate(history):
            player_name = "黑" if move["player"] == 1 else "白"
            self.history_listbox.insert(
                tk.END,
                f"{i+1:3d}. {player_name}: ({move['x']}, {move['y']})"
            )
        
        # 滚动到最后
        if history:
            self.history_listbox.see(tk.END)
    
    def show_game_over(self, winner):
        """显示游戏结束弹窗
        
        Args:
            winner: 获胜方 (1=黑胜, 2=白胜, 3=平局)
        """
        winner_text = get_winner_text(winner)
        messagebox.showinfo("游戏结束", f"{winner_text}！")
