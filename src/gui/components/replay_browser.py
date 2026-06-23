"""
回放浏览器组件 - 负责选择回放轮次和对局
"""
import tkinter as tk
from tkinter import messagebox
import os
import json

from .constants import COLORS, DIMENSIONS, FONTS


class RoundSelector:
    """轮次选择器 - 选择要查看的轮次目录"""
    
    COLOR_BG = COLORS["bg"]
    COLOR_TEXT = COLORS["text"]
    COLOR_BEAR = COLORS["bear"]
    COLOR_BUTTON_BG = "#FFE4B5"
    
    def __init__(self, parent, ai_data_dir, on_select=None):
        """
        初始化轮次选择器
        
        Args:
            parent: 父窗口
            ai_data_dir: ai_data 目录路径
            on_select: 选择回调函数，参数为轮次信息 dict
        """
        self.parent = parent
        self.ai_data_dir = ai_data_dir
        self.on_select = on_select
        
        # 创建窗口
        if parent is None:
            self.root = tk.Tk()
        else:
            self.root = tk.Toplevel(parent)
        
        self.root.title("选择回放轮次")
        self.root.geometry("500x400")
        self.root.resizable(False, False)
        self.root.configure(bg=self.COLOR_BG)
        
        if parent:
            self.root.transient(parent)
            self.root.grab_set()
        
        self._create_widgets()
    
    def _create_widgets(self):
        """创建界面元素"""
        tk.Label(
            self.root,
            text="📁 选择回放轮次",
            font=("微软雅黑", 16, "bold"),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT
        ).pack(pady=15)
        
        # 获取所有轮次目录
        rounds = self._get_rounds()
        
        if not rounds:
            tk.Label(
                self.root,
                text="没有找到回放数据",
                font=("微软雅黑", 12),
                bg=self.COLOR_BG,
                fg="gray"
            ).pack(pady=50)
            
            tk.Button(
                self.root,
                text="关闭",
                font=("微软雅黑", 11),
                width=10,
                command=self._close
            ).pack(pady=20)
            return
        
        # 轮次列表
        self.listbox = tk.Listbox(
            self.root,
            font=("微软雅黑", 11),
            height=12,
            selectmode=tk.SINGLE
        )
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)
        
        self.rounds = rounds
        for r in rounds:
            self.listbox.insert(tk.END, r["display"])
        
        # 绑定双击事件
        self.listbox.bind("<Double-Button-1>", lambda e: self._select_round())
        
        tk.Button(
            self.root,
            text="查看对局",
            font=("微软雅黑", 12, "bold"),
            bg=self.COLOR_BUTTON_BG,
            fg=self.COLOR_TEXT,
            width=15,
            command=self._select_round
        ).pack(pady=20)
    
    def _get_rounds(self):
        """获取所有轮次目录"""
        rounds = []
        
        if not os.path.exists(self.ai_data_dir):
            return rounds
        
        for round_dir in sorted(os.listdir(self.ai_data_dir)):
            round_path = os.path.join(self.ai_data_dir, round_dir)
            if os.path.isdir(round_path):
                # 计算该局数
                game_count = len([d for d in os.listdir(round_path) 
                                if os.path.isdir(os.path.join(round_path, d))])
                rounds.append({
                    "name": round_dir,
                    "path": round_path,
                    "display": f"{round_dir} ({game_count} 局对战)"
                })
        
        return rounds
    
    def _select_round(self):
        """选择轮次"""
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showwarning("提示", "请选择一个轮次")
            return
        
        round_info = self.rounds[selection[0]]
        
        if self.on_select:
            self.on_select(round_info)
        
        self._close()
    
    def _close(self):
        """关闭窗口"""
        self.root.destroy()
    
    def run(self):
        """运行选择器（阻塞）"""
        self.root.mainloop()


class GameSelector:
    """对局选择器 - 选择单个对局进行回放"""
    
    COLOR_BG = COLORS["bg"]
    COLOR_TEXT = COLORS["text"]
    COLOR_BEAR = COLORS["bear"]
    COLOR_BUTTON_BG = "#FFE4B5"
    
    def __init__(self, parent, round_dir, on_select=None):
        """
        初始化对局选择器
        
        Args:
            parent: 父窗口
            round_dir: 轮次目录路径
            on_select: 选择回调函数，参数为对局信息 dict
        """
        self.parent = parent
        self.round_dir = round_dir
        self.on_select = on_select
        
        # 创建窗口
        if parent is None:
            self.root = tk.Tk()
        else:
            self.root = tk.Toplevel(parent)
        
        self.root.title("选择要回放的对局")
        self.root.geometry("500x400")
        self.root.resizable(False, False)
        self.root.configure(bg=self.COLOR_BG)
        
        if parent:
            self.root.transient(parent)
            self.root.grab_set()
        
        self._create_widgets()
    
    def _create_widgets(self):
        """创建界面元素"""
        tk.Label(
            self.root,
            text="📁 选择对局",
            font=("微软雅黑", 16, "bold"),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT
        ).pack(pady=15)
        
        # 获取所有对局
        games = self._get_games()
        
        if not games:
            tk.Label(
                self.root,
                text="没有找到可回放的对局",
                font=("微软雅黑", 12),
                bg=self.COLOR_BG,
                fg="gray"
            ).pack(pady=50)
            
            tk.Button(
                self.root,
                text="关闭",
                font=("微软雅黑", 11),
                width=10,
                command=self._close
            ).pack(pady=20)
            return
        
        # 对局列表
        self.listbox = tk.Listbox(
            self.root,
            font=("微软雅黑", 11),
            height=12,
            selectmode=tk.SINGLE
        )
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)
        
        self.games = games
        for game in games:
            self.listbox.insert(tk.END, game["display"])
        
        # 绑定双击事件
        self.listbox.bind("<Double-Button-1>", lambda e: self._select_game())
        
        tk.Button(
            self.root,
            text="开始回放",
            font=("微软雅黑", 12, "bold"),
            bg=self.COLOR_BUTTON_BG,
            fg=self.COLOR_TEXT,
            width=15,
            command=self._select_game
        ).pack(pady=20)
    
    def _get_games(self):
        """获取所有对局"""
        games = []
        
        if not os.path.exists(self.round_dir):
            return games
        
        for game_dir in sorted(os.listdir(self.round_dir)):
            game_path = os.path.join(self.round_dir, game_dir)
            record_file = os.path.join(game_path, "record.json")
            
            if os.path.isdir(game_path) and os.path.exists(record_file):
                try:
                    with open(record_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    winner = data.get("winner", 0)
                    winner_text = {1: "黑胜", 2: "白胜", 3: "平局", 0: "未完成"}.get(winner, "未知")
                    moves = len(data.get("history", []))
                    
                    games.append({
                        "id": game_dir,
                        "path": game_path,
                        "record_file": record_file,
                        "display": f"第 {game_dir} 局 - {winner_text} ({moves} 步)"
                    })
                except Exception:
                    pass
        
        return games
    
    def _select_game(self):
        """选择对局"""
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showwarning("提示", "请选择一个对局")
            return
        
        game = self.games[selection[0]]
        
        if self.on_select:
            self.on_select(game)
        
        self._close()
    
    def _close(self):
        """关闭窗口"""
        self.root.destroy()
    
    def run(self):
        """运行选择器（阻塞）"""
        self.root.mainloop()


class ReplayBrowser:
    """回放浏览器 - 组合轮次选择和对局选择"""
    
    @staticmethod
    def select_round(ai_data_dir, parent=None):
        """
        选择轮次
        
        Args:
            ai_data_dir: ai_data 目录路径
            parent: 父窗口（可选）
            
        Returns:
            dict: 轮次信息，或 None 如果用户取消
        """
        result = {"round": None}
        
        def on_select(round_info):
            result["round"] = round_info
        
        selector = RoundSelector(parent, ai_data_dir, on_select)
        selector.run()
        
        return result["round"]
    
    @staticmethod
    def select_game(round_dir, parent=None):
        """
        选择对局
        
        Args:
            round_dir: 轮次目录路径
            parent: 父窗口（可选）
            
        Returns:
            dict: 对局信息，或 None 如果用户取消
        """
        result = {"game": None}
        
        def on_select(game_info):
            result["game"] = game_info
        
        selector = GameSelector(parent, round_dir, on_select)
        selector.run()
        
        return result["game"]
    
    @staticmethod
    def browse(ai_data_dir, parent=None):
        """
        完整的浏览流程：先选轮次，再选对局
        
        Args:
            ai_data_dir: ai_data 目录路径
            parent: 父窗口（可选）
            
        Returns:
            dict: 包含 record_file 和 sample_file 路径，或 None 如果用户取消
        """
        # 选择轮次
        round_info = ReplayBrowser.select_round(ai_data_dir, parent)
        if not round_info:
            return None
        
        # 选择对局
        game_info = ReplayBrowser.select_game(round_info["path"], parent)
        if not game_info:
            return None
        
        # 构建返回结果
        record_file = game_info["record_file"]
        
        # 尝试找到对应的 sample.json
        sample_file = record_file.replace("ai_data", "train_data").replace("record.json", "sample.json")
        if not os.path.exists(sample_file):
            sample_file = None
        
        return {
            "record_file": record_file,
            "sample_file": sample_file,
            "round_name": round_info["name"],
            "game_id": game_info["id"]
        }
