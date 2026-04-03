"""游戏模式选择器 - 启动时显示

这个模块包含 GameModeSelector 类，用于在游戏启动时
显示游戏模式选择界面。
"""

import tkinter as tk
from tkinter import messagebox, ttk
from .constants import COLORS, DIMENSIONS, FONTS


class GameModeSelector:
    """游戏模式选择器 - 启动时显示"""
    
    # 配色方案 - 从 constants 导入
    COLOR_BG = COLORS["bg"]              # 背景色（米色）
    COLOR_BEAR = COLORS["bear"]          # 小熊棕
    COLOR_BUNNY = COLORS["rabbit"]       # 小兔粉
    COLOR_TEXT = COLORS["text"]          # 文字颜色（深棕）
    COLOR_BUTTON_BG = "#FFE4B5"          # 按钮背景色（暖黄）
    COLOR_BORDER = COLORS["border"]      # 边框颜色
    
    # 窗口尺寸
    WINDOW_WIDTH = DIMENSIONS["mode_selector_width"]
    WINDOW_HEIGHT = DIMENSIONS["mode_selector_height"]
    
    def __init__(self, root):
        """初始化模式选择器
        
        Args:
            root: Tkinter 根窗口
        """
        self.root = root
        self.root.title("🐻 森林五子棋 - 选择游戏模式 🐰")
        self.root.geometry(f"{self.WINDOW_WIDTH}x{self.WINDOW_HEIGHT}")
        self.root.resizable(False, False)
        self.root.configure(bg=self.COLOR_BG)
        
        # 选择结果
        self.result = None
        
        # 创建界面
        self._create_widgets()
        
        # 居中显示
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
    
    def _create_widgets(self):
        """创建界面元素"""
        # 标题
        title_frame = tk.Frame(self.root, bg=self.COLOR_BG)
        title_frame.pack(pady=30)
        
        tk.Label(
            title_frame,
            text="🐻",
            font=("微软雅黑", 36),
            bg=self.COLOR_BG
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Label(
            title_frame,
            text="森林五子棋",
            font=("微软雅黑", 24, "bold"),
            bg=self.COLOR_BG,
            fg=self.COLOR_BEAR
        ).pack(side=tk.LEFT)
        
        tk.Label(
            title_frame,
            text="🐰",
            font=("微软雅黑", 36),
            bg=self.COLOR_BG
        ).pack(side=tk.LEFT, padx=10)
        
        # 副标题
        tk.Label(
            self.root,
            text="请选择游戏模式",
            font=("微软雅黑", 14),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT
        ).pack(pady=20)
        
        # 模式选择按钮
        modes = [
            {
                "text": "👥 人人对战",
                "desc": "两位玩家轮流对弈",
                "command": self._select_pvp,
                "color": "#90EE90"
            },
            {
                "text": "🤖 人机对战",
                "desc": "玩家 vs AI 电脑",
                "command": self._select_pve,
                "color": "#87CEEB"
            },
            {
                "text": "⚔️ AI 内战",
                "desc": "AI 自动对战模式",
                "command": self._select_ai_vs_ai,
                "color": "#DDA0DD"
            }
        ]
        
        for mode in modes:
            self._create_mode_button(mode)
        
        # 分隔线
        tk.Frame(self.root, height=2, bg=self.COLOR_BORDER).pack(fill=tk.X, padx=40, pady=15)
        
        # 查看回放按钮
        replay_frame = tk.Frame(self.root, bg=self.COLOR_BG)
        replay_frame.pack(pady=10, padx=40, fill=tk.X)
        
        tk.Button(
            replay_frame,
            text="📺 查看回放",
            font=("微软雅黑", 12, "bold"),
            bg="#F0E68C",
            fg=self.COLOR_TEXT,
            activebackground=self.COLOR_BEAR,
            activeforeground="white",
            relief=tk.RAISED,
            bd=3,
            cursor="hand2",
            command=self._select_replay
        ).pack(fill=tk.X, pady=5)
        
        tk.Label(
            replay_frame,
            text="查看历史 AI 对战记录",
            font=("微软雅黑", 10),
            bg=self.COLOR_BG,
            fg="gray"
        ).pack()
    
    def _create_mode_button(self, mode):
        """创建模式选择按钮
        
        Args:
            mode: 包含按钮信息的字典 {"text", "desc", "command", "color"}
        """
        frame = tk.Frame(self.root, bg=self.COLOR_BG)
        frame.pack(pady=10, padx=40, fill=tk.X)
        
        btn = tk.Button(
            frame,
            text=mode["text"],
            font=("微软雅黑", 14, "bold"),
            bg=mode["color"],
            fg=self.COLOR_TEXT,
            activebackground=self.COLOR_BEAR,
            activeforeground="white",
            relief=tk.RAISED,
            bd=3,
            cursor="hand2",
            command=mode["command"]
        )
        btn.pack(fill=tk.X, pady=5)
        
        tk.Label(
            frame,
            text=mode["desc"],
            font=("微软雅黑", 10),
            bg=self.COLOR_BG,
            fg="gray"
        ).pack()
    
    def _select_pvp(self):
        """选择人人对战"""
        self.result = {"mode": "pvp"}
        self.root.destroy()
    
    def _select_replay(self):
        """选择查看回放"""
        self.result = {"mode": "replay"}
        self.root.destroy()
    
    def _select_pve(self):
        """选择人机对战，弹出颜色选择"""
        self._show_color_selection()
    
    def _show_color_selection(self):
        """显示执棋颜色和AI选择对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title("人机对战配置")
        dialog.geometry("350x350")
        dialog.resizable(False, False)
        dialog.configure(bg=self.COLOR_BG)
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(
            dialog,
            text="🤖 人机对战配置",
            font=("微软雅黑", 14, "bold"),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT
        ).pack(pady=15)
        
        # 获取可用 AI 列表
        available_ais = self._get_available_ais()
        
        # AI 选择
        tk.Label(
            dialog,
            text="选择 AI 对手:",
            font=("微软雅黑", 11),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT
        ).pack(pady=(10, 5))
        
        ai_var = tk.StringVar(value=available_ais[0] if available_ais else "random")
        ai_combo = ttk.Combobox(
            dialog,
            textvariable=ai_var,
            values=available_ais,
            font=("微软雅黑", 11),
            width=20,
            state="readonly"
        )
        ai_combo.pack()
        
        # 分隔线
        tk.Frame(dialog, height=2, bg=self.COLOR_BORDER).pack(fill=tk.X, padx=30, pady=15)
        
        tk.Label(
            dialog,
            text="请选择您要执的棋子颜色",
            font=("微软雅黑", 11),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT
        ).pack(pady=10)
        
        def select_color(color):
            """处理颜色选择
            
            Args:
                color: 选择的颜色 "black" 或 "white"
            """
            self.result = {
                "mode": "pve",
                "player_color": color,
                "ai": ai_var.get()
            }
            dialog.destroy()
            self.root.destroy()
        
        # 黑棋按钮
        tk.Button(
            dialog,
            text="🐻 执黑棋（先手）",
            font=("微软雅黑", 11, "bold"),
            bg=self.COLOR_BEAR,
            fg="white",
            width=20,
            command=lambda: select_color("black")
        ).pack(pady=10)
        
        # 白棋按钮
        tk.Button(
            dialog,
            text="🐰 执白棋（后手）",
            font=("微软雅黑", 11, "bold"),
            bg=self.COLOR_BUNNY,
            fg=self.COLOR_TEXT,
            width=20,
            command=lambda: select_color("white")
        ).pack(pady=10)
    
    def _select_ai_vs_ai(self):
        """选择 AI 内战，弹出局数设置"""
        self._show_ai_vs_ai_config()
    
    def _show_ai_vs_ai_config(self):
        """显示 AI 内战配置对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title("AI 内战配置")
        dialog.geometry("400x400")
        dialog.resizable(False, False)
        dialog.configure(bg=self.COLOR_BG)
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(
            dialog,
            text="⚔️ AI 内战模式",
            font=("微软雅黑", 16, "bold"),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT
        ).pack(pady=15)
        
        # 获取可用 AI 列表
        available_ais = self._get_available_ais()
        
        # 黑方 AI 选择
        tk.Label(
            dialog,
            text="🐻 黑方 AI:",
            font=("微软雅黑", 11),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT
        ).pack(pady=(15, 5))
        
        ai1_var = tk.StringVar(value=available_ais[0] if available_ais else "random")
        ai1_combo = ttk.Combobox(
            dialog,
            textvariable=ai1_var,
            values=available_ais,
            font=("微软雅黑", 11),
            width=20,
            state="readonly"
        )
        ai1_combo.pack()
        
        # 白方 AI 选择
        tk.Label(
            dialog,
            text="🐰 白方 AI:",
            font=("微软雅黑", 11),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT
        ).pack(pady=(15, 5))
        
        ai2_var = tk.StringVar(value=available_ais[0] if available_ais else "random")
        ai2_combo = ttk.Combobox(
            dialog,
            textvariable=ai2_var,
            values=available_ais,
            font=("微软雅黑", 11),
            width=20,
            state="readonly"
        )
        ai2_combo.pack()
        
        # 局数选择
        tk.Label(
            dialog,
            text="对战局数:",
            font=("微软雅黑", 11),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT
        ).pack(pady=(20, 5))
        
        num_games_var = tk.StringVar(value="1")
        spinbox = tk.Spinbox(
            dialog,
            from_=1,
            to=100,
            textvariable=num_games_var,
            font=("微软雅黑", 12),
            width=10,
            justify=tk.CENTER
        )
        spinbox.pack()
        
        def start_battle():
            """验证并启动对战"""
            try:
                num_games = int(num_games_var.get())
                if 1 <= num_games <= 100:
                    self.result = {
                        "mode": "ai_vs_ai",
                        "num_games": num_games,
                        "ai1": ai1_var.get(),
                        "ai2": ai2_var.get()
                    }
                    dialog.destroy()
                    self.root.destroy()
                else:
                    messagebox.showwarning("提示", "局数必须在 1-100 之间")
            except ValueError:
                messagebox.showwarning("提示", "请输入有效的数字")
        
        tk.Button(
            dialog,
            text="开始对战",
            font=("微软雅黑", 12, "bold"),
            bg=self.COLOR_BUTTON_BG,
            fg=self.COLOR_TEXT,
            width=15,
            command=start_battle
        ).pack(pady=25)
    
    def _get_available_ais(self):
        """获取可用的 AI 列表
        
        动态扫描 ai 目录下的所有 AI 模型子目录。
        
        Returns:
            list: 可用 AI 模型名称列表
        """
        import os
        
        # 获取 ai/models 目录路径（相对于项目根目录）
        ai_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "ai", "models"
        )
        
        ais = []
        if os.path.exists(ai_dir):
            for item in os.listdir(ai_dir):
                item_path = os.path.join(ai_dir, item)
                # 只包含目录，且排除 __pycache__ 等特殊目录
                if os.path.isdir(item_path) and not item.startswith("__"):
                    ais.append(item)
        
        return sorted(ais) if ais else ["random"]
    
    def get_result(self):
        """获取选择结果
        
        Returns:
            dict: 用户选择的模式和参数，或 None 如果用户取消
        """
        return self.result
