"""
AI 内战组件 - 负责 AI 对战模式的配置、执行和进度显示
"""
import tkinter as tk
from tkinter import messagebox, ttk
import threading
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app import BattleSession

from .constants import COLORS, DIMENSIONS, FONTS


class AIBattleConfigDialog:
    """AI 内战配置对话框"""
    
    COLOR_BG = COLORS["bg"]
    COLOR_BEAR = COLORS["bear"]
    COLOR_BUNNY = COLORS["rabbit"]
    COLOR_TEXT = COLORS["text"]
    COLOR_BUTTON_BG = "#FFE4B5"
    COLOR_BORDER = COLORS["border"]
    
    def __init__(self, parent):
        """
        初始化配置对话框
        
        Args:
            parent: 父窗口
        """
        self.parent = parent
        self.result = None
        
        # 创建对话框
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("AI 内战配置")
        self.dialog.geometry("400x400")
        self.dialog.resizable(False, False)
        self.dialog.configure(bg=self.COLOR_BG)
        self.dialog.transient(parent)
        # 只在父窗口有 grab 时设置 grab_set，避免测试时冲突
        try:
            self.dialog.grab_set()
        except tk.TclError:
            pass  # 忽略 grab 失败
        
        self._create_widgets()
    
    def _create_widgets(self):
        """创建界面元素"""
        tk.Label(
            self.dialog,
            text="⚔️ AI 内战模式",
            font=("微软雅黑", 16, "bold"),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT
        ).pack(pady=15)
        
        # 获取可用 AI 列表
        available_ais = self._get_available_ais()
        
        # 黑方 AI 选择
        tk.Label(
            self.dialog,
            text="🐻 黑方 AI:",
            font=("微软雅黑", 11),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT
        ).pack(pady=(15, 5))
        
        self.ai1_var = tk.StringVar(value=available_ais[0] if available_ais else "random")
        ai1_combo = ttk.Combobox(
            self.dialog,
            textvariable=self.ai1_var,
            values=available_ais,
            font=("微软雅黑", 11),
            width=20,
            state="readonly"
        )
        ai1_combo.pack()
        
        # 白方 AI 选择
        tk.Label(
            self.dialog,
            text="🐰 白方 AI:",
            font=("微软雅黑", 11),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT
        ).pack(pady=(15, 5))
        
        self.ai2_var = tk.StringVar(value=available_ais[0] if available_ais else "random")
        ai2_combo = ttk.Combobox(
            self.dialog,
            textvariable=self.ai2_var,
            values=available_ais,
            font=("微软雅黑", 11),
            width=20,
            state="readonly"
        )
        ai2_combo.pack()
        
        # 局数选择
        tk.Label(
            self.dialog,
            text="对战局数:",
            font=("微软雅黑", 11),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT
        ).pack(pady=(20, 5))
        
        self.num_games_var = tk.StringVar(value="1")
        spinbox = tk.Spinbox(
            self.dialog,
            from_=1,
            to=100,
            textvariable=self.num_games_var,
            font=("微软雅黑", 12),
            width=10,
            justify=tk.CENTER
        )
        spinbox.pack()
        
        tk.Button(
            self.dialog,
            text="开始对战",
            font=("微软雅黑", 12, "bold"),
            bg=self.COLOR_BUTTON_BG,
            fg=self.COLOR_TEXT,
            width=15,
            command=self._start_battle
        ).pack(pady=25)
    
    def _get_available_ais(self):
        """获取可用的 AI 列表 - 动态扫描 ai/models/ 目录下的子目录"""
        ai_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        ai_dir = os.path.join(ai_dir, "ai", "models")
        ais = []
        for name in os.listdir(ai_dir):
            # 排除 __ 开头的目录（如 __pycache__）
            if name.startswith("__"):
                continue
            subdir = os.path.join(ai_dir, name)
            if os.path.isdir(subdir):
                # 检查是否有 __init__.py 和模型导出
                init_file = os.path.join(subdir, "__init__.py")
                if os.path.exists(init_file):
                    ais.append(name)
        return sorted(ais)
    
    def _start_battle(self):
        """验证并启动对战"""
        try:
            num_games = int(self.num_games_var.get())
            if 1 <= num_games <= 100:
                self.result = {
                    "num_games": num_games,
                    "ai1": self.ai1_var.get(),
                    "ai2": self.ai2_var.get()
                }
                self.dialog.destroy()
            else:
                messagebox.showwarning("提示", "局数必须在 1-100 之间")
        except ValueError:
            messagebox.showwarning("提示", "请输入有效的数字")
    
    def get_result(self):
        """获取配置结果"""
        return self.result


class AIBattleProgressWindow:
    """AI 内战进度窗口"""
    
    COLOR_BG = COLORS["bg"]
    COLOR_TEXT = COLORS["text"]
    COLOR_BEAR = COLORS["bear"]
    
    def __init__(self, num_games, ai1_name="random", ai2_name="random"):
        """
        初始化进度窗口
        
        Args:
            num_games: 对战局数
            ai1_name: 黑方 AI 名称
            ai2_name: 白方 AI 名称
        """
        self.num_games = num_games
        self.ai1_name = ai1_name
        self.ai2_name = ai2_name
        self.recorder = None
        self.on_complete = None
        
        # 创建窗口
        self.root = tk.Tk()
        self.root.title("AI 内战进行中...")
        self.root.geometry("400x250")
        self.root.resizable(False, False)
        self.root.configure(bg=self.COLOR_BG)
        
        self._create_widgets()
    
    def _create_widgets(self):
        """创建界面元素"""
        tk.Label(
            self.root,
            text="⚔️ AI 内战模式",
            font=("微软雅黑", 16, "bold"),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT
        ).pack(pady=15)
        
        tk.Label(
            self.root,
            text=f"{self.ai1_name} (黑) vs {self.ai2_name} (白)",
            font=("微软雅黑", 11),
            bg=self.COLOR_BG,
            fg="gray"
        ).pack()
        
        self.status_label = tk.Label(
            self.root,
            text="准备开始对战...",
            font=("微软雅黑", 11),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT
        )
        self.status_label.pack(pady=10)
        
        self.progress_label = tk.Label(
            self.root,
            text=f"0 / {self.num_games}",
            font=("微软雅黑", 12),
            bg=self.COLOR_BG,
            fg=self.COLOR_BEAR
        )
        self.progress_label.pack(pady=10)

    def start_battle(self, on_complete=None):
        """
        开始对战

        Args:
            on_complete: 完成回调函数，参数为 (recorder, should_replay)
        """
        self.on_complete = on_complete

        # 延迟启动，让窗口先显示
        self.root.after(500, self._run_battle)
        self.root.mainloop()

    def _run_battle(self):
        """在后台线程中运行对战"""
        def battle_thread():
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

            session = BattleSession(
                ai1_name=self.ai1_name,
                ai2_name=self.ai2_name,
                num_games=self.num_games,
                project_root=project_root,
            )
            self._battle_session = session

            def on_progress(current, total):
                self.root.after(0, lambda idx=current: (
                    self.status_label.config(text=f"正在进行第 {idx} 局对战..."),
                    self.progress_label.config(text=f"{idx} / {self.num_games}"),
                ))

            session.run(on_progress=on_progress)
            self.recorder = session.get_recorder()

            # 完成对战
            self.root.after(0, self._on_battle_complete)

        thread = threading.Thread(target=battle_thread)
        thread.daemon = True
        thread.start()

    def _on_battle_complete(self):
        """对战完成处理"""
        self.status_label.config(text="✅ 所有对战已完成！训练样本已生成。", fg="green")
        self.progress_label.config(text=f"{self.num_games} / {self.num_games}")

        # 延迟询问是否回放
        self.root.after(1500, self._ask_replay)
    
    def _ask_replay(self):
        """询问是否回放"""
        if messagebox.askyesno("对战完成", "是否回放对局？"):
            should_replay = True
        else:
            should_replay = False
        
        self.root.destroy()
        
        if self.on_complete:
            self.on_complete(self.recorder, should_replay)


class AIBattleManager:
    """AI 内战管理器 - 简化调用的接口"""
    
    @staticmethod
    def run_ai_vs_ai(num_games, ai1_name="random", ai2_name="random", parent=None):
        """
        运行 AI 内战模式
        
        Args:
            num_games: 对战局数
            ai1_name: 黑方 AI 名称
            ai2_name: 白方 AI 名称
            parent: 父窗口（可选）
            
        Returns:
            tuple: (recorder, should_replay) 或 (None, False) 如果用户取消
        """
        result = {"recorder": None, "should_replay": False}
        
        def on_complete(recorder, should_replay):
            result["recorder"] = recorder
            result["should_replay"] = should_replay
        
        progress = AIBattleProgressWindow(num_games, ai1_name, ai2_name)
        progress.start_battle(on_complete)
        
        return result["recorder"], result["should_replay"]
