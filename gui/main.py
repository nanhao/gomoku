"""
五子棋 GUI 主程序 - 重构版

支持三种游戏模式：
1. 人人对战（PvP）- 传统双人对战
2. 人机对战（PvE）- 玩家 vs AI
3. AI 内战（AI vs AI）- AI 自动对战
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tkinter as tk
from tkinter import messagebox, simpledialog

from game import GomokuController
from app import GameSession, GUIHelper, get_ai_data_dir, create_pve_config

from gui.components import GameModeSelector, GameBoard
from gui.components import ReplayBrowser, ReplayWindow
from gui.components import AIBattleConfigDialog, AIBattleProgressWindow
from gui.components import LoadGameDialog
from gui.components.constants import ANIMALS, COLORS, FONTS
from gui.display_updater import GameDisplayUpdater
from gui.ai_battle import run_ai_vs_ai_mode


class GomokuGUI:
    """五子棋主界面 - 使用组件重构版"""
    
    # 配色方案 - 森林/小动物主题
    COLOR_BG = COLORS["bg"]
    COLOR_TEXT = COLORS["text"]
    COLOR_BEAR = COLORS["bear"]
    COLOR_BUNNY = COLORS["rabbit"]
    COLOR_BORDER = COLORS["border"]
    
    
    def __init__(self, root, game_mode="pvp", ai_player=None, ai_model="random"):
        """
        初始化主界面
        
        Args:
            root: Tkinter 根窗口
            game_mode: 游戏模式
            ai_player: AI 执棋颜色（人机对战模式）
            ai_model: AI 模型名称（人机对战模式）
        """
        self.root = root
        self.root.title("🐻 森林五子棋 🐰")
        self.root.resizable(False, False)
        self.root.configure(bg=self.COLOR_BG)
        
        # 初始化控制器
        self.controller = GomokuController()

        # 创建游戏会话（协调游戏流程）
        self.session = GameSession(
            mode=game_mode,
            controller=self.controller,
            ai_name=ai_model,
            ai_player=ai_player,
        )

        # 创建 GUI 辅助对象
        self.helper = GUIHelper(self.controller, self.session)

        # 创建界面
        self._create_menu()
        self._create_main_frame()

        # 创建显示更新器（必须在 _create_main_frame 之后，因为需要控件引用）
        self._display_updater = GameDisplayUpdater(
            board=self.board,
            save_name_label=self.save_name_label,
            status_label=self.status_label,
            game_status_label=self.game_status_label,
            history_listbox=self.history_listbox,
            helper=self.helper,
            color_text=self.COLOR_TEXT,
            color_bear=self.COLOR_BEAR,
        )

        # 如果是人机对战且 AI 执黑，自动启动游戏（由 session 处理 AI 落子）
        if game_mode == "pve" and ai_player == 1:
            self.root.after(500, self._start_game_with_ai_first)
    
    def _create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 游戏菜单
        game_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="游戏", menu=game_menu)
        game_menu.add_command(label="新建游戏", command=self._new_game)
        game_menu.add_command(label="保存游戏", command=self._save_game)
        game_menu.add_command(label="加载游戏", command=self._load_game)
        game_menu.add_separator()
        game_menu.add_command(label="退出", command=self.root.quit)
    
    def _create_main_frame(self):
        """创建主框架"""
        # 主容器
        main_frame = tk.Frame(self.root, bg=self.COLOR_BG)
        main_frame.pack(padx=20, pady=20)
        
        # 创建棋盘组件
        self.board = GameBoard(main_frame, on_click_callback=self._on_board_click)
        self.board.pack(side=tk.LEFT)
        
        # 右侧信息面板
        info_frame = tk.Frame(main_frame, bg=self.COLOR_BG, padx=20)
        info_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # 标题（动物主题）
        title_frame = tk.Frame(info_frame, bg=self.COLOR_BG)
        title_frame.pack(pady=10)
        
        tk.Label(
            title_frame,
            text=ANIMALS["black"],
            font=("微软雅黑", 24),
            bg=self.COLOR_BG
        ).pack(side=tk.LEFT, padx=3)
        
        tk.Label(
            title_frame,
            text="森林五子棋",
            font=FONTS["title"],
            bg=self.COLOR_BG,
            fg=self.COLOR_BEAR
        ).pack(side=tk.LEFT)
        
        tk.Label(
            title_frame,
            text=ANIMALS["white"],
            font=("微软雅黑", 24),
            bg=self.COLOR_BG
        ).pack(side=tk.LEFT, padx=3)
        
        # 分隔线
        tk.Frame(info_frame, height=2, bg=self.COLOR_BORDER).pack(fill=tk.X, pady=10)
        
        # 存档名称
        self.save_name_label = tk.Label(
            info_frame,
            text="🌲 当前存档: 无",
            font=FONTS["normal"],
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT
        )
        self.save_name_label.pack(pady=8)
        
        # 游戏状态标签
        self.status_label = tk.Label(
            info_frame,
            text="🎯 当前玩家: -",
            font=("微软雅黑", 14, "bold"),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT
        )
        self.status_label.pack(pady=10)
        
        # 游戏状态
        self.game_status_label = tk.Label(
            info_frame,
            text="🌳 游戏状态: 未开始",
            font=FONTS["normal"],
            bg=self.COLOR_BG,
            fg="gray"
        )
        self.game_status_label.pack(pady=8)
        
        # 分隔线
        tk.Frame(info_frame, height=2, bg=self.COLOR_BORDER).pack(fill=tk.X, pady=10)
        
        # 历史记录
        tk.Label(
            info_frame,
            text="📝 对局记录",
            font=FONTS["normal"],
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT
        ).pack(pady=(10, 5), anchor=tk.W)
        
        # 历史记录列表框
        history_outer_frame = tk.Frame(
            info_frame,
            bg=self.COLOR_BORDER,
            bd=2,
            relief=tk.RIDGE
        )
        history_outer_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        history_frame = tk.Frame(history_outer_frame, bg=self.COLOR_BG)
        history_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        scrollbar = tk.Scrollbar(history_frame, bg="#FFE4B5")
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.history_listbox = tk.Listbox(
            history_frame,
            height=12,
            width=22,
            yscrollcommand=scrollbar.set,
            font=("Consolas", 10),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT,
            selectbackground=self.COLOR_BEAR,
            selectforeground="white",
            relief=tk.FLAT,
            highlightthickness=0
        )
        self.history_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.history_listbox.yview)
        
        # 控制按钮
        self._create_control_buttons(info_frame)
    
    def _on_board_click(self, row, col):
        """处理棋盘点击"""
        if self.controller.game is None:
            messagebox.showinfo("提示", "请先新建游戏")
            return

        result = self.helper.handle_player_move(row, col)

        if result["status"] == "success":
            self._update_display()

            if result["game_over"]:
                self._show_game_over(result["winner"])
                return

            if result["need_ai_move"]:
                self.root.after(500, self._ai_move)
        else:
            messagebox.showinfo("提示", result.get("error", "落子失败"))
    
    def _start_game_with_ai_first(self):
        """AI 先走：启动游戏并更新界面"""
        result = self.helper.start_game_with_ai_first()

        if result["status"] == "error":
            messagebox.showerror("错误", result.get("error", "启动游戏失败"))
            return

        self._update_display()

        if result.get("game_over"):
            self._show_game_over(result["winner"])

    def _ai_move(self):
        """AI 落子"""
        result = self.helper.handle_ai_move()

        if result["status"] == "success":
            self._update_display()

            if result["game_over"]:
                self._show_game_over(result["winner"])
        else:
            messagebox.showerror("错误", result.get("error", "AI 落子失败"))
    
    def _update_display(self):
        """更新显示"""
        display_state = self.helper.get_display_state()
        self._display_updater.update(display_state)
    
    def _update_history(self, history):
        """更新历史记录显示"""
        self._display_updater.update_history(history)

    def _show_game_over(self, winner):
        """显示游戏结束弹窗

        Args:
            winner: 获胜方 (1=黑胜, 2=白胜, 3=平局)
        """
        self._display_updater.show_game_over(winner)
    

    def _create_control_buttons(self, parent):
        """创建控制按钮"""
        button_frame = tk.Frame(parent, bg=self.COLOR_BG)
        button_frame.pack(pady=15)
        
        # 按钮配置（森林动物主题）
        button_config = [
            ("🌱 新建游戏", self._new_game, "#90EE90"),
            ("🌲 加载游戏", self._load_game, "#DEB887"),
            ("💾 保存游戏", self._save_game, "#87CEEB"),
            ("📺 回放对局", self._open_replay, "#DDA0DD"),
        ]
        
        for text, command, bg_color in button_config:
            btn = tk.Button(
                button_frame,
                text=text,
                command=command,
                width=16,
                font=FONTS["button"],
                bg=bg_color,
                fg=self.COLOR_TEXT,
                activebackground=self.COLOR_BEAR,
                activeforeground="white",
                relief=tk.RAISED,
                bd=3,
                cursor="hand2"
            )
            btn.pack(pady=6)
            
            # 添加悬停效果
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg=self.COLOR_BEAR, fg="white"))
            btn.bind("<Leave>", lambda e, b=btn, c=bg_color: b.config(bg=c, fg=self.COLOR_TEXT))

    def _new_game(self):
        """新建游戏"""
        # 弹出对话框询问存档名称
        save_name = simpledialog.askstring(
            "新建游戏",
            "请输入存档名称（留空使用默认名称）:",
            parent=self.root
        )

        if save_name is None:  # 用户取消
            return

        save_name = save_name.strip() if save_name else None

        result = self.session.start_game(save_name)

        if result["status"] == "error":
            messagebox.showerror("错误", result.get("error", "创建游戏失败"))
            return

        self._update_display()
        messagebox.showinfo("成功", f"游戏已创建！\n存档名称: {self.controller.current_save_name}")
    
    def _save_game(self):
        """保存游戏"""
        if self.controller.game is None:
            messagebox.showwarning("提示", "没有正在进行的游戏")
            return
        
        # 游戏会自动保存，这里只是提示
        save_name = self.controller.current_save_name or "未知"
        messagebox.showinfo("提示", f"游戏已自动保存！\n存档名称: {save_name}")
    
    def _load_game(self):
        """加载游戏"""
        # 获取所有存档
        saves = self.controller.list_saves()
        
        if not saves:
            messagebox.showinfo("提示", "暂无存档")
            return
        
        # 弹出选择对话框
        save_name = LoadGameDialog.show(self.root, saves)
        
        if save_name is None:
            return
        
        result = self.controller.load_game(save_name)
        
        if "error" in result:
            messagebox.showerror("错误", result["error"])
        else:
            self._update_display()
            messagebox.showinfo("成功", f"存档 '{save_name}' 已加载！")
    

    
    def _open_replay(self):
        """打开回放浏览器"""
        # 获取 ai_data 目录路径
        ai_data_dir = get_ai_data_dir()

        if not os.path.exists(ai_data_dir):
            messagebox.showinfo("提示", "暂无回放数据\n请先进行 AI 内战对战")
            return
        
        # 使用回放浏览器选择对局
        result = ReplayBrowser.browse(ai_data_dir, self.root)
        
        if result:
            # 关闭当前窗口，打开回放窗口
            self.root.destroy()
            # 显示回放
            ReplayWindow.show(result["record_file"], result["sample_file"])


def main():
    """主函数"""
    # 创建模式选择器
    selector_root = tk.Tk()
    selector = GameModeSelector(selector_root)
    selector_root.mainloop()
    
    # 获取选择结果
    if selector.result is None:
        return
    
    mode = selector.result.get("mode")
    
    # AI 内战模式
    if mode == "ai_vs_ai":
        num_games = selector.result.get("num_games", 1)
        ai1 = selector.result.get("ai1", "random")
        ai2 = selector.result.get("ai2", "random")
        run_ai_vs_ai_mode(num_games, ai1, ai2)
        return
    
    # 回放模式
    if mode == "replay":
        # 获取 ai_data 目录路径
        ai_data_dir = get_ai_data_dir()

        if not os.path.exists(ai_data_dir):
            messagebox.showinfo("提示", "暂无回放数据\n请先进行 AI 内战对战")
            return

        # 使用回放浏览器
        result = ReplayBrowser.browse(ai_data_dir)
        if result:
            ReplayWindow.show(result["record_file"], result["sample_file"])
        return

    # 人人对战或人机对战
    game_mode = mode
    ai_player = None
    ai_model = "random"

    if game_mode == "pve":
        # 使用辅助函数创建 PvE 配置
        pve_config = create_pve_config(selector.result)
        ai_player = pve_config["ai_player"]
        ai_model = pve_config["ai_model"]
    
    # 创建主游戏窗口
    root = tk.Tk()
    app = GomokuGUI(root, game_mode=game_mode, ai_player=ai_player, ai_model=ai_model)
    root.mainloop()


if __name__ == "__main__":
    main()
