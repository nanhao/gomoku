"""
回放查看器组件 - 负责游戏回放的显示和控制
"""
import tkinter as tk
from tkinter import ttk
import json
import os

from .heatmap import HeatmapCanvas, HeatmapLegend
from .constants import COLORS, DIMENSIONS, FONTS


class ReplayViewer:
    """回放查看器组件"""
    
    def __init__(self, parent, record_file, sample_file=None):
        """
        初始化回放查看器
        
        Args:
            parent: 父窗口
            record_file: record.json 文件路径
            sample_file: sample.json 文件路径（可选）
        """
        self.parent = parent
        self.record_file = record_file
        self.sample_file = sample_file
        
        # 加载数据
        self.history = []
        self.sample_data = []
        self._load_data()
        
        # 当前步骤
        self.current_step = 0
        
        # 显示选项
        self.show_board = tk.BooleanVar(value=True)
        self.show_input = tk.BooleanVar(value=False)
        self.show_output = tk.BooleanVar(value=False)
        
        # 采样热力图（第四十一次追加：温度采样展示）
        self.sampling_heatmap = None
        
        # 创建界面
        self._create_widgets()
        
        # 初始显示
        self._update_display()
    
    def _load_data(self):
        """加载回放数据"""
        # 加载 record.json
        with open(self.record_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.history = data.get("history", [])
        
        # 加载 sample.json（如果存在）
        # sample.json 是一个列表，每个元素包含 input, output, mask
        if self.sample_file and os.path.exists(self.sample_file):
            with open(self.sample_file, 'r', encoding='utf-8') as f:
                self.sample_data = json.load(f)
                # 确保是列表
                if not isinstance(self.sample_data, list):
                    self.sample_data = []
    
    def _create_widgets(self):
        """创建界面元素"""
        # 主框架
        main_frame = tk.Frame(self.parent, bg="#F5F5DC")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 控制按钮区域
        control_frame = tk.Frame(main_frame, bg="#F5F5DC")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 步骤控制按钮
        btn_frame = tk.Frame(control_frame, bg="#F5F5DC")
        btn_frame.pack(side=tk.LEFT)
        
        tk.Button(
            btn_frame, text="⏮ 上一步",
            command=self._prev_step
        ).pack(side=tk.LEFT, padx=2)
        
        tk.Button(
            btn_frame, text="⏭ 下一步",
            command=self._next_step
        ).pack(side=tk.LEFT, padx=2)
        
        tk.Button(
            btn_frame, text="⏩ 跳转到结局",
            command=self._goto_end
        ).pack(side=tk.LEFT, padx=2)
        
        tk.Button(
            btn_frame, text="🔄 重置",
            command=self._reset
        ).pack(side=tk.LEFT, padx=2)
        
        # 步骤标签
        self.step_label = tk.Label(
            control_frame,
            text="第 0 / 0 步",
            font=("微软雅黑", 12, "bold"),
            bg="#F5F5DC",
            fg="#5D4E37"
        )
        self.step_label.pack(side=tk.RIGHT, padx=10)
        
        # 内容区域（棋盘 + 热力图）
        content_frame = tk.Frame(main_frame, bg="#F5F5DC")
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧：棋盘
        board_frame = tk.Frame(content_frame, bg="#F5F5DC")
        board_frame.pack(side=tk.LEFT, padx=10)
        
        # 创建棋盘画布
        self.board_canvas = tk.Canvas(
            board_frame,
            width=500,
            height=500,
            bg="#E3C16F",
            highlightthickness=2,
            highlightbackground="#8B4513"
        )
        self.board_canvas.pack()
        
        # 右侧：信息面板
        right_frame = tk.Frame(content_frame, bg="#F5F5DC")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        # 热力图区域（上部）
        heatmap_frame = tk.Frame(right_frame, bg="#F5F5DC")
        heatmap_frame.pack(fill=tk.X, pady=(0, 10))
        
        # AI 热力图
        ai_frame = tk.Frame(heatmap_frame, bg="#F5F5DC")
        ai_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        self.ai_heatmap = HeatmapCanvas(ai_frame, title="AI", width=280, height=280)
        self.ai_heatmap.pack()
        
        tk.Label(
            ai_frame,
            text="AI",
            font=("微软雅黑", 9, "bold"),
            bg="#F5F5DC",
            fg="#8B4513"
        ).pack(pady=(5, 0))
        
        # Sample 热力图（如果有数据）
        if self.sample_data:
            sample_frame = tk.Frame(heatmap_frame, bg="#F5F5DC")
            sample_frame.pack(side=tk.LEFT, padx=(20, 0))
            
            self.sample_heatmap = HeatmapCanvas(sample_frame, title="Sample", width=280, height=280)
            self.sample_heatmap.pack()
            
            tk.Label(
                sample_frame,
                text="Sample",
                font=("微软雅黑", 9, "bold"),
                bg="#F5F5DC",
                fg="#D2691E"
            ).pack(pady=(5, 0))
        else:
            self.sample_heatmap = None
        
        # Sampling 热力图（第四十一次追加：温度采样展示）
        # 当有 value_evaluations 时复用为 Value Evals 展示
        sampling_frame = tk.Frame(heatmap_frame, bg="#F5F5DC")
        sampling_frame.pack(side=tk.LEFT, padx=(20, 0))
        
        self.sampling_heatmap = HeatmapCanvas(sampling_frame, title="Sampling", width=280, height=280)
        self.sampling_heatmap.pack()
        
        self.sampling_label = tk.Label(
            sampling_frame,
            text="Sampling",
            font=("微软雅黑", 9, "bold"),
            bg="#F5F5DC",
            fg="#228B22"
        )
        self.sampling_label.pack(pady=(5, 0))
        
        # 颜色标尺
        self.legend = HeatmapLegend(right_frame)
        self.legend.pack(fill=tk.X, pady=(10, 0))
        
        # 信息显示区域
        self.info_text = tk.Text(
            right_frame,
            font=("Consolas", 9),
            wrap=tk.WORD,
            bg="#FFF8DC",
            fg="#5D4E37",
            height=10
        )
        self.info_text.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # 历史记录列表
        history_frame = tk.Frame(main_frame, bg="#F5F5DC")
        history_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(
            history_frame,
            text="历史记录:",
            font=("微软雅黑", 11, "bold"),
            bg="#F5F5DC",
            fg="#5D4E37"
        ).pack(anchor=tk.W)
        
        self.history_listbox = tk.Listbox(
            history_frame,
            font=("微软雅黑", 10),
            height=6
        )
        self.history_listbox.pack(fill=tk.X, pady=5)
        
        # 填充历史记录
        for i, move in enumerate(self.history, 1):
            player = "黑" if i % 2 == 1 else "白"
            pos = move.get("position", [0, 0])
            self.history_listbox.insert(tk.END, f"{i}. {player}: ({pos[0]}, {pos[1]})")
    
    def _prev_step(self):
        """上一步"""
        if self.current_step > 0:
            self.current_step -= 1
            self._update_display()
    
    def _next_step(self):
        """下一步"""
        if self.current_step < len(self.history):
            self.current_step += 1
            self._update_display()
    
    def _goto_end(self):
        """跳转到结局"""
        self.current_step = len(self.history)
        self._update_display()
    
    def _reset(self):
        """重置"""
        self.current_step = 0
        self._update_display()
    
    def _update_display(self):
        """更新显示"""
        step = self.current_step
        self.step_label.config(text=f"第 {step} / {len(self.history)} 步")
        
        # 清空棋盘
        self.board_canvas.delete("all")
        
        # 绘制棋盘网格
        self._draw_board_grid()
        
        # 绘制棋子和热力图
        if step > 0:
            move = self.history[step - 1]
            input_vector = move.get("input", [])
            
            # 绘制热力图
            self._draw_heatmaps(step, move)
            
            # 绘制棋子
            if self.show_board.get() and len(input_vector) >= 225:
                self._draw_pieces(input_vector, move)
            
            # 更新信息文本
            self._update_info_text(step, move)
        else:
            # 步骤 0 时清除热力图（显示空白棋盘）
            self._clear_heatmaps()
            # 清除信息文本
            self._clear_info_text()
    
    def _draw_board_grid(self):
        """绘制棋盘网格"""
        margin = 25
        cell_size = 30
        
        for i in range(15):
            y = margin + i * cell_size
            self.board_canvas.create_line(
                margin, y, margin + 14 * cell_size, y,
                fill="#8B4513", width=1
            )
            
            x = margin + i * cell_size
            self.board_canvas.create_line(
                x, margin, x, margin + 14 * cell_size,
                fill="#8B4513", width=1
            )
    
    def _draw_heatmaps(self, step, move):
        """绘制热力图

        MCTS 对局时优先使用 mcts_probs（访问计数归一化），
        其次 mcts_q_values（[-1,1] Q 值），最后 output（模型原始输出）。
        """
        # AI 热力图
        ai_rewards = []
        mcts_probs = move.get("mcts_probs", [])
        if len(mcts_probs) >= 225:
            total = sum(mcts_probs)
            if total > 0:
                ai_rewards = [c / total for c in mcts_probs[:225]]
            else:
                ai_rewards = [1.0 / 225] * 225
        else:
            mcts_q = move.get("mcts_q_values", [])
            if len(mcts_q) >= 225:
                # Q 值 [-1, 1] → [0, 1] 方便热力图渲染
                ai_rewards = [(q + 1.0) / 2.0 for q in mcts_q[:225]]
            else:
                output_vector = move.get("output", [])
                if len(output_vector) >= 225:
                    ai_rewards = list(output_vector[:225])
        if ai_rewards:
            self.ai_heatmap.draw_heatmap(ai_rewards, self._on_cell_hover)
        
        # Sample 热力图
        if self.sample_heatmap and self.sample_data and step > 0 and step - 1 < len(self.sample_data):
            sample_step = self.sample_data[step - 1]
            output_vector = sample_step.get("output", [])
            if len(output_vector) >= 225:
                rewards = list(output_vector[:225])
                self.sample_heatmap.draw_heatmap(rewards, self._on_cell_hover)
        
        # Sampling 热力图（第四十一次追加：温度变换后的采样概率）
        # 当有 value_evaluations 时优先展示 cnn_v10 原始评估值
        if self.sampling_heatmap:
            value_evals = move.get("value_evaluations", [])
            strategy = move.get("strategy", "")
            sampling_probs = move.get("sampling_probs", [])
            
            if len(value_evals) >= 225:
                # cnn_v10: 展示原始评估值（值域 [-1, 1] → 映射到 [0, 1]）
                evals_list = [(v + 1.0) / 2.0 for v in value_evals[:225]]
                self.sampling_heatmap.draw_heatmap(evals_list, self._on_cell_hover)
                self.sampling_heatmap.set_title("Value Evals")
                self.sampling_label.config(text="Value Evals", fg="#8B4513")
            elif strategy == "sampling" and len(sampling_probs) >= 225:
                probs_list = list(sampling_probs[:225])
                self.sampling_heatmap.draw_heatmap(probs_list, self._on_cell_hover)
                self.sampling_heatmap.set_title("Sampling")
                self.sampling_label.config(text="Sampling", fg="#228B22")
            else:
                self.sampling_heatmap.canvas.delete("all")
                self.sampling_heatmap.set_title("Sampling")
                self.sampling_label.config(text="Sampling", fg="#228B22")
    
    def _draw_pieces(self, input_vector, move):
        """绘制棋子"""
        margin = 25
        cell_size = 30
        
        for i in range(15):
            for j in range(15):
                idx = i * 15 + j
                piece = input_vector[idx]
                if piece != 0:
                    x = margin + j * cell_size
                    y = margin + i * cell_size
                    color = "#2C2C2C" if piece == 1 else "#FFFAF0"
                    self.board_canvas.create_oval(
                        x-12, y-12, x+12, y+12,
                        fill=color, outline="#8B4513"
                    )
        
        # 高亮最后落子
        pos = move.get("position", [0, 0])
        x = margin + pos[1] * cell_size
        y = margin + pos[0] * cell_size
        self.board_canvas.create_oval(
            x-15, y-15, x+15, y+15,
            outline="#FF6B35", width=3
        )
    
    def _update_info_text(self, step, move):
        """更新信息文本"""
        self.info_text.delete(1.0, tk.END)
        
        self.info_text.insert(tk.END, f"=== 第 {step} 步 ===\n\n")
        
        # 显示输入/输出向量
        if self.show_input.get():
            input_vec = move.get("input", [])[:30]
            self.info_text.insert(tk.END, f"输入向量（前30个）:\n{input_vec}...\n")
            self.info_text.insert(tk.END, f"长度: {len(move.get('input', []))}\n\n")
        
        if self.show_output.get():
            output_vec = move.get("output", [])[:30]
            self.info_text.insert(tk.END, f"输出向量（前30个）:\n{output_vec}...\n")
            self.info_text.insert(tk.END, f"长度: {len(move.get('output', []))}\n\n")
        
        # 显示 Sample 数据
        if self.sample_data and step > 0 and step <= len(self.sample_data):
            sample_step = self.sample_data[step - 1]
            self.info_text.insert(tk.END, "--- Sample 数据 ---\n")
            
            pos = move.get("position", [0, 0])
            output_vec = sample_step.get("output", [])
            
            # 按行优先索引定位实际落子位置的 reward
            idx = pos[0] * 15 + pos[1]
            if 0 <= idx < len(output_vec):
                reward = output_vec[idx]
                self.info_text.insert(tk.END, f"实际落子 reward: {reward}\n")
        
        pos = move.get("position", [0, 0])
        player = "黑" if step % 2 == 1 else "白"
        self.info_text.insert(tk.END, f"\n落子方: {player}\n")
        self.info_text.insert(tk.END, f"落子位置: ({pos[0]}, {pos[1]})\n")

        # cnn_v10 价值头信息
        state_val = move.get("state_value")
        if state_val is not None:
            self.info_text.insert(tk.END, f"\n--- cnn_v10 价值评估 ---\n")
            self.info_text.insert(tk.END, f"当前局面价值: {state_val:+.4f}\n")
            value_evals = move.get("value_evaluations", [])
            if len(value_evals) >= 225:
                # Top-5 按原始评估值排序
                indexed = [(i, value_evals[i]) for i in range(225)]
                indexed.sort(key=lambda x: -x[1])
                self.info_text.insert(tk.END, f"Top-5 着法 (原始评估值):\n")
                for idx, val in indexed[:5]:
                    r, c = idx // 15, idx % 15
                    marker = " ◀ 实际落子" if [r, c] == pos else ""
                    self.info_text.insert(tk.END, f"  ({r:2d},{c:2d}): {val:+.4f}{marker}\n")

        # MCTS 信息
        mcts_probs = move.get("mcts_probs", [])
        mcts_q = move.get("mcts_q_values", [])
        has_mcts = len(mcts_probs) >= 225 or len(mcts_q) >= 225
        if has_mcts:
            mcts_value = move.get("mcts_value")
            self.info_text.insert(tk.END, f"\n--- MCTS 搜索结果 ---\n")
            if mcts_value is not None:
                self.info_text.insert(tk.END, f"根节点价值: {mcts_value:+.4f}\n")
            if len(mcts_probs) >= 225:
                # 按访问计数排序
                indexed = [(i, mcts_probs[i]) for i in range(225)]
                indexed.sort(key=lambda x: -x[1])
                self.info_text.insert(tk.END, f"Top-5 着法 (访问占比):\n")
                total = sum(mcts_probs)
                for idx, count in indexed[:5]:
                    r, c = idx // 15, idx % 15
                    marker = " ◀ 实际落子" if [r, c] == pos else ""
                    share = count / total * 100 if total > 0 else 0.0
                    q_val = mcts_q[idx] if idx < len(mcts_q) else 0.0
                    self.info_text.insert(tk.END, f"  ({r:2d},{c:2d}): {share:5.1f}% (访问 {count} 次, Q={q_val:+.4f}){marker}\n")
            else:
                # 按 Q 值排序（兜底）
                indexed = [(i, mcts_q[i]) for i in range(225)]
                indexed.sort(key=lambda x: -x[1])
                self.info_text.insert(tk.END, f"Top-5 着法 (按 Q 值):\n")
                for idx, q in indexed[:5]:
                    r, c = idx // 15, idx % 15
                    marker = " ◀ 实际落子" if [r, c] == pos else ""
                    self.info_text.insert(tk.END, f"  ({r:2d},{c:2d}): Q={q:+.4f}{marker}\n")

        # 温度采样信息（第四十一次追加：展示落子策略和采样参数）
        strategy = move.get("strategy", "argmax")
        self.info_text.insert(tk.END, f"\n--- 温度采样 ---\n")
        self.info_text.insert(tk.END, f"策略: {strategy}\n")
        if strategy == "sampling":
            self.info_text.insert(tk.END, f"温度: {move.get('temperature', 'N/A')}\n")
            self.info_text.insert(tk.END, f"Top-K: {move.get('top_k', 'None')}\n")
    
    def _clear_heatmaps(self):
        """清除热力图（步骤归零时调用）"""
        self.ai_heatmap.canvas.delete("all")
        if self.sample_heatmap:
            self.sample_heatmap.canvas.delete("all")
        if self.sampling_heatmap:
            self.sampling_heatmap.canvas.delete("all")
    
    def _clear_info_text(self):
        """清除信息文本（步骤归零时调用）"""
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, "=== 空白棋盘 ===\n\n重置到初始状态。\n")
    
    def _on_cell_hover(self, reward, row, col, x_root, y_root):
        """处理格子悬停"""
        # 创建 tooltip
        pass  # 可以在这里添加 tooltip 显示


class ReplayWindow:
    """独立的回放窗口 - 用于显示完整回放界面"""
    
    COLOR_BG = COLORS["bg"]
    
    def __init__(self, record_file, sample_file=None):
        """
        初始化回放窗口
        
        Args:
            record_file: record.json 文件路径
            sample_file: sample.json 文件路径（可选）
        """
        self.record_file = record_file
        self.sample_file = sample_file
        
        # 创建窗口
        self.root = tk.Tk()
        
        # 设置窗口标题
        if sample_file and os.path.exists(sample_file):
            self.root.title("AI 内战回放 (含 Sample 数据)")
        else:
            self.root.title("AI 内战回放")
        
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        self.root.configure(bg=self.COLOR_BG)
        
        # 创建回放查看器
        self.viewer = ReplayViewer(self.root, record_file, sample_file)
    
    def run(self):
        """运行回放窗口"""
        self.root.mainloop()
    
    @staticmethod
    def show(record_file, sample_file=None):
        """
        静态方法：显示回放窗口
        
        Args:
            record_file: record.json 文件路径
            sample_file: sample.json 文件路径（可选）
        """
        window = ReplayWindow(record_file, sample_file)
        window.run()
