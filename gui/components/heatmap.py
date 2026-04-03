"""
热力图组件 - 负责 AI/Sample 热力图的绘制和交互
"""
import tkinter as tk


class HeatmapCanvas:
    """热力图画布组件"""
    
    # 9种正色定义 (RGB)
    PRIMARY_COLORS = [
        (0, 0, 0),       # 0: 黑
        (128, 0, 128),   # 1: 紫
        (0, 0, 255),     # 2: 蓝
        (0, 128, 0),     # 3: 绿
        (255, 255, 255), # 4: 白
        (255, 255, 0),   # 5: 黄
        (255, 165, 0),   # 6: 橙
        (255, 192, 203), # 7: 粉
        (255, 0, 0),     # 8: 红
    ]
    
    # 固定最大值（评价器的 max_reward）
    GLOBAL_MAX_ABS = 1000000
    
    def __init__(self, parent, title="热力图", width=280, height=280):
        """
        初始化热力图组件
        
        Args:
            parent: 父容器
            title: 标题
            width: 画布宽度
            height: 画布高度
        """
        self.parent = parent
        self.title = title
        self.width = width
        self.height = height
        
        # 热力图参数
        self.cell_size = 16
        self.margin = 20
        self.board_size = 15
        
        # 创建框架
        self.frame = tk.Frame(parent, bg="#F5F5DC")
        
        # 创建画布
        self.canvas = tk.Canvas(
            self.frame,
            width=width,
            height=height,
            bg="#E3C16F",
            highlightthickness=1,
            highlightbackground="#8B4513"
        )
        self.canvas.pack()
        
        # Tooltip 窗口
        self.tooltip_window = None
    
    def pack(self, **kwargs):
        """包装布局"""
        self.frame.pack(**kwargs)
    
    def pack_side(self, side=tk.LEFT, **kwargs):
        """侧边布局"""
        self.frame.pack(side=side, **kwargs)
    
    def clear(self):
        """清空热力图"""
        self.canvas.delete("all")
    
    def draw_heatmap(self, rewards, on_cell_hover=None):
        """
        绘制热力图
        
        Args:
            rewards: 225个reward值的列表
            on_cell_hover: 鼠标悬停回调函数，参数为 (reward, row, col)
        """
        self.clear()
        
        if len(rewards) != 225:
            return
        
        for i in range(225):
            row = i // 15
            col = i % 15
            reward = rewards[i]
            
            x = self.margin + col * self.cell_size
            y = self.margin + row * self.cell_size
            
            # 获取颜色
            color = self._get_reward_color(reward)
            
            # 绘制格子
            rect = self.canvas.create_rectangle(
                x-7, y-7, x+8, y+8,
                fill=color, outline="#D2691E", width=1
            )
            
            # 绑定悬停事件
            if on_cell_hover:
                self.canvas.tag_bind(rect, "<Enter>", 
                    lambda e, r=reward, rrow=row, c=col: self._show_tooltip(e, r, rrow, c, on_cell_hover))
                self.canvas.tag_bind(rect, "<Leave>", 
                    lambda e: self._hide_tooltip())
    
    def _get_reward_color(self, reward):
        """
        根据 reward 值返回颜色
        9种正色: 黑、紫、蓝、绿、白、黄、橙、粉、红
        相邻正色之间分5档，共40档
        reward=0 对应白色，|reward|越小越接近白色
        """
        max_abs = self.GLOBAL_MAX_ABS
        
        # 将 reward 归一化到 [-1, 1]
        normalized = reward / max_abs
        
        # 映射到 40 档 (8个区间 * 5档 = 40档)
        level = (normalized + 1) * 20  # [-1,1] -> [0,40]
        
        # 限制范围
        if level < 0:
            level = 0
        elif level >= 40:
            level = 39
        
        # 计算在哪个区间和该区间内的位置
        zone = int(level) // 5  # 0-7，表示8个区间
        t = (int(level) % 5) / 4.0  # 0.0, 0.25, 0.5, 0.75, 1.0
        
        # 获取区间的两个端点颜色
        c1 = self.PRIMARY_COLORS[zone]
        c2 = self.PRIMARY_COLORS[zone + 1]
        
        # 线性插值
        r = int(c1[0] + (c2[0] - c1[0]) * t)
        g = int(c1[1] + (c2[1] - c1[1]) * t)
        b = int(c1[2] + (c2[2] - c1[2]) * t)
        
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def _show_tooltip(self, event, reward, row, col, callback):
        """显示 tooltip，在鼠标位置弹出格子的 reward 信息"""
        # 隐藏之前的 tooltip
        self._hide_tooltip()
        
        # 直接创建 tooltip 显示 reward 值
        text = f"({row}, {col})  reward: {reward}"
        self.create_tooltip(event.x_root, event.y_root, text)
        
        # 调用外部回调（供扩展）
        if callback:
            callback(reward, row, col, event.x_root, event.y_root)
    
    def _hide_tooltip(self):
        """隐藏 tooltip"""
        if self.tooltip_window and self.tooltip_window.winfo_exists():
            self.tooltip_window.destroy()
            self.tooltip_window = None
    
    def create_tooltip(self, x, y, text):
        """创建 tooltip 窗口"""
        self._hide_tooltip()
        
        self.tooltip_window = tk.Toplevel(self.parent)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x + 15}+{y + 15}")
        self.tooltip_window.configure(bg="#FFF8DC")
        
        label = tk.Label(
            self.tooltip_window,
            text=text,
            font=("微软雅黑", 9),
            bg="#FFF8DC",
            fg="#5D4E37",
            relief=tk.SOLID,
            bd=1,
            padx=8,
            pady=5
        )
        label.pack()


class HeatmapLegend:
    """热力图颜色标尺组件"""
    
    # 9种正色
    COLORS_FULL = [
        ("黑", "#000000"),
        ("紫", "#800080"),
        ("蓝", "#0000FF"),
        ("绿", "#008000"),
        ("白", "#FFFFFF"),
        ("黄", "#FFFF00"),
        ("橙", "#FFA500"),
        ("粉", "#FFC0CB"),
        ("红", "#FF0000"),
    ]
    
    # 基础颜色定义
    BASE_COLORS = [
        (0, 0, 0),       # 黑
        (128, 0, 128),   # 紫
        (0, 0, 255),     # 蓝
        (0, 128, 0),     # 绿
        (255, 255, 255), # 白
        (255, 255, 0),   # 黄
        (255, 165, 0),   # 橙
        (255, 192, 203), # 粉
        (255, 0, 0),     # 红
    ]
    
    def __init__(self, parent):
        """初始化颜色标尺"""
        self.parent = parent
        
        # 创建框架
        self.frame = tk.Frame(parent, bg="#F5F5DC")
        
        # 标题
        tk.Label(
            self.frame,
            text="颜色标尺 (min → max):",
            font=("微软雅黑", 9, "bold"),
            bg="#F5F5DC",
            fg="#5D4E37"
        ).pack(anchor=tk.W, pady=(0, 5))
        
        # 生成所有渐变颜色
        all_colors = self._generate_gradient_colors()
        
        # 显示渐变条
        gradient_bar = tk.Frame(self.frame, bg="#F5F5DC")
        gradient_bar.pack(fill=tk.X)
        
        for color in all_colors:
            tk.Frame(
                gradient_bar,
                bg=color,
                width=6,
                height=20
            ).pack(side=tk.LEFT)
        
        # 显示颜色标签
        labels_frame = tk.Frame(self.frame, bg="#F5F5DC")
        labels_frame.pack(fill=tk.X, pady=(2, 0))
        
        for name, color in self.COLORS_FULL:
            label_frame = tk.Frame(labels_frame, bg="#F5F5DC")
            label_frame.pack(side=tk.LEFT, expand=True)
            
            tk.Frame(
                label_frame,
                bg=color,
                width=12,
                height=12
            ).pack()
            
            tk.Label(
                label_frame,
                text=name,
                font=("微软雅黑", 7),
                bg="#F5F5DC",
                fg="#5D4E37"
            ).pack()
    
    def _generate_gradient_colors(self):
        """生成所有渐变颜色"""
        all_colors = []
        for i in range(len(self.BASE_COLORS) - 1):
            c1 = self.BASE_COLORS[i]
            c2 = self.BASE_COLORS[i + 1]
            for level in range(5):
                t = level / 4.0
                r = int(c1[0] + (c2[0] - c1[0]) * t)
                g = int(c1[1] + (c2[1] - c1[1]) * t)
                b = int(c1[2] + (c2[2] - c1[2]) * t)
                all_colors.append(f"#{r:02x}{g:02x}{b:02x}")
        return all_colors
    
    def pack(self, **kwargs):
        """包装布局"""
        self.frame.pack(**kwargs)
