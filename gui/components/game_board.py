"""
棋盘组件 - 负责棋盘的绘制和交互
"""
import tkinter as tk


class GameBoard:
    """五子棋棋盘组件"""
    
    # 棋盘配置
    BOARD_SIZE = 15
    CELL_SIZE = 42
    BOARD_MARGIN = 50
    
    # 配色方案
    COLOR_BOARD = "#E3C16F"        # 棋盘背景色（木质原色）
    COLOR_LINE = "#8B4513"         # 线条颜色（深棕色）
    COLOR_BLACK = "#2C2C2C"        # 黑子颜色
    COLOR_WHITE = "#FFFAF0"        # 白子颜色
    COLOR_LAST_MOVE = "#FF6B35"    # 最后落子标记颜色（橙色）
    
    def __init__(self, parent, on_click_callback=None):
        """
        初始化棋盘组件
        
        Args:
            parent: 父容器
            on_click_callback: 点击回调函数，参数为 (row, col)
        """
        self.parent = parent
        self.on_click_callback = on_click_callback
        
        # 计算画布尺寸
        self.canvas_size = self.BOARD_MARGIN * 2 + self.CELL_SIZE * (self.BOARD_SIZE - 1)
        
        # 创建画布
        self.canvas = tk.Canvas(
            parent,
            width=self.canvas_size,
            height=self.canvas_size,
            bg=self.COLOR_BOARD,
            highlightthickness=2,
            highlightbackground="#8B4513"
        )
        
        # 绑定点击事件
        if on_click_callback:
            self.canvas.bind("<Button-1>", self._on_click)
        
        # 初始绘制
        self.draw_board()
    
    def pack(self, **kwargs):
        """包装布局方法"""
        self.canvas.pack(**kwargs)
    
    def grid(self, **kwargs):
        """网格布局方法"""
        self.canvas.grid(**kwargs)
    
    def draw_board(self):
        """绘制空棋盘"""
        self.canvas.delete("all")
        
        # 绘制网格线
        for i in range(self.BOARD_SIZE):
            # 横线
            y = self.BOARD_MARGIN + i * self.CELL_SIZE
            self.canvas.create_line(
                self.BOARD_MARGIN, y,
                self.BOARD_MARGIN + (self.BOARD_SIZE - 1) * self.CELL_SIZE, y,
                fill=self.COLOR_LINE, width=1
            )
            
            # 竖线
            x = self.BOARD_MARGIN + i * self.CELL_SIZE
            self.canvas.create_line(
                x, self.BOARD_MARGIN,
                x, self.BOARD_MARGIN + (self.BOARD_SIZE - 1) * self.CELL_SIZE,
                fill=self.COLOR_LINE, width=1
            )
        
        # 绘制星位（天元及四角）
        star_positions = [
            (3, 3), (3, 7), (3, 11),
            (7, 3), (7, 7), (7, 11),
            (11, 3), (11, 7), (11, 11)
        ]
        for row, col in star_positions:
            x = self.BOARD_MARGIN + col * self.CELL_SIZE
            y = self.BOARD_MARGIN + row * self.CELL_SIZE
            self.canvas.create_oval(
                x-3, y-3, x+3, y+3,
                fill=self.COLOR_LINE, outline=self.COLOR_LINE
            )
    
    def draw_pieces(self, board_state, last_move=None):
        """
        绘制棋子
        
        Args:
            board_state: 15x15 的二维数组，0=空, 1=黑, 2=白
            last_move: 最后落子位置 (row, col)，可选
        """
        # 先清除棋子（保留棋盘线）
        self.canvas.delete("piece")
        self.canvas.delete("last_move")
        
        # 绘制棋子
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                piece = board_state[row][col]
                if piece != 0:
                    self._draw_piece(row, col, piece)
        
        # 高亮最后落子
        if last_move:
            self._highlight_last_move(last_move[0], last_move[1])
    
    def _draw_piece(self, row, col, piece_type):
        """绘制单个棋子"""
        x = self.BOARD_MARGIN + col * self.CELL_SIZE
        y = self.BOARD_MARGIN + row * self.CELL_SIZE
        
        color = self.COLOR_BLACK if piece_type == 1 else self.COLOR_WHITE
        tag = "piece"
        
        # 绘制棋子（带立体效果）
        # 外圈阴影
        self.canvas.create_oval(
            x-18, y-18, x+18, y+18,
            fill="#5D4E37", outline="", tags=tag
        )
        # 主棋子
        self.canvas.create_oval(
            x-16, y-16, x+16, y+16,
            fill=color, outline="#5D4E37", width=1, tags=tag
        )
    
    def _highlight_last_move(self, row, col):
        """高亮最后落子位置"""
        x = self.BOARD_MARGIN + col * self.CELL_SIZE
        y = self.BOARD_MARGIN + row * self.CELL_SIZE
        
        # 绘制红色圆圈标记
        self.canvas.create_oval(
            x-20, y-20, x+20, y+20,
            outline=self.COLOR_LAST_MOVE, width=3, tags="last_move"
        )
    
    def _on_click(self, event):
        """处理点击事件"""
        # 计算点击的格子位置
        x = event.x - self.BOARD_MARGIN + self.CELL_SIZE // 2
        y = event.y - self.BOARD_MARGIN + self.CELL_SIZE // 2
        
        col = int(x / self.CELL_SIZE)
        row = int(y / self.CELL_SIZE)
        
        # 检查边界
        if 0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE:
            if self.on_click_callback:
                self.on_click_callback(row, col)
    
    def get_canvas(self):
        """获取画布对象"""
        return self.canvas
