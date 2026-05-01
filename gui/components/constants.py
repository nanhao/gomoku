"""GUI 常量定义"""

# 颜色配置 - 森林/小动物主题
COLORS = {
    # 背景色
    "bg": "#F5F5DC",           # 米色背景
    "board": "#E3C16F",        # 棋盘颜色
    
    # 文字颜色
    "text": "#5D4E37",         # 深棕色文字
    "text_light": "#8B7355",   # 浅棕色文字
    
    # 边框/线条
    "border": "#8B4513",       # 棋盘线颜色
    "highlight": "#FF6B35",    # 高亮颜色
    
    # 棋子颜色
    "black_piece": "#2C2C2C",  # 黑棋
    "white_piece": "#FFFAF0",  # 白棋
    
    # 按钮颜色
    "bear": "#D2691E",         # 小熊棕
    "rabbit": "#FFB6C1",       # 小兔粉
    "squirrel": "#CD853F",     # 松鼠褐
    
    # 热力图颜色
    "heatmap_outline": "#D2691E",  # 热力图边框
}

# 尺寸配置
DIMENSIONS = {
    # 窗口尺寸
    "window_width": 900,
    "window_height": 750,
    "mode_selector_width": 400,
    "mode_selector_height": 600,
    
    # 棋盘尺寸
    "canvas_size": 500,
    "cell_size": 30,
    "board_margin": 25,
    "board_size": 15,
    
    # 棋子尺寸
    "piece_radius": 12,
    "highlight_radius": 15,
    "heatmap_size": 12,
}

# 字体配置
FONTS = {
    "title": ("微软雅黑", 20, "bold"),
    "subtitle": ("微软雅黑", 14),
    "button": ("微软雅黑", 12, "bold"),
    "normal": ("微软雅黑", 11),
    "small": ("微软雅黑", 9),
    "mono": ("Consolas", 9),
}

# 动物表情符号（森林/小动物主题）
ANIMALS = {
    "black": "🐻",
    "white": "🐰",
    "last_move": "🐾",
}
