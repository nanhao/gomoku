"""游戏核心逻辑模块"""

from typing import List


def check_winner(board: List[int], last_move: int) -> int:
    """检查棋盘上最近落子是否形成五连，独立纯函数。

    供 MCTS 搜索器等外部调用，不依赖 GomokuGame 对象。

    Args:
        board: 225 维展平棋盘（按行展开），0=空, 1=黑, 2=白
        last_move: 最近落子的展平索引 0-224

    Returns:
        0=无胜者, 1=黑胜, 2=白胜
    """
    x = last_move // 15
    y = last_move % 15
    player = board[last_move]

    if player == 0:
        return 0

    # 四个方向：水平、垂直、主对角线、副对角线
    directions = [
        [(0, -1), (0, 1)],      # 水平
        [(-1, 0), (1, 0)],      # 垂直
        [(-1, -1), (1, 1)],     # 主对角线
        [(-1, 1), (1, -1)]      # 副对角线
    ]

    for direction in directions:
        count = 1  # 当前落子

        # 正向检测
        dx, dy = direction[0]
        nx, ny = x + dx, y + dy
        while 0 <= nx < 15 and 0 <= ny < 15 and board[nx * 15 + ny] == player:
            count += 1
            nx += dx
            ny += dy

        # 反向检测
        dx, dy = direction[1]
        nx, ny = x + dx, y + dy
        while 0 <= nx < 15 and 0 <= ny < 15 and board[nx * 15 + ny] == player:
            count += 1
            nx += dx
            ny += dy

        if count >= 5:
            return player

    return 0


def is_board_full(board: List[int]) -> bool:
    """检查棋盘是否已满（无空位），独立纯函数。

    Args:
        board: 225 维展平棋盘（按行展开），0=空, 1=黑, 2=白

    Returns:
        True 如果棋盘已满（所有 225 个位置均非空）
    """
    return all(cell != 0 for cell in board)


class GomokuGame:
    """游戏核心逻辑类"""
    
    def __init__(self, state=None):
        """初始化游戏。state 是游戏状态，dict 格式，为空则创建新游戏"""
        if state is None:
            # 创建新游戏
            self.board = [[0] * 15 for _ in range(15)]
            self.current_player = 1  # 黑方先手
            self.winner = 0  # 0=未结束, 1=黑胜, 2=白胜, 3=平局
            self.history = []
        else:
            # 从状态字典初始化（深拷贝防止外部修改影响游戏状态）
            self.board = [row[:] for row in state["board"]]
            self.current_player = state["current_player"]
            self.winner = state["winner"]
            self.history = [move.copy() for move in state.get("history", [])]
    
    def make_move(self, x, y):
        """在当前玩家落子，返回获胜方（1=黑, 2=白, 3=平局, 0=未结束）"""
        # 验证坐标范围
        if not (0 <= x < 15 and 0 <= y < 15):
            raise ValueError(f"坐标超出范围 (x={x}, y={y}, 要求 0-14)")
        
        # 验证位置是否为空
        if self.board[x][y] != 0:
            raise ValueError(f"位置 ({x}, {y}) 已被占用")
        
        # 验证游戏是否已结束
        if self.winner != 0:
            raise ValueError("游戏已结束")
        
        # 落子
        self.board[x][y] = self.current_player
        
        # 添加到历史记录
        self.history.append({"x": x, "y": y, "player": self.current_player})
        
        # 检查胜利条件
        if self._check_winner(x, y):
            self.winner = self.current_player
            return self.winner
        
        # 检查平局条件
        if self._is_draw():
            self.winner = 3  # 3=平局
            return self.winner
        
        # 切换玩家
        self.current_player = 2 if self.current_player == 1 else 1
        
        # 返回未结束
        return 0
    
    def get_state(self):
        """返回游戏状态，dict 格式（深拷贝防止外部修改影响游戏状态）"""
        return {
            "board": [row[:] for row in self.board],
            "current_player": self.current_player,
            "winner": self.winner,
            "history": [move.copy() for move in self.history]
        }
    
    def _check_winner(self, x, y):
        """检查胜利条件，以(x,y)为中心在4个方向检测

        委托给模块级纯函数 check_winner，传递展平棋盘。
        """
        # 将二维棋盘展平为一维列表
        flat_board = [cell for row in self.board for cell in row]
        flat_index = x * 15 + y
        return check_winner(flat_board, flat_index) != 0
    
    def _is_draw(self):
        """检查平局条件：棋盘已满且无获胜方

        委托给模块级纯函数 is_board_full。
        """
        if self.winner != 0:
            return False
        flat_board = [cell for row in self.board for cell in row]
        return is_board_full(flat_board)

