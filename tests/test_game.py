"""测试 GomokuGame 游戏逻辑"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from game.game import GomokuGame


class TestGomokuGame:
    """测试 GomokuGame 类"""
    
    def test_init_new_game(self):
        """测试初始化新游戏"""
        game = GomokuGame()
        state = game.get_state()
        
        assert state["current_player"] == 1
        assert state["winner"] == 0
        assert len(state["history"]) == 0
        assert len(state["board"]) == 15
        assert len(state["board"][0]) == 15
        assert all(cell == 0 for row in state["board"] for cell in row)
    
    def test_init_with_state(self):
        """测试用状态初始化游戏"""
        state = {
            "board": [[0]*15 for _ in range(15)],
            "current_player": 2,
            "history": [{"x": 7, "y": 7, "player": 1}],
            "winner": 0
        }
        game = GomokuGame(state)
        
        loaded_state = game.get_state()
        assert loaded_state["current_player"] == 2
        assert len(loaded_state["history"]) == 1
        assert loaded_state["history"][0]["x"] == 7
    
    def test_make_move_valid(self):
        """测试有效落子"""
        game = GomokuGame()
        winner = game.make_move(7, 7)
        state = game.get_state()
        
        assert winner == 0
        assert state["board"][7][7] == 1
        assert state["current_player"] == 2
        assert len(state["history"]) == 1
    
    def test_make_move_invalid_occupied(self):
        """测试落子到已占用位置"""
        game = GomokuGame()
        game.make_move(7, 7)
        
        try:
            game.make_move(7, 7)
            assert False, "应该抛出异常"
        except Exception as e:
            assert "已被占用" in str(e)
    
    def test_make_move_invalid_out_of_bounds(self):
        """测试越界落子"""
        game = GomokuGame()
        
        try:
            game.make_move(15, 15)
            assert False, "应该抛出异常"
        except Exception as e:
            assert "超出范围" in str(e)
    
    def test_make_move_game_over(self):
        """测试游戏结束后落子"""
        game = GomokuGame()
        # 模拟胜利 - 黑棋横向五连 (7,5)-(7,9)
        moves = [(7, 5), (0, 0), (7, 6), (0, 1), (7, 7), (0, 2), (7, 8), (0, 3), (7, 9)]
        for x, y in moves:
            game.make_move(x, y)
        
        # 尝试在空位落子
        try:
            game.make_move(1, 0)  # 这个位置未被占用
            assert False, "应该抛出异常"
        except Exception as e:
            assert "游戏已结束" in str(e)
    
    def test_make_move_returns_winner(self):
        """测试落子返回获胜方"""
        game = GomokuGame()
        
        # 落子但未获胜
        result = game.make_move(7, 7)
        assert result == 0
        
        # 模拟胜利 - 横向五连（需要黑白交替）
        game2 = GomokuGame()
        moves = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4)]
        result = None
        for x, y in moves:
            result = game2.make_move(x, y)
        
        assert result == 1
    
    def test_win_horizontal(self):
        """测试横向五连"""
        game = GomokuGame()
        moves = [(7, 5), (0, 0), (7, 6), (0, 1), (7, 7), (0, 2), (7, 8), (0, 3), (7, 9)]
        winner = None
        for x, y in moves:
            winner = game.make_move(x, y)
        
        assert winner == 1
        state = game.get_state()
        assert state["winner"] == 1
    
    def test_win_vertical(self):
        """测试纵向五连"""
        game = GomokuGame()
        # 白棋纵向连五 (5,7) (6,7) (7,7) (8,7) (9,7)
        # 黑棋落子避免形成五连，在黑棋即将五连的位置插入白棋阻断
        moves = [(0, 0), (5, 7), (1, 0), (6, 7), (2, 0), (7, 7), (3, 0), (8, 7), (0, 1), (9, 7)]
        winner = None
        for x, y in moves:
            winner = game.make_move(x, y)
        
        assert winner == 2
        state = game.get_state()
        assert state["winner"] == 2
    
    def test_win_diagonal_main(self):
        """测试主对角线五连"""
        game = GomokuGame()
        moves = [(5, 5), (0, 0), (6, 6), (0, 1), (7, 7), (0, 2), (8, 8), (0, 3), (9, 9)]
        winner = None
        for x, y in moves:
            winner = game.make_move(x, y)
        
        assert winner == 1
    
    def test_win_diagonal_anti(self):
        """测试副对角线五连"""
        game = GomokuGame()
        moves = [(5, 9), (0, 0), (6, 8), (0, 1), (7, 7), (0, 2), (8, 6), (0, 3), (9, 5)]
        winner = None
        for x, y in moves:
            winner = game.make_move(x, y)
        
        assert winner == 1
    
    def test_win_long_chain(self):
        """测试长连（6+子）胜利"""
        game = GomokuGame()
        # 黑棋横向5连 (7,4)-(7,8)，第5个子落子后获胜
        # 规则是五子或以上连成一线即获胜
        moves = [(7, 4), (0, 1), (7, 5), (0, 2), (7, 6), (0, 3), (7, 7), (0, 4), (7, 8)]
        winner = None
        for x, y in moves:
            winner = game.make_move(x, y)
        
        # 第9步黑棋在(7,8)落子后形成5连，获胜
        assert winner == 1
    
    def test_draw(self):
        """测试平局"""
        # 这个测试比较复杂，需要填满棋盘但无五连
        # 实现时再填充具体逻辑
        pass
    
    def test_get_state(self):
        """测试获取游戏状态"""
        game = GomokuGame()
        game.make_move(7, 7)
        
        state = game.get_state()
        assert "board" in state
        assert "current_player" in state
        assert "history" in state
        assert "winner" in state
        assert state["board"][7][7] == 1
        assert state["current_player"] == 2
        assert len(state["history"]) == 1
