"""测试 GomokuController 接口"""

import sys
import os
import shutil

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from game.controller import GomokuController
from game.save import GomokuSave


class TestGomokuController:
    """测试 GomokuController 类"""
    
    def setup_method(self):
        """每个测试方法前执行"""
        # 使用独立的测试存档目录
        self.test_save_dir = os.path.join(os.path.dirname(__file__), 'game', 'save')
        # 清理测试目录中的旧存档
        if os.path.exists(self.test_save_dir):
            for filename in os.listdir(self.test_save_dir):
                if filename.endswith('.json'):
                    os.remove(os.path.join(self.test_save_dir, filename))
        
        self.controller = GomokuController()
        # 替换为测试用的存档目录
        self.controller.save = GomokuSave(self.test_save_dir)
    
    def test_new_game(self):
        """测试新建游戏"""
        # 新建游戏
        result = self.controller.new_game("测试游戏1")
        
        # 验证结果
        assert result["status"] == "success"
        assert result["state"]["current_player"] == 1
        assert result["state"]["winner"] == 0
        assert len(result["state"]["history"]) == 0
        assert result["state"]["name"] == "测试游戏1"
    
    def test_make_move_valid(self):
        """测试有效落子"""
        self.controller.new_game("测试游戏2")
        result = self.controller.make_move(7, 7)
        
        assert result["status"] == "success"
        assert result["winner"] == 0  # 未结束
        assert result["current_player"] == 2  # 切换为白方
    
    def test_make_move_invalid_occupied(self):
        """测试重复落子"""
        self.controller.new_game("测试游戏3")
        self.controller.make_move(7, 7)
        result = self.controller.make_move(7, 7)  # 重复落子
        
        assert "error" in result
        assert "已被占用" in result["error"]
    
    def test_make_move_invalid_out_of_bounds(self):
        """测试越界落子"""
        self.controller.new_game("测试游戏4")
        result = self.controller.make_move(15, 15)  # 越界
        
        assert "error" in result
        assert "超出范围" in result["error"]
    
    def test_win_horizontal(self):
        """测试横向胜利"""
        self.controller.new_game("测试游戏5")
        # 黑棋横向连五 (7,5) (7,6) (7,7) (7,8) (7,9)
        moves = [(7, 5), (0, 0), (7, 6), (0, 1), (7, 7), (0, 2), (7, 8), (0, 3), (7, 9)]
        for x, y in moves:
            result = self.controller.make_move(x, y)
        
        assert result["winner"] == 1  # 黑胜
    
    def test_win_vertical(self):
        """测试纵向胜利"""
        self.controller.new_game("测试游戏6")
        # 白棋纵向连五 (5,7) (6,7) (7,7) (8,7) (9,7)
        # 黑方分散下在棋盘上，避免形成任何连线
        moves = [(14, 0), (5, 7), (14, 2), (6, 7), (14, 4), (7, 7), (14, 6), (8, 7), (14, 8), (9, 7)]
        for x, y in moves:
            result = self.controller.make_move(x, y)
        
        assert result["winner"] == 2  # 白胜
    
    def test_win_diagonal(self):
        """测试对角线胜利"""
        self.controller.new_game("测试游戏7")
        # 黑棋主对角线连五 (5,5) (6,6) (7,7) (8,8) (9,9)
        moves = [(5, 5), (0, 1), (6, 6), (0, 2), (7, 7), (0, 3), (8, 8), (0, 4), (9, 9)]
        for x, y in moves:
            result = self.controller.make_move(x, y)
        
        assert result["winner"] == 1  # 黑胜
    
    def test_new_game_duplicate_name(self):
        """测试新建游戏时存档名已存在"""
        # 先创建一个游戏
        result = self.controller.new_game("测试游戏8")
        assert result["status"] == "success"
        
        # 再创建同名游戏，应该报错
        result = self.controller.new_game("测试游戏8")
        assert "error" in result
        assert "已存在" in result["error"]
    
    def test_draw(self):
        """测试平局"""
        self.controller.new_game("测试游戏9")
        # 模拟平局：填满棋盘但无五连（这里简化为一个小棋盘测试）
        # 实际测试需要更复杂的落子序列
        # 暂时留空，实现时再补充具体落子序列
        pass
    
    def test_load_game(self):
        """测试加载存档"""
        self.controller.new_game("测试游戏10")
        self.controller.make_move(7, 7)
        
        # 重新加载
        result = self.controller.load_game("测试游戏10")
        assert result["status"] == "success"
        state = result["state"]
        assert len(state["history"]) == 1
        assert state["history"][0]["x"] == 7
        assert state["history"][0]["y"] == 7
    
    def test_list_saves(self):
        """测试列出存档"""
        self.controller.new_game("测试游戏A")
        self.controller.new_game("测试游戏B")
        
        saves = self.controller.list_saves()
        assert "测试游戏A" in saves
        assert "测试游戏B" in saves
    
    def test_game_over_cannot_move(self):
        """测试游戏结束后不能落子"""
        self.controller.new_game("测试游戏C")
        # 模拟胜利
        moves = [(7, 5), (0, 0), (7, 6), (0, 1), (7, 7), (0, 2), (7, 8), (0, 3), (7, 9)]
        for x, y in moves:
            self.controller.make_move(x, y)
        
        # 游戏结束后尝试落子
        result = self.controller.make_move(0, 0)
        assert "error" in result
        assert "游戏已结束" in result["error"]
    
    def test_get_state_full(self):
        """测试获取完整状态"""
        self.controller.new_game("测试游戏D")
        self.controller.make_move(7, 7)
        
        state = self.controller.get_state(short=False)
        assert "name" in state
        assert "start_time" in state
        assert "board" in state
        assert "current_player" in state
        assert "history" in state
        assert "winner" in state
        assert len(state["history"]) == 1
    
    def test_get_state_short(self):
        """测试获取简化状态"""
        self.controller.new_game("测试游戏E")
        self.controller.make_move(7, 7)
        
        state = self.controller.get_state(short=True)
        assert "board" in state
        assert "current_player" in state
        assert "winner" in state
        assert "name" not in state
        assert "start_time" not in state
        assert "history" not in state
    
    def test_get_state_json_full(self):
        """测试获取完整JSON状态"""
        self.controller.new_game("测试游戏F")
        json_str = self.controller.get_state_json(short=False)
        
        # 验证是有效的JSON字符串
        import json
        state = json.loads(json_str)
        assert "name" in state
        assert "board" in state
    
    def test_get_state_json_short(self):
        """测试获取简化JSON状态"""
        self.controller.new_game("测试游戏G")
        json_str = self.controller.get_state_json(short=True)
        
        # 验证是有效的JSON字符串
        import json
        state = json.loads(json_str)
        assert "board" in state
        assert "name" not in state
