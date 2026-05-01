"""测试 GomokuController 接口"""

import sys
import os
import shutil
import tempfile
import unittest

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from game.controller import GomokuController
from game.game import GomokuGame
from game.save import GomokuSave


def _build_draw_residual_state(empty_cell=(14, 14)):
    """构造 224 格满 + 1 格空且保证无五连的平局残局 state。

    染色公式: color(i, j) = 1 + ((i + j // 2) % 2)。
    - 横向: 同一行 j 变化时，j//2 每 2 格 +1，色序 {i,i,~i,~i,i}，连续最多 2 同色。
    - 纵向: 同一列 i 变化时，色按 (i+常数) 交替，连续最多 1 同色。
    - 主/副对角: 逐位展开后连续最多 2 同色（运行时自检兜底）。

    空位由黑方 (color=1) 落下，调用 make_move(empty_cell) 即可触发平局分支。
    """
    board = [[0] * 15 for _ in range(15)]
    history = []
    for i in range(15):
        for j in range(15):
            if (i, j) == empty_cell:
                continue
            board[i][j] = 1 + ((i + j // 2) % 2)
            history.append({"x": i, "y": j, "player": board[i][j]})
    # 防御性自检: 若染色公式失效, 测试必须以 AssertionError 失败而不是静默通过
    for i in range(15):
        for j in range(15):
            if board[i][j] == 0:
                continue
            p = board[i][j]
            for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                count = 1
                for k in range(1, 5):
                    ni, nj = i + dx * k, j + dy * k
                    if not (0 <= ni < 15 and 0 <= nj < 15) or board[ni][nj] != p:
                        break
                    count += 1
                assert count < 5, (
                    f"平局残局构造失效: ({i},{j}) 方向 ({dx},{dy}) 出现 5 连同色 {p}"
                )
    return {
        "board": board,
        "current_player": 1,  # 空位由黑方落下触发平局
        "winner": 0,
        "history": history,
    }


class TestGomokuController:
    """测试 GomokuController 类"""
    
    def setup_method(self):
        """每个测试方法前执行"""
        # 使用临时目录，避免测试间相互干扰
        self._temp_root = tempfile.mkdtemp(prefix="test_controller_")
        self.test_save_dir = os.path.join(self._temp_root, "saves")
        self.controller = GomokuController()
        # 替换为测试用的存档目录
        self.controller.save = GomokuSave(self.test_save_dir)
    
    def teardown_method(self):
        """每个测试方法后执行：完整删除临时目录"""
        # 关闭游戏对象释放文件句柄
        if hasattr(self, 'controller') and self.controller is not None:
            self.controller.game = None
            self.controller.save = None
        if hasattr(self, '_temp_root') and os.path.exists(self._temp_root):
            try:
                shutil.rmtree(self._temp_root)
            except PermissionError:
                pass  # Windows 可能文件被占用，忽略清理错误
    
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
        """测试平局

        通过预置 224 格满 + 1 格空的确定性残局，
        验证 make_move 返回 game_over=True 且 winner=3（平局）。
        """
        self.controller.new_game("测试平局")

        # 注入预置残局，确保必走平局分支（原版条件断言会静默通过）
        empty = (14, 14)
        self.controller.game = GomokuGame(state=_build_draw_residual_state(empty))

        result = self.controller.make_move(*empty)

        assert result["status"] == "success"
        assert result["winner"] == 3
        assert result["game_over"] is True
        assert result["result"] == "平局"
        assert self.controller.game.winner == 3
        assert self.controller.is_draw() is True
        assert self.controller.is_over() is True
    
    def test_load_game(self):
        """测试加载存档
        
        make_move 已移除自动存档逻辑，所以加载时只能恢复到
        new_game 保存的初始状态（history 为空）。
        """
        self.controller.new_game("测试游戏10")
        self.controller.make_move(7, 7)
        
        # 重新加载（只能恢复到 new_game 保存的初始状态）
        result = self.controller.load_game("测试游戏10")
        assert result["status"] == "success"
        state = result["state"]
        # make_move 不再自动保存，加载后 history 为初始状态
        assert len(state["history"]) == 0
    
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

    def test_game_over_draw_cannot_move(self):
        """测试平局后不能落子"""
        self.controller.new_game("测试游戏平局")

        # 注入预置残局并走出平局状态
        empty = (14, 14)
        self.controller.game = GomokuGame(state=_build_draw_residual_state(empty))
        self.controller.make_move(*empty)
        assert self.controller.game.winner == 3

        # 平局后尝试再落子应返回错误（winner!=0 分支由 controller 先拦截）
        result = self.controller.make_move(0, 0)
        assert "error" in result
        assert "游戏已结束" in result["error"]

    def test_make_move_no_game(self):
        """测试无游戏时落子返回错误"""
        result = self.controller.make_move(7, 7)
        assert "error" in result
        assert "请先创建或加载游戏" in result["error"]

    def test_load_game_not_exists(self):
        """测试加载不存在的存档"""
        result = self.controller.load_game("不存在的存档")
        assert "error" in result

    def test_is_over_no_game(self):
        """测试无游戏时 is_over 返回 False"""
        assert self.controller.is_over() is False

    def test_is_draw_no_game(self):
        """测试无游戏时 is_draw 返回 False"""
        assert self.controller.is_draw() is False

    def test_get_current_player_no_game(self):
        """测试无游戏时 get_current_player 返回 None"""
        assert self.controller.get_current_player() is None

    def test_get_winner_no_game(self):
        """测试无游戏时 get_winner 返回 None"""
        assert self.controller.get_winner() is None

    def test_get_state_no_game(self):
        """测试无游戏时 get_state 返回错误"""
        result = self.controller.get_state()
        assert "error" in result
        assert "请先创建或加载游戏" in result["error"]
    
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
