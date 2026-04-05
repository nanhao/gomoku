"""游戏接口模块 - 唯一对外接口"""

import os
import json

from .game import GomokuGame
from .save import GomokuSave


class GomokuController:
    """游戏控制器 - 所有外部调用都通过此类"""
    
    def __init__(self, save_dir=None):
        """初始化控制器
        
        Args:
            save_dir: 存档目录路径，默认为项目根目录下的 game/save/
        """
        self.game = None
        self.current_save_name = None
        
        # 默认使用项目根目录下的 game/save/
        if save_dir is None:
            # 获取项目根目录（controller.py 的上级目录）
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            save_dir = os.path.join(project_root, "game", "save")
        
        self.save = GomokuSave(save_dir)
    
    def new_game(self, save_name=None):
        """新建游戏并保存
        
        Args:
            save_name: 存档名称，如果为None则自动生成
        """
        # 自动生成存档名称
        if save_name is None:
            save_name = self._generate_save_name()
        
        # 检查存档名是否已存在
        existing_saves = self.save.list_saves()
        if save_name in existing_saves:
            return {
                "error": f"存档 '{save_name}' 已存在"
            }
        
        self.game = GomokuGame()
        self.current_save_name = save_name
        
        # 保存初始状态
        state = self.game.get_state()
        state["name"] = save_name
        state["start_time"] = self._get_current_time()
        
        self.save.save_game(state, save_name)
        
        return {
            "status": "success",
            "state": state
        }
    
    def make_move(self, x, y):
        """在当前玩家落子"""
        if self.game is None:
            return {"error": "请先创建或加载游戏"}
        
        # 检查游戏是否已结束
        if self.game.winner != 0:
            winner = self.game.winner
            if winner == 3:
                return {"error": "游戏已结束，平局"}
            else:
                return {"error": f"游戏已结束，{'黑方' if winner == 1 else '白方'}获胜"}
        
        try:
            # 执行落子
            winner = self.game.make_move(x, y)
            
            # 获取游戏状态
            state = self.game.get_state()
            
            # 返回结果
            result = {
                "status": "success",
                "winner": winner,
                "current_player": self.game.current_player,
                "board": state["board"]
            }
            
            # 如果有获胜方，添加获胜信息
            if winner != 0:
                result["game_over"] = True
                if winner == 3:
                    result["result"] = "平局"
                else:
                    result["result"] = f"{'黑方' if winner == 1 else '白方'}获胜"
            
            return result
            
        except Exception as e:
            return {"error": str(e)}
    
    def list_saves(self):
        """列出所有存档"""
        return self.save.list_saves()
    
    def load_game(self, save_name):
        """加载存档"""
        try:
            # 加载存档
            state = self.save.load_game(save_name)
            
            # 创建游戏实例
            self.game = GomokuGame(state)
            self.current_save_name = save_name
            
            return {
                "status": "success",
                "state": state
            }
        except Exception as e:
            return {"error": str(e)}
    
    def is_over(self):
        """检测游戏是否结束"""
        if self.game is None:
            return False
        return self.game.winner != 0
    
    def is_draw(self):
        """检测平局"""
        if self.game is None:
            return False
        return self.game.winner == 3
    
    def get_current_player(self):
        """获取当前玩家"""
        if self.game is None:
            return None
        return self.game.current_player
    
    def get_winner(self):
        """获取获胜方"""
        if self.game is None:
            return None
        return self.game.winner
    
    def get_state(self, short=False):
        """返回游戏状态，dict 格式"""
        if self.game is None:
            return {"error": "请先创建或加载游戏"}
        
        state = self.game.get_state()
        
        if short:
            # 简化版：只包含 board, current_player, winner
            return {
                "board": state["board"],
                "current_player": state["current_player"],
                "winner": state["winner"]
            }
        else:
            # 完整版：包含所有信息
            # 如果是新游戏，需要添加 name 和 start_time
            if "name" not in state:
                state["name"] = self.current_save_name or "未知游戏"
            if "start_time" not in state:
                state["start_time"] = self._get_current_time()
            
            return state
    
    def get_state_json(self, short=False):
        """返回游戏状态，JSON 格式"""
        state = self.get_state(short)
        return json.dumps(state, ensure_ascii=False, indent=4)
    
    def _get_current_time(self):
        """获取当前时间字符串"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _generate_save_name(self):
        """生成默认存档名称
        
        格式: 我的游戏{i}，i从1开始，跳过已存在的编号
        """
        existing_saves = self.save.list_saves()
        i = 1
        while True:
            name = f"我的游戏{i}"
            if name not in existing_saves:
                return name
            i += 1

