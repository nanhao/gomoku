"""AI 对战数据记录器模块

记录 AI 对战的每一轮输入输出、落子位置、游戏结果。
数据保存到 ai_data/ 目录下。
"""

import os
import json
from typing import List, Tuple


class DataRecorder:
    """AI 对战数据记录器
    
    负责：
    1. 创建轮次目录（ai_data/{ai_name}_{递增整数}/）
    2. 每局游戏一个子目录（1, 2, 3, ...）
    3. 每次落子记录 input、output、position
    4. 游戏结束保存完整记录文件
    """
    
    def __init__(self, ai_name: str, ai_data_dir: str = "ai_data"):
        """
        初始化数据记录器
        
        Args:
            ai_name: AI 名称，用于命名轮次目录
            ai_data_dir: 数据存储根目录，默认为项目根目录下的 ai_data/
        """
        self.ai_name = ai_name
        self.ai_data_dir = ai_data_dir
        self.round_dir = self._create_round_dir()
        self.current_game_id = 0
        self.current_game_data = None
    
    def _create_round_dir(self) -> str:
        """
        创建轮次目录
        
        目录命名规则：{ai_name}_{从1递增整数}
        例如：random_1, random_2, ...
        
        Returns:
            轮次目录的完整路径
        """
        # 确保根目录存在
        os.makedirs(self.ai_data_dir, exist_ok=True)
        
        # 查找下一个可用的轮次编号
        round_num = 1
        while True:
            dir_name = f"{self.ai_name}_{round_num}"
            dir_path = os.path.join(self.ai_data_dir, dir_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                return dir_path
            round_num += 1
    
    def new_game(self) -> int:
        """
        开始新一局游戏
        
        Returns:
            游戏编号（从 1 开始递增）
        """
        self.current_game_id += 1
        self.current_game_data = {
            "history": [],
            "winner": 0,
            "end_board": []
        }
        return self.current_game_id
    
    def record_move(self, game_id: int, input_vector: List[int],
                    output_vector: List[int], position: Tuple[int, int]):
        """
        记录一次落子
        
        每次落子都要记录，不是游戏结束才记录。
        
        Args:
            game_id: 游戏编号（由 new_game 返回）
            input_vector: 226 维输入向量（棋盘 + 当前落子方）
            output_vector: 675 维输出向量（x, y, reward 列表）
            position: 引擎最终成功落子的位置 (x, y)
            
        Raises:
            ValueError: 游戏编号不匹配
        """
        if game_id != self.current_game_id:
            raise ValueError(
                f"游戏编号不匹配: 传入 {game_id}, 当前是 {self.current_game_id}"
            )
        
        self.current_game_data["history"].append({
            "input": input_vector,
            "output": output_vector,
            "position": list(position)
        })
    
    def save_game_result(self, game_id: int, winner: int, end_board: List[int]):
        """
        保存游戏结果到文件
        
        Args:
            game_id: 游戏编号
            winner: 获胜方（1=黑胜, 2=白胜, 3=平局, 0=未结束）
            end_board: 最终棋盘状态（226 维向量）
            
        Raises:
            ValueError: 游戏编号不匹配
        """
        if game_id != self.current_game_id:
            raise ValueError(
                f"游戏编号不匹配: 传入 {game_id}, 当前是 {self.current_game_id}"
            )
        
        self.current_game_data["winner"] = winner
        self.current_game_data["end_board"] = end_board
        
        # 创建局数子目录
        game_dir = os.path.join(self.round_dir, str(game_id))
        os.makedirs(game_dir, exist_ok=True)
        
        # 保存到文件
        file_path = os.path.join(game_dir, "record.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.current_game_data, f, ensure_ascii=False, indent=2)
    
    def get_round_dir(self) -> str:
        """获取当前轮次目录路径"""
        return self.round_dir
