"""评价器模块

从 AI 对战记录生成训练样本。
输入：ai_data/.../record.json
输出：train_data/.../sample.json
"""

import os
import json
import shutil
from typing import List, Dict, Tuple


class Evaluator:
    """评价器 - 生成训练样本"""
    
    # 常量定义
    MAX_REWARD = 1000000
    MIN_REWARD = -1000000
    DECAY_RATE = 0.9
    
    def __init__(self, ai_data_dir: str = None, train_data_dir: str = None, decay_rate: float = None):
        """
        初始化评价器
        
        Args:
            ai_data_dir: AI 数据目录路径，默认为 ai/ai_data
            train_data_dir: 训练数据目录路径，默认为 ai/train_data
            decay_rate: 奖励衰减率 r，默认使用类常量 DECAY_RATE (0.9)
        """
        # 默认路径指向 ai/ 子目录
        if ai_data_dir is None:
            ai_data_dir = os.path.join("ai", "ai_data")
        if train_data_dir is None:
            train_data_dir = os.path.join("ai", "train_data")
        self.ai_data_dir = ai_data_dir
        self.train_data_dir = train_data_dir
        self.decay_rate = decay_rate if decay_rate is not None else self.DECAY_RATE
    
    def evaluate_record(self, record_file: str) -> str:
        """
        评价单个记录文件，生成 sample.json
        
        Args:
            record_file: record.json 文件路径
            
        Returns:
            生成的 sample.json 文件路径
        """
        # 读取记录
        with open(record_file, 'r', encoding='utf-8') as f:
            record = json.load(f)
        
        history = record.get("history", [])
        winner = record.get("winner", 0)
        
        if not history or winner == 0:
            raise ValueError("记录文件无效或游戏未结束")
        
        # 第一步：初始化 sample_dict
        sample_dict = self._initialize_sample(history)
        
        # 第二步：反向传播计算 reward
        self._compute_rewards(sample_dict, history, winner)
        
        # 生成输出文件路径
        sample_file = self._get_sample_path(record_file)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(sample_file), exist_ok=True)
        
        # 保存 sample.json
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_dict, f, ensure_ascii=False, indent=2)
        
        return sample_file
    
    def _initialize_sample(self, history: List[Dict]) -> List[Dict]:
        """
        第一步：初始化 sample_dict
        
        对于每个 step：
        - 输入向量前 225 位是棋盘状态
        - output 是 225 维 reward 列表，索引 i 对应位置 (i//15, i%15)
        - 如果位置已有棋子，reward = MIN_REWARD
        - 否则 reward = 0
        - mask 全部初始化为 1（当前版本所有位置均参与损失计算）
        """
        sample_dict = []
        
        for step in history:
            input_vector = step.get("input", [])
            output_vector = step.get("output", [])
            
            if len(input_vector) < 226 or len(output_vector) < 225:
                continue
            
            # 获取棋盘状态（前 225 位）
            board = input_vector[:225]
            
            # mask 全部初始化为 1：当前版本所有位置均参与损失计算
            # （未来版本可能重新启用 mask 机制，算法不变）
            mask = [1] * 225
            new_output = [0] * 225
            
            for i in range(225):
                # 索引 i 对应位置 (i//15, i%15)
                if board[i] != 0:
                    # 已有棋子，设置 reward 为 MIN_REWARD
                    new_output[i] = self.MIN_REWARD
                else:
                    # 空位，reward 为 0
                    new_output[i] = 0
            
            sample_dict.append({
                "input": input_vector,
                "output": new_output,
                "mask": mask
            })
        
        return sample_dict
    
    def _find_five_in_a_row(self, winning_actions: List[Tuple[int, List[int]]], 
                             history: List[Dict]) -> List[int]:
        """
        从获胜方落子序列中找出构成五子连线的5个步骤索引
        
        Args:
            winning_actions: 获胜方落子列表，每项为 (step_index, [x, y])
            history: 完整历史记录
            
        Returns:
            构成五子连线的5个步骤索引 [j1, j2, j3, j4, j5]，如果找不到返回空列表
        """
        if len(winning_actions) < 5:
            return []
        
        # 获取最后一步的棋盘状态
        last_step_idx = winning_actions[-1][0]
        if last_step_idx >= len(history):
            return []
        
        final_board = history[last_step_idx]["input"][:225]
        winner = history[last_step_idx]["input"][225]  # 获胜方编号
        
        # 四个方向：水平、垂直、主对角线、副对角线
        directions = [
            (0, 1),   # 水平
            (1, 0),   # 垂直
            (1, 1),   # 主对角线
            (1, -1)   # 副对角线
        ]
        
        # 获取获胜方所有落子位置及其步骤索引
        winner_positions = {}  # (x, y) -> step_index
        for step_idx, pos in winning_actions:
            winner_positions[(pos[0], pos[1])] = step_idx
        
        # 从最后一步位置开始，向四个方向寻找五子连线
        last_pos = winning_actions[-1][1]
        last_x, last_y = last_pos[0], last_pos[1]
        
        for dx, dy in directions:
            # 收集该方向上的所有获胜方落子
            line_positions = []
            
            # 向正方向搜索
            x, y = last_x, last_y
            while 0 <= x < 15 and 0 <= y < 15:
                if final_board[x * 15 + y] == winner:
                    if (x, y) in winner_positions:
                        line_positions.append((winner_positions[(x, y)], (x, y)))
                else:
                    break
                x += dx
                y += dy
            
            # 向反方向搜索
            x, y = last_x - dx, last_y - dy
            while 0 <= x < 15 and 0 <= y < 15:
                if final_board[x * 15 + y] == winner:
                    if (x, y) in winner_positions:
                        line_positions.append((winner_positions[(x, y)], (x, y)))
                else:
                    break
                x -= dx
                y -= dy
            
            # 去重并按步骤索引排序
            line_positions = list(set(line_positions))
            line_positions.sort(key=lambda x: x[0])
            
            # 如果该方向上有至少5个落子，返回最后5个的step_index
            if len(line_positions) >= 5:
                # 返回最后5个的step_index
                return [pos[0] for pos in line_positions[-5:]]
        
        return []
    
    def _compute_rewards(self, sample_dict: List[Dict], history: List[Dict], winner: int):
        """
        第二步：反向传播计算 reward（新算法）
        
        新算法规则：
        1. 只找出构成五子连线的5步，对这5步进行 reward 反向传播
        2. 失败方不做处理（保持初始化值）
        
        Args:
            sample_dict: 初始化的样本字典
            history: 历史记录
            winner: 获胜方（1=黑，2=白）
        """
        # 分离黑白双方的落子
        black_actions = []  # (step_index, position)
        white_actions = []  # (step_index, position)
        
        for i, step in enumerate(history):
            position = step.get("position", [0, 0])
            player = step.get("input", [0] * 226)[225]  # 最后一位是当前落子方
            
            if player == 1:
                black_actions.append((i, position))
            else:
                white_actions.append((i, position))
        
        # 确定获胜方
        if winner == 1:
            winning_actions = black_actions
        else:
            winning_actions = white_actions
        
        # 找出构成五子连线的5步
        five_steps = self._find_five_in_a_row(winning_actions, history)
        
        # 对这5步进行 reward 反向传播
        if five_steps:
            current_reward = self.MAX_REWARD
            # 从最后一步（j5）向前遍历
            for step_idx in reversed(five_steps):
                position = history[step_idx]["position"]
                reward_idx = position[0] * 15 + position[1]
                
                if 0 <= reward_idx < 225:
                    sample_dict[step_idx]["output"][reward_idx] = int(current_reward)
                
                current_reward *= self.decay_rate
        
        # 失败方不做处理（保持初始化值：已落子=min_reward，空位=0）
    
    def _get_sample_path(self, record_file: str) -> str:
        """
        根据 record.json 路径生成对应的 sample.json 路径
        
        将 ai_data 替换为 train_data
        """
        # 获取相对路径
        rel_path = os.path.relpath(record_file, self.ai_data_dir)
        # 替换目录名
        sample_file = os.path.join(self.train_data_dir, rel_path)
        # 替换文件名
        sample_file = sample_file.replace("record.json", "sample.json")
        
        return sample_file
    
    def evaluate_round(self, round_dir: str):
        """
        评价整个轮次的所有对局
        
        Args:
            round_dir: 轮次目录路径（如 ai_data/random_1）
        """
        if not os.path.exists(round_dir):
            return
        
        for game_dir in sorted(os.listdir(round_dir)):
            game_path = os.path.join(round_dir, game_dir)
            record_file = os.path.join(game_path, "record.json")
            
            if os.path.isdir(game_path) and os.path.exists(record_file):
                try:
                    self.evaluate_record(record_file)
                except Exception as e:
                    print(f"评价失败 {record_file}: {e}")
