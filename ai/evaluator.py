"""评价器模块

从 AI 对战记录生成训练样本。
输入：ai/ai_data/.../record.json
输出：ai/train_data/.../sample.json

使用双向棋型检测法计算每个位置的 reward 值，
并通过 Softmax 转换为概率分布。
"""

import os
import json
import math
from typing import List, Dict, Tuple
from copy import deepcopy

from ai.constants import AI_DATA_DIR, TRAIN_DATA_DIR
from ai.shape_detector import detect_shapes as _detect_shapes, detect_single_direction as _detect_single_direction
from ai.line_tracer import LineTracer
from ai.reward_calculator import RewardCalculator, RewardConfig
from ai.probability_utils import masked_softmax


class Evaluator:
    """评价器 - 生成训练样本（双向棋型检测法）"""
    
    # 常量定义
    MAX_REWARD = 1000000
    MIN_REWARD = -1000000
    DECAY_RATE = 0.9
    BLEND_ALPHA = 1.0  # 五连追溯奖励融合权重
    
    # 四个检测方向：横向、纵向、主对角线、副对角线
    DIRECTIONS = [
        (0, 1),   # 横向
        (1, 0),   # 纵向
        (1, 1),   # 主对角线
        (1, -1),  # 副对角线
    ]
    
    # 方向名称映射
    DIRECTION_NAMES = {
        (0, 1): "横向",
        (1, 0): "纵向",
        (1, 1): "主对角线",
        (1, -1): "副对角线",
    }
    
    def __init__(self, ai_data_dir: str = None, train_data_dir: str = None, decay_rate: float = None):
        """
        初始化评价器
        
        Args:
            ai_data_dir: AI 数据目录路径，默认为 AI_DATA_DIR
            train_data_dir: 训练数据目录路径，默认为 TRAIN_DATA_DIR
            decay_rate: 奖励衰减率 r（已弃用，保留参数兼容性）
        """
        if ai_data_dir is None:
            ai_data_dir = AI_DATA_DIR
        if train_data_dir is None:
            train_data_dir = TRAIN_DATA_DIR
        self.ai_data_dir = ai_data_dir
        self.train_data_dir = train_data_dir
        self.decay_rate = decay_rate if decay_rate is not None else self.DECAY_RATE
        self._line_tracer = LineTracer(
            max_reward=self.MAX_REWARD,
            min_reward=self.MIN_REWARD,
            decay_rate=self.decay_rate,
            blend_alpha=self.BLEND_ALPHA
        )
        self._reward_calculator = RewardCalculator(
            RewardConfig(
                max_reward=self.MAX_REWARD,
                min_reward=self.MIN_REWARD
            )
        )
    
    def evaluate_record(self, record_file: str) -> str:
        """
        评价单个记录文件，生成 sample.json
        
        Args:
            record_file: record.json 文件路径
            
        Returns:
            生成的 sample.json 文件路径
        """
        with open(record_file, 'r', encoding='utf-8') as f:
            record = json.load(f)
        
        history = record.get("history", [])
        winner = record.get("winner", 0)
        end_board = record.get("end_board", [])
        
        if not history or winner == 0:
            raise ValueError("记录文件无效或游戏未结束")
        
        # 计算每个 step 的 reward（融合棋型评估和五连追溯）
        sample_dict = self._compute_all_rewards(history, winner, end_board)
        
        # 生成输出文件路径
        sample_file = self._get_sample_path(record_file)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(sample_file), exist_ok=True)
        
        # 性能优化：去掉 indent=2，紧凑格式减少文件大小约50%
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_dict, f, ensure_ascii=False)
        
        return sample_file
    
    def _compute_all_rewards(self, history: List[Dict], winner: int = 0,
                              end_board: List[int] = None) -> List[Dict]:
        """
        计算所有步骤的 reward（双向棋型检测法 + 五连追溯）
        
        对每个 step 的每个空位，计算其 reward 值，
        然后通过 Softmax 转换为概率分布。
        
        如果有胜者，叠加五连追溯奖励。
        
        Args:
            history: 历史记录
            winner: 获胜方（1=黑, 2=白, 0=未结束, 3=平局）
            end_board: 最终棋盘状态（225 或 226 维）
            
        Returns:
            包含 input, output, mask 的样本列表
            output 为 Softmax 概率分布，值域 (0, 1)，总和为 1
        """
        sample_dict = []
        
        # 计算五连追溯奖励
        line_bonuses = {}
        if end_board and winner > 0 and winner < 3:
            # end_board 可能是 226 维，取前 225 位
            board_225 = end_board[:225] if len(end_board) >= 225 else end_board
            bonuses = self.trace_winning_line(board_225, winner, history)
            # 转换为 {step_index: bonus_reward} 字典
            line_bonuses = dict(bonuses)
        
        for step_idx, step in enumerate(history):
            input_vector = step.get("input", [])
            if len(input_vector) < 226:
                continue
            
            # 获取棋盘状态（前 225 位）和当前落子方
            board = input_vector[:225]
            current_player = input_vector[225]
            
            # 初始化 mask 和 raw_rewards
            mask = [0] * 225
            raw_rewards = [0.0] * 225
            
            # 对每个位置计算 reward
            for i in range(225):
                if board[i] != 0:
                    # 已落子位置：mask = 0, reward = 0
                    mask[i] = 0
                    raw_rewards[i] = 0.0
                else:
                    # 空位：mask = 1，计算 reward
                    mask[i] = 1
                    x, y = i // 15, i % 15
                    raw_rewards[i] = float(self._compute_position_reward(board, x, y, current_player))
            
            # 如果当前步骤有五连追溯奖励，叠加到实际落子位置
            if step_idx in line_bonuses:
                position = step.get("position", [])
                if len(position) >= 2:
                    px, py = position[0], position[1]
                    pos_idx = px * 15 + py
                    if 0 <= pos_idx < 225 and mask[pos_idx] == 1:
                        # 融合棋型评估奖励和五连追溯奖励
                        raw_rewards[pos_idx] = self.blend_rewards(
                            raw_rewards[pos_idx],
                            line_bonuses[step_idx]
                        )
            
            # 对 raw_rewards 应用 Softmax 转换为概率分布
            output = self._softmax(raw_rewards, mask)
            
            sample_dict.append({
                "input": input_vector,
                "output": output,
                "mask": mask
            })
        
        return sample_dict
    
    def _compute_position_reward(self, board: List[int], x: int, y: int, current_player: int) -> int:
        """计算单个位置的 reward（委托给 RewardCalculator）"""
        return self._reward_calculator.compute_position_reward(board, x, y, current_player)

    def detect_shapes(self, board: List[int], position: Tuple[int, int], player: int) -> List[Dict]:
        """检测落子后形成的所有棋型（委托给 ai.shape_detector）"""
        return _detect_shapes(board, position, player)

    def _detect_shape_in_direction(self, board: List[int], x: int, y: int,
                                    player: int, dx: int, dy: int) -> str:
        """检测单个方向上的棋型（委托给 ai.shape_detector）"""
        return _detect_single_direction(board, x, y, player, dx, dy)

    def shape_to_reward(self, board: List[int], shape_info: Dict,
                        player: int, is_current_player: bool, position=None) -> int:
        """将棋型转换为 reward 值（委托给 RewardCalculator）"""
        return self._reward_calculator.shape_to_reward(board, shape_info, player, is_current_player, position)

    def _compute_four_reward(self, board: List[int], shape_info: Dict,
                              player: int, is_current_player: bool, position=None) -> int:
        """计算冲四的递归 reward（委托给 RewardCalculator）"""
        return self._reward_calculator._compute_four_reward(board, shape_info, player, is_current_player, position)

    def _find_four_defense(self, simulated_board, x, y, player, dx, dy):
        """找到冲四的防守位置（委托给 RewardCalculator）"""
        return self._reward_calculator._find_four_defense(simulated_board, x, y, player, dx, dy)

    def _shape_reward_value(self, shape_type, count_directions, factor):
        """获取棋型的奖励值（委托给 RewardCalculator）"""
        return self._reward_calculator._shape_reward_value(shape_type, count_directions, factor)
    
    # Softmax 缩放因子（保留类常量以兼容外部引用）
    SOFTMAX_SCALE = 1.0 / 100000.0

    def _softmax(self, rewards: List[float], mask: List[int]) -> List[float]:
        """对 reward 值应用 Softmax 转换为概率分布（委托给 probability_utils）"""
        return masked_softmax(rewards, mask, scale=self.SOFTMAX_SCALE)
    
    def _get_sample_path(self, record_file: str) -> str:
        """
        根据 record.json 路径生成对应的 sample.json 路径
        
        将 ai_data 替换为 train_data
        """
        rel_path = os.path.relpath(record_file, self.ai_data_dir)
        sample_file = os.path.join(self.train_data_dir, rel_path)
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
    
    # ==================== 五连追溯奖励相关方法（委托给 LineTracer）====================

    def find_five_in_a_row(self, board: List[int], winner: int) -> List[Tuple[int, int]]:
        """在棋盘上找到五连的位置（委托给 LineTracer）"""
        return self._line_tracer.find_five_in_a_row(board, winner)

    def trace_winning_line(self, end_board: List[int], winner: int,
                           history: List[Dict]) -> List[Tuple[int, float]]:
        """追溯形成五连的那 5 步落子，计算反向传播奖励（委托给 LineTracer）"""
        return self._line_tracer.trace_winning_line(end_board, winner, history)

    def blend_rewards(self, shape_reward: float, line_bonus: float,
                      alpha: float = None) -> float:
        """融合棋型评估奖励和五连追溯奖励（委托给 LineTracer）"""
        return self._line_tracer.blend_rewards(shape_reward, line_bonus, alpha)
