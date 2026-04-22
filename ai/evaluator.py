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
        
        # 保存 sample.json
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_dict, f, ensure_ascii=False, indent=2)
        
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
        """
        计算单个位置的 reward（双向检测）
        
        Args:
            board: 棋盘状态（225 维）
            x: 行坐标
            y: 列坐标
            current_player: 当前落子方（1=黑，2=白）
            
        Returns:
            该位置的 reward 值
        """
        opponent = 3 - current_player  # 1->2, 2->1
        
        # 检测当前落子方在此位置落子后的棋型
        shapes1 = self.detect_shapes(board, (x, y), current_player)
        if shapes1:
            reward1 = max([self.shape_to_reward(board, shape, current_player, True, (x, y)) for shape in shapes1])
        else:
            reward1 = 0
        
        # 检测对手方在此位置落子后的棋型
        shapes2 = self.detect_shapes(board, (x, y), opponent)
        if shapes2:
            reward2 = max([self.shape_to_reward(board, shape, opponent, False, (x, y)) for shape in shapes2])
        else:
            reward2 = 0
        
        return max(reward1, reward2)
    
    def detect_shapes(self, board: List[int], position: Tuple[int, int], player: int) -> List[Dict]:
        """
        检测落子后形成的所有棋型
        
        Args:
            board: 当前棋盘状态
            position: 落子位置 (x, y)
            player: 落子方
            
        Returns:
            棋型列表，每项包含 {"shape": 棋型类型, "directions": [方向列表]}
            
        棋型汇总规则：
        - 多冲四：两个或以上方向形成冲四 -> 返回 "多冲四"
        - 冲四：仅一个方向形成冲四 -> 返回 "冲四"
        - 多活三：两个或以上方向形成活三 -> 返回 "多活三"
        - 活三：仅一个方向形成活三 -> 返回 "活三"
        - 眠三/活二：记录所有方向的数量，用于 count_directions
        """
        x, y = position
        raw_shapes = {}  # shape_type -> list of directions
        
        # 模拟落子
        simulated_board = list(board)  # 复制棋盘
        simulated_board[x * 15 + y] = player
        
        # 检测四个方向
        for dx, dy in self.DIRECTIONS:
            shape_type = self._detect_shape_in_direction(simulated_board, x, y, player, dx, dy)
            if shape_type:
                dir_name = self.DIRECTION_NAMES[(dx, dy)]
                if shape_type not in raw_shapes:
                    raw_shapes[shape_type] = []
                raw_shapes[shape_type].append(dir_name)
        
        
        # 合并棋型：处理多冲四、多活三等
        result = []
        
        # 优先级：连五 > 活四 > 多冲四 > 冲四 > 多活三 > 活三 > 眠三 > 活二 > 眠二 > 独一
        # 如果有连五，直接返回
        if "连五" in raw_shapes:
            return [{"shape": "连五", "directions": raw_shapes["连五"]}]
        
        # 活四
        if "活四" in raw_shapes:
            result.append({"shape": "活四", "directions": raw_shapes["活四"]})
        
        # 冲四 -> 多冲四 或 冲四
        if "冲四" in raw_shapes:
            if len(raw_shapes["冲四"]) >= 2:
                result.append({"shape": "多冲四", "directions": raw_shapes["冲四"]})
            else:
                result.append({"shape": "冲四", "directions": raw_shapes["冲四"]})
        
        # 活三 -> 多活三 或 活三
        if "活三" in raw_shapes:
            if len(raw_shapes["活三"]) >= 2:
                result.append({"shape": "多活三", "directions": raw_shapes["活三"]})
            else:
                result.append({"shape": "活三", "directions": raw_shapes["活三"]})
        
        # 眠三（记录所有方向）
        if "眠三" in raw_shapes:
            result.append({"shape": "眠三", "directions": raw_shapes["眠三"]})
        
        # 活二（记录所有方向）
        if "活二" in raw_shapes:
            result.append({"shape": "活二", "directions": raw_shapes["活二"]})
        
        
        # 眠二
        if "眠二" in raw_shapes:
            result.append({"shape": "眠二", "directions": raw_shapes["眠二"]})
        
        
        # 独一（保留所有方向，即使同时存在其他棋型）
        if "独一" in raw_shapes:
            result.append({"shape": "独一", "directions": raw_shapes["独一"]})
        
        
        return result
    
    def _detect_shape_in_direction(self, board: List[int], x: int, y: int, 
                                    player: int, dx: int, dy: int) -> str:
        """
        检测单个方向上的棋型
        
        Args:
            board: 棋盘状态（已模拟落子）
            x, y: 当前位置
            player: 玩家
            dx, dy: 检测方向
            
        Returns:
            棋型类型字符串，或 None
        """
        opponent = 3 - player
        
        # 统计该方向上连续己方棋子数（包含当前位置）
        count = 1
        
        # 正方向计数
        nx, ny = x + dx, y + dy
        while 0 <= nx < 15 and 0 <= ny < 15 and board[nx * 15 + ny] == player:
            count += 1
            nx += dx
            ny += dy
        # 检查正方向是否被阻挡
        blocked_positive = (not (0 <= nx < 15 and 0 <= ny < 15)) or (board[nx * 15 + ny] == opponent)
        
        # 反方向计数
        nx, ny = x - dx, y - dy
        while 0 <= nx < 15 and 0 <= ny < 15 and board[nx * 15 + ny] == player:
            count += 1
            nx -= dx
            ny -= dy
        # 检查反方向是否被阻挡
        blocked_negative = (not (0 <= nx < 15 and 0 <= ny < 15)) or (board[nx * 15 + ny] == opponent)
        
        # 计算被阻挡数
        blocked_count = int(blocked_positive) + int(blocked_negative)
        
        # 判断棋型
        if count >= 5:
            return "连五"
        
        if count == 4:
            if blocked_count == 0:
                return "活四"
            elif blocked_count == 1:
                return "冲四"
            else:  # blocked_count == 2: 两端都被阻挡，不构成威胁
                return None
        
        if count == 3:
            if blocked_count == 0:
                return "活三"
            elif blocked_count == 1:
                return "眠三"
            else:  # 两端都被阻挡
                return None
        
        if count == 2:
            if blocked_count == 0:
                return "活二"
            elif blocked_count == 1:
                return "眠二"
            else:
                return None
        
        if count == 1:
            return "独一"
        
        return None
    
    def shape_to_reward(self, board: List[int], shape_info: Dict, 
                        player: int, is_current_player: bool, position=None) -> int:
        """
        将棋型转换为 reward 值
        
        Args:
            board: 当前棋盘状态
            shape_info: 棋型信息 {"shape": 类型, "directions": [方向列表]}
            player: 落子方
            is_current_player: 是否是当前落子方（True=我方，False=敌方）
            position: 落子位置 (x, y)，用于冲四递归计算
            
        Returns:
            reward 值
        """
        shape_type = shape_info["shape"]
        directions = shape_info["directions"]
        count_directions = len(directions)
        
        # 敌方系数
        factor = 1.0 if is_current_player else 0.95
        
        if shape_type == "连五":
            return int(self.MAX_REWARD * factor)
        
        elif shape_type == "活四":
            return int(0.9 * self.MAX_REWARD * factor)
        
        elif shape_type == "多冲四":
            return int(0.9 * self.MAX_REWARD * factor)
        
        elif shape_type == "冲四":
            # 需要递归计算
            return self._compute_four_reward(board, shape_info, player, is_current_player, position)
        
        elif shape_type == "多活三":
            return int(0.8 * self.MAX_REWARD * factor)
        
        elif shape_type == "活三":
            return int(0.5 * self.MAX_REWARD * factor)
        
        elif shape_type == "眠三":
            return int((0.1 + count_directions * 0.1) * self.MAX_REWARD * factor)
        
        elif shape_type == "活二":
            return int((0.3 + count_directions * 0.1) * self.MAX_REWARD * factor)
        
        else:  # 眠二, 独一
            return 0
    
    def _compute_four_reward(self, board: List[int], shape_info: Dict,
                              player: int, is_current_player: bool, position=None) -> int:
        """计算冲四的递归 reward
        
        冲四递归流程：
        1. 模拟落子形成冲四
        2. 找到唯一的防守位置
        3. 模拟防守后，检测后续棋型
        4. 返回后续棋型的 reward
        
        Args:
            board: 当前棋盘状态（225 维）
            shape_info: 冲四棋型信息
            player: 落子方
            is_current_player: 是否是当前落子方
            position: 落子位置 (x, y)
            
        Returns:
            递归计算的 reward 值
        """
        opponent = 3 - player
        factor = 1.0 if is_current_player else 0.95
        
        # 如果没有传 position，回退到简化计算
        if position is None:
            return int(0.5 * self.MAX_REWARD * factor)
        
        x, y = position
        
        # 1. 模拟落子
        simulated_board = list(board)
        simulated_board[x * 15 + y] = player
        
        # 2. 找防守位置并模拟防守，检测后续棋型
        best_subsequent_reward = 0
        
        for dir_name in shape_info["directions"]:
            # 找到方向对应的 (dx, dy)
            direction_tuple = None
            for dxdy, name in self.DIRECTION_NAMES.items():
                if name == dir_name:
                    direction_tuple = dxdy
                    break
            
            if direction_tuple is None:
                continue
            
            dx, dy = direction_tuple
            defense_pos = self._find_four_defense(simulated_board, x, y, player, dx, dy)
            
            if defense_pos is None:
                continue
            
            def_x, def_y = defense_pos
            
            # 3. 模拟防守
            defense_board = list(simulated_board)
            defense_board[def_x * 15 + def_y] = opponent
            
            # 4. 检测防守后我方在原位置的后续棋型
            for ddx, ddy in self.DIRECTIONS:
                shape_type = self._detect_shape_in_direction(defense_board, x, y, player, ddx, ddy)
                if shape_type:
                    # 使用非递归方法获取奖励值（避免无限递归）
                    subsequent_reward = self._shape_reward_value(shape_type, 1, factor)
                    best_subsequent_reward = max(best_subsequent_reward, subsequent_reward)
        
        # 冲四的基础价值 = 防守后最佳后续棋型的奖励
        if best_subsequent_reward > 0:
            return best_subsequent_reward
        else:
            # 防守后无后续棋型，冲四价值较低
            return int(0.2 * self.MAX_REWARD * factor)
    
    def _find_four_defense(self, simulated_board, x, y, player, dx, dy):
        """找到冲四的防守位置（空端）
        
        冲四：四子连续，一端被阻挡，另一端为空。
        防守位置就是那个空端。
        """
        # 正方向：从(x,y)沿(dx,dy)找到最后一个己方棋子之后的空位
        nx, ny = x + dx, y + dy
        while 0 <= nx < 15 and 0 <= ny < 15 and simulated_board[nx * 15 + ny] == player:
            nx += dx
            ny += dy
        # 检查正方向端点是否为空
        positive_open = (0 <= nx < 15 and 0 <= ny < 15 and simulated_board[nx * 15 + ny] == 0)
        
        # 反方向
        nx, ny = x - dx, y - dy
        while 0 <= nx < 15 and 0 <= ny < 15 and simulated_board[nx * 15 + ny] == player:
            nx -= dx
            ny -= dy
        # 检查反方向端点是否为空
        negative_open = (0 <= nx < 15 and 0 <= ny < 15 and simulated_board[nx * 15 + ny] == 0)
        
        # 冲四：恰好一端开放
        if positive_open and not negative_open:
            # 正方向端点
            nx, ny = x + dx, y + dy
            while 0 <= nx < 15 and 0 <= ny < 15 and simulated_board[nx * 15 + ny] == player:
                nx += dx
                ny += dy
            return (nx, ny) if (0 <= nx < 15 and 0 <= ny < 15) else None
        elif negative_open and not positive_open:
            # 反方向端点
            nx, ny = x - dx, y - dy
            while 0 <= nx < 15 and 0 <= ny < 15 and simulated_board[nx * 15 + ny] == player:
                nx -= dx
                ny -= dy
            return (nx, ny) if (0 <= nx < 15 and 0 <= ny < 15) else None
        
        return None
    
    def _shape_reward_value(self, shape_type, count_directions, factor):
        """获取棋型的奖励值（不包含冲四递归计算，避免无限递归）
        
        用于冲四递归计算中获取防守后棋型的奖励值。
        """
        if shape_type == "连五":
            return int(self.MAX_REWARD * factor)
        elif shape_type == "活四":
            return int(0.9 * self.MAX_REWARD * factor)
        elif shape_type == "多冲四":
            return int(0.9 * self.MAX_REWARD * factor)
        elif shape_type == "冲四":
            return int(0.5 * self.MAX_REWARD * factor)
        elif shape_type == "多活三":
            return int(0.8 * self.MAX_REWARD * factor)
        elif shape_type == "活三":
            return int(0.5 * self.MAX_REWARD * factor)
        elif shape_type == "眠三":
            return int((0.1 + count_directions * 0.1) * self.MAX_REWARD * factor)
        elif shape_type == "活二":
            return int((0.3 + count_directions * 0.1) * self.MAX_REWARD * factor)
        else:  # 眠二, 独一
            return 0
    
    # Softmax 缩放因子：将 reward 从 [-1000000, 1000000] 缩放到 [-10, 10]
    # 避免 exp(x) 在大数值时的下溢问题
    SOFTMAX_SCALE = 1.0 / 100000.0
    
    def _softmax(self, rewards: List[float], mask: List[int]) -> List[float]:
        """
        对 reward 值应用 Softmax 转换为概率分布
        
        只考虑 mask=1 的位置（空位），已落子位置保持为 0。
        
        注意：reward 值可能高达 1000000，直接计算 exp 会导致数值下溢。
        因此先将 reward 缩放到合理范围，再计算 softmax。
        
        Args:
            rewards: 225 维 reward 列表
            mask: 225 维 mask 列表
            
        Returns:
            225 维概率分布，值域 (0, 1)，所有 mask=1 位置的概率和为 1
        """
        # 找到所有有效位置（mask=1）
        valid_indices = [i for i in range(225) if mask[i] == 1]
        
        if not valid_indices:
            # 没有有效位置，返回全 0
            return [0.0] * 225
        
        # 提取有效位置的 reward 值
        valid_rewards = [rewards[i] for i in valid_indices]
        
        # 缩放 reward 到合理范围，避免数值下溢
        scaled_rewards = [r * self.SOFTMAX_SCALE for r in valid_rewards]
        
        # 数值稳定性处理：减去最大值
        max_reward = max(scaled_rewards)
        exp_values = [math.exp(r - max_reward) for r in scaled_rewards]
        
        # 计算 softmax
        sum_exp = sum(exp_values)
        softmax_values = [exp_val / sum_exp for exp_val in exp_values]
        
        # 构建完整的 225 维输出
        output = [0.0] * 225
        for idx, prob in zip(valid_indices, softmax_values):
            output[idx] = prob
        
        return output
    
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
    
    # ==================== 五连追溯奖励相关方法 ====================
    
    def find_five_in_a_row(self, board: List[int], winner: int) -> List[Tuple[int, int]]:
        """
        在棋盘上找到五连的位置
    
        Args:
            board: 225 维棋盘向量
            winner: 获胜方 (1=黑, 2=白)
    
        Returns:
            五连的 5 个位置坐标列表 [(x, y), ...]
        """
        # 四个方向：横向、纵向、主对角线、副对角线
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
        for x in range(15):
            for y in range(15):
                if board[x * 15 + y] != winner:
                    continue
    
                for dx, dy in directions:
                    positions = [(x, y)]
                    nx, ny = x + dx, y + dy
    
                    while 0 <= nx < 15 and 0 <= ny < 15:
                        if board[nx * 15 + ny] == winner:
                            positions.append((nx, ny))
                            nx += dx
                            ny += dy
                        else:
                            break
    
                    if len(positions) >= 5:
                        # 返回前 5 个位置
                        return positions[:5]
    
        return []
    
    def trace_winning_line(self, end_board: List[int], winner: int,
                           history: List[Dict]) -> List[Tuple[int, float]]:
        """
        追溯形成五连的那 5 步落子，计算反向传播奖励
    
        Args:
            end_board: 最终棋盘状态（225 维）
            winner: 获胜方 (1=黑, 2=白, 0=未结束, 3=平局)
            history: 对局历史记录
    
        Returns:
            [(step_index, bonus_reward), ...] 落子步骤索引和奖励
        """
        # 无胜者或平局，返回空列表
        if winner == 0 or winner == 3:
            return []
    
        # 1. 找到五连位置
        winning_positions = self.find_five_in_a_row(end_board, winner)
        if not winning_positions:
            return []
    
        # 2. 在 history 中找出这 5 个位置对应的落子步骤
        step_indices = []
        for pos in winning_positions:
            pos_list = list(pos)
            for i, step in enumerate(history):
                step_pos = step.get("position", [])
                if step_pos == pos_list:
                    # 检查落子方是否正确
                    input_vector = step.get("input", [])
                    if len(input_vector) > 225:
                        player = input_vector[225]
                        if player == winner:
                            step_indices.append(i)
                            break
    
        # 如果没有找到足够的落子步骤，返回空
        if len(step_indices) < 5:
            return []
    
        # 3. 按 step 排序（从小到大）
        step_indices.sort()
    
        # 4. 反向传播奖励
        # j5 是最后一步（索引最大的），奖励 = MAX_REWARD
        # j4 奖励 = r * j5 的奖励
        # ...
        results = []
        reward = float(self.MAX_REWARD)
        for i in range(len(step_indices) - 1, -1, -1):
            results.append((step_indices[i], reward))
            reward *= self.DECAY_RATE
    
        return results
    
    def blend_rewards(self, shape_reward: float, line_bonus: float,
                      alpha: float = None) -> float:
        """融合棋型评估奖励和五连追溯奖励
        
        使用加法融合，确保两种信号都能贡献到最终奖励。
        加法融合避免了取 max 时弱信号被完全丢弃的问题，
        使得五连追溯的早期步骤（衰减后奖励较小）也能影响训练信号。
        
        融合后的奖励限制在 [MIN_REWARD, MAX_REWARD] 范围内，
        确保 Softmax 缩放的一致性。
    
        Args:
            shape_reward: 棋型评估奖励
            line_bonus: 五连追溯奖励
            alpha: 五连追溯奖励权重，默认使用 BLEND_ALPHA
    
        Returns:
            融合后的奖励，裁剪到 [MIN_REWARD, MAX_REWARD] 范围
        """
        if alpha is None:
            alpha = self.BLEND_ALPHA
        blended = shape_reward + alpha * line_bonus
        # 裁剪到合法范围，防止超出 MAX_REWARD
        return max(self.MIN_REWARD, min(self.MAX_REWARD, blended))
