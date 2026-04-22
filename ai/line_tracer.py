"""五连追溯奖励模块

从 evaluator.py 提取，负责从终局棋盘找到五连位置，
追溯对应历史步骤，反向传播衰减奖励。
"""

from typing import List, Dict, Tuple


class LineTracer:
    """五连追溯奖励计算器"""

    def __init__(self, max_reward: int = 1000000, min_reward: int = -1000000,
                 decay_rate: float = 0.9, blend_alpha: float = 1.0):
        """初始化

        Args:
            max_reward: 最大奖励值
            min_reward: 最小奖励值
            decay_rate: 奖励衰减率
            blend_alpha: 五连追溯奖励融合权重
        """
        self.max_reward = max_reward
        self.min_reward = min_reward
        self.decay_rate = decay_rate
        self.blend_alpha = blend_alpha

    def find_five_in_a_row(self, board: List[int], winner: int) -> List[Tuple[int, int]]:
        """在棋盘上找到五连的位置

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
        """追溯形成五连的那 5 步落子，计算反向传播奖励

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
        reward = float(self.max_reward)
        for i in range(len(step_indices) - 1, -1, -1):
            results.append((step_indices[i], reward))
            reward *= self.decay_rate

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
            alpha: 五连追溯奖励权重，默认使用 blend_alpha

        Returns:
            融合后的奖励，裁剪到 [MIN_REWARD, MAX_REWARD] 范围
        """
        if alpha is None:
            alpha = self.blend_alpha
        blended = shape_reward + alpha * line_bonus
        # 裁剪到合法范围，防止超出 MAX_REWARD
        return max(self.min_reward, min(self.max_reward, blended))
