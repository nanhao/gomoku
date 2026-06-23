"""Reward 计算核心模块

从 evaluator.py 提取，负责将棋型映射为数值 reward，处理冲四递归防守模拟。
引入 RewardConfig 配置驱动，集中管理所有 reward 数值。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from ai.shape_detector import detect_shapes, detect_single_direction, DIRECTION_NAMES


@dataclass
class RewardConfig:
    """Reward 配置数据类，集中管理所有棋型的 reward 数值比例

    所有数值均为相对于 max_reward 的比例系数。
    """
    max_reward: int = 1_000_000
    min_reward: int = -1_000_000
    values: Dict[str, float] = field(default_factory=lambda: {
        "连五": 1.0,
        "活四": 0.9,
        "多冲四": 0.9,
        "冲四": 0.5,
        "多活三": 0.8,
        "活三": 0.5,
        "眠三": 0.1,   # 基础值，实际 = (0.1 + count_directions * 0.1) * max_reward
        "活二": 0.3,   # 基础值，实际 = (0.3 + count_directions * 0.1) * max_reward
        "眠二": 0.0,
        "独一": 0.0,
    })


class RewardCalculator:
    """Reward 计算核心"""

    def __init__(self, config: RewardConfig = None):
        """初始化

        Args:
            config: RewardConfig 实例，默认使用默认值
        """
        self.config = config or RewardConfig()

    def compute_position_reward(self, board: List[int], x: int, y: int, current_player: int) -> int:
        """计算单个位置的 reward（双向检测）

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
        shapes1 = detect_shapes(board, (x, y), current_player)
        if shapes1:
            reward1 = max([self.shape_to_reward(board, shape, current_player, True, (x, y)) for shape in shapes1])
        else:
            reward1 = 0

        # 检测对手方在此位置落子后的棋型
        shapes2 = detect_shapes(board, (x, y), opponent)
        if shapes2:
            reward2 = max([self.shape_to_reward(board, shape, opponent, False, (x, y)) for shape in shapes2])
        else:
            reward2 = 0

        return max(reward1, reward2)

    def shape_to_reward(self, board: List[int], shape_info: Dict,
                        player: int, is_current_player: bool, position=None) -> int:
        """将棋型转换为 reward 值

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
        count_directions = len(shape_info["directions"])
        factor = 1.0 if is_current_player else 0.95

        if shape_type == "冲四":
            # 需要递归计算
            return self._compute_four_reward(board, shape_info, player, is_current_player, position)

        ratio = self._get_shape_ratio(shape_type, count_directions)
        return int(ratio * self.config.max_reward * factor)

    def _get_shape_ratio(self, shape_type: str, count_directions: int) -> float:
        """获取棋型对应的 reward 比例系数

        Args:
            shape_type: 棋型类型字符串
            count_directions: 方向数量（用于眠三、活二的动态计算）

        Returns:
            相对于 max_reward 的比例系数
        """
        base = self.config.values.get(shape_type, 0.0)
        if shape_type == "眠三":
            return base + count_directions * 0.1
        elif shape_type == "活二":
            return base + count_directions * 0.1
        return base

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
            return int(self.config.values["冲四"] * self.config.max_reward * factor)

        x, y = position

        # 1. 模拟落子
        simulated_board = list(board)
        simulated_board[x * 15 + y] = player

        # 2. 找防守位置并模拟防守，检测后续棋型
        best_subsequent_reward = 0

        for dir_name in shape_info["directions"]:
            # 找到方向对应的 (dx, dy)
            direction_tuple = None
            for dxdy, name in DIRECTION_NAMES.items():
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
            for ddx, ddy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                shape_type = detect_single_direction(defense_board, x, y, player, ddx, ddy)
                if shape_type:
                    # 使用非递归方法获取奖励值（避免无限递归）
                    subsequent_reward = self._shape_reward_value(shape_type, 1, factor)
                    best_subsequent_reward = max(best_subsequent_reward, subsequent_reward)

        # 冲四的基础价值 = 防守后最佳后续棋型的奖励
        if best_subsequent_reward > 0:
            return best_subsequent_reward
        else:
            # 防守后无后续棋型，冲四价值较低
            return int(0.2 * self.config.max_reward * factor)

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
        ratio = self._get_shape_ratio(shape_type, count_directions)
        return int(ratio * self.config.max_reward * factor)
