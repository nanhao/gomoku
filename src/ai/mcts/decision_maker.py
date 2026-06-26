"""MCTS 落子决策器模块

基于 MCTSSearcher 的搜索结果做出落子决策。
支持温度采样、子树复用。
"""

import math
import random
from typing import List, Optional

from ai.engine import MoveDecision


class MCTSMoveDecider:
    """MCTS 落子决策器

    封装 MCTSSearcher，提供高层次的落子决策 API：
    - get_move(): 搜索 + 温度选择 + 返回 MoveDecision
    - make_move(): 子树复用

    不继承/不修改 AIEngine，独立实现。
    """

    def __init__(
        self,
        searcher,
        temperature: float = 1.0,
        temp_steps: int = 8,
    ):
        """初始化决策器

        Args:
            searcher: MCTSSearcher 实例
            temperature: 温度参数 τ，控制落子随机性
            temp_steps: 温度探索步数（前 N 步 τ=1.0，之后 τ→0），默认 8
        """
        self.searcher = searcher
        self.temperature = temperature
        self.temp_steps = temp_steps
        self._step_count = 0  # 当前局已走步数

    # ---------- 公共 API ----------

    def get_move(
        self,
        state: List[int],
        log_key: Optional[str] = None,
    ) -> MoveDecision:
        """对给定状态做出落子决策

        流程：
        1. 调用 searcher.search() 执行 MCTS
        2. 根据温度和步数计算 π 分布
        3. 按 π 分布采样落子（或贪婪选择）
        4. 返回 MoveDecision（含 position 和 mcts_visit_counts）

        Args:
            state: 226 维 board_vector
            log_key: 日志标识，透传给 searcher.search()

        Returns:
            MoveDecision 实例，包含：
            - position: (x, y) 落子坐标
            - mcts_visit_counts: 225 维原始访问计数
        """
        # MCTS 搜索
        visit_counts, root_value, root_policy = self.searcher.search(
            state, log_key=log_key
        )

        # 决定当前步的温度
        tau = self._get_temperature()

        # 从访问计数计算 π 并选择动作
        if tau <= 0:
            # 贪婪：选访问计数最大的合法空位
            action = self._select_greedy(visit_counts, state[:225])
        else:
            # 温度采样
            action = self._sample_action(visit_counts, state[:225], tau)

        # 转换 flat index 为 (x, y) 坐标
        x, y = action // 15, action % 15

        return MoveDecision(x, y, mcts_visit_counts=visit_counts)

    def make_move(self, action: int):
        """执行落子，复用子树

        调用 searcher.advance_root(action) 将子节点提升为新根。
        失败时 reset 搜索树。

        Args:
            action: 落子位置 0-224
        """
        self._step_count += 1
        success = self.searcher.advance_root(action)
        if not success:
            self.searcher.reset()

    def reset(self):
        """重置决策器状态"""
        self._step_count = 0
        self.searcher.reset()

    # ---------- 内部方法 ----------

    def _get_temperature(self) -> float:
        """根据当前步数和温度步数决定温度值

        前 temp_steps 步使用 temperature，之后 τ=0（纯贪婪）。
        """
        if self._step_count < self.temp_steps:
            return self.temperature
        return 0.0  # 探索步数之后纯贪婪

    @staticmethod
    def _select_greedy(visit_counts: List[int], board: List[int]) -> int:
        """贪婪选择：在合法空位中选访问计数最大的

        Args:
            visit_counts: 225 维访问计数
            board: 225 维棋盘（0=空）

        Returns:
            选中动作索引
        """
        best_action = None
        best_count = -1
        for action in range(225):
            if board[action] == 0 and visit_counts[action] > best_count:
                best_count = visit_counts[action]
                best_action = action
        return best_action if best_action is not None else 0

    @staticmethod
    def _sample_action(
        visit_counts: List[int], board: List[int], tau: float
    ) -> int:
        """温度采样选择动作

        π(a) ∝ N(a)^(1/τ)，仅考虑空位。

        Args:
            visit_counts: 225 维访问计数
            board: 225 维棋盘
            tau: 温度参数

        Returns:
            选中动作索引
        """
        # 计算采样权重（仅空位）
        weights = []
        actions = []
        for action in range(225):
            if board[action] == 0:
                weight = visit_counts[action] ** (1.0 / tau) if visit_counts[action] > 0 else 0.0
                weights.append(weight)
                actions.append(action)

        if not actions:
            return 0

        total = sum(weights)
        if total <= 0:
            # 所有权重为 0，均匀随机选
            return random.choice(actions)

        # 归一化 + 采样
        probs = [w / total for w in weights]
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return actions[i]

        return actions[-1]  # 兜底
