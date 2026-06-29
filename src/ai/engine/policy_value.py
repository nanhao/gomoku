"""策略-价值 AI 引擎模块

PolicyValueAiEngine：基于 MCTS 搜索的落子引擎。
封装 MCTSSearcher，提供高层次的落子决策 API：
- get_move(board, current_player, strategy): 搜索 + 温度选择 + 返回 MoveDecision
- after_move(action): 子树复用（替代原 make_move + reset）
"""

import math
import random
from typing import List, Optional

from ai.constants import STRATEGY_ARGMAX, STRATEGY_SAMPLING, _SUPPORTED_STRATEGIES
from ai.engine.ai_engine import AIEngine, MoveDecision


class PolicyValueAiEngine(AIEngine):
    """策略-价值 AI 引擎

    基于 MCTSSearcher 的搜索结果做出落子决策。
    支持温度采样、子树复用、自动日志键生成。

    日志键由构造时注入的 log_key_prefix + 内部自增 _search_id 自动生成，
    调用方无需每次 get_move 传入 log_key。

    不可跨局复用：内部 searcher 的搜索树在局末会推进到终局根节点，
    后续局的 get_move 会从错误的根状态开始。每局必须创建新实例。
    """

    def __init__(
        self,
        searcher,
        temperature: float = 1.0,
        temp_steps: int = 8,
        log_key_prefix: Optional[str] = None,
    ):
        """初始化策略-价值引擎

        Args:
            searcher: MCTSSearcher 实例
            temperature: 温度参数 τ，控制落子随机性
            temp_steps: 温度探索步数（前 N 步 τ=1.0，之后 τ→0），默认 8
            log_key_prefix: 日志键前缀，格式 "run{run_id}_game{game_idx}"。
                None 时不输出日志；非 None 时每次 get_move 自动拼接
                "{log_key_prefix}_search{_search_id}" 作为日志标识。
        """
        self.searcher = searcher
        self.temperature = temperature
        self.temp_steps = temp_steps
        self.log_key_prefix = log_key_prefix
        self._step_count = 0  # 当前局已走步数
        self._search_id = 0   # 日志搜索 ID，从 0 递增

    # ---------- 公共 API ----------

    def get_move(
        self,
        board: List[List[int]],
        current_player: int,
        strategy: str = STRATEGY_ARGMAX,
    ) -> MoveDecision:
        """对给定棋盘状态做出落子决策

        流程：
        1. 检查棋盘是否已满
        2. 将 board + current_player 转换为 226 维向量
        3. 生成 log_key（若 log_key_prefix 非 None）
        4. 调用 searcher.search() 执行 MCTS
        5. 根据 strategy 决定落子方式：
           - "argmax"：强制贪婪选择（访问计数最大）
           - "sampling"：按内部温度逻辑采样（tau 由 _get_temperature() 决定）
        6. 返回 MoveDecision（含 position 和 mcts_visit_counts）

        Args:
            board: 15×15 棋盘矩阵，0=空, 1=黑, 2=白
            current_player: 当前落子方 (1=黑, 2=白)
            strategy: 落子策略，与 SinglePolicyAiEngine 一致：
                - "argmax"（默认）：贪婪选择访问计数最大者
                - "sampling"：按温度从访问计数采样

        Returns:
            MoveDecision 实例，包含：
            - position: (x, y) 落子坐标
            - mcts_visit_counts: 225 维原始访问计数

        Raises:
            ValueError: 棋盘已满，无空位可落子；strategy 取值未知
        """
        # 棋盘已满检查（与 SinglePolicyAiEngine 行为一致）
        if all(board[i][j] != 0 for i in range(15) for j in range(15)):
            raise ValueError("棋盘已满，无空位可落子")

        # 校验策略
        if strategy not in _SUPPORTED_STRATEGIES:
            raise ValueError(
                f"未知落子策略: {strategy!r}，支持的策略为 {_SUPPORTED_STRATEGIES}"
            )

        # 1. 将 board + current_player 转换为 226 维向量
        state = AIEngine.board_to_vector(board, current_player)

        # 2. 生成 log_key（自增 _search_id）
        log_key = None
        if self.log_key_prefix is not None:
            log_key = f"{self.log_key_prefix}_search{self._search_id}"
            self._search_id += 1

        # 3. MCTS 搜索
        visit_counts, root_value, root_policy = self.searcher.search(
            state, log_key=log_key
        )

        # 4. 按策略选择动作
        if strategy == STRATEGY_ARGMAX:
            # 强制贪婪选择（忽略温度）
            action = self._select_greedy(visit_counts, state[:225])
        else:  # STRATEGY_SAMPLING
            # 按内部温度逻辑决定 tau，然后采样
            tau = self._get_temperature()
            if tau <= 0:
                action = self._select_greedy(visit_counts, state[:225])
            else:
                action = self._sample_action(visit_counts, state[:225], tau)

        # 转换 flat index 为 (x, y) 坐标
        x, y = action // 15, action % 15

        return MoveDecision(x, y, mcts_visit_counts=visit_counts)

    def after_move(self, action: int):
        """落子后钩子：推进搜索树根节点

        调用 searcher.advance_root(action) 将子节点提升为新根。
        与旧 make_move 的关键差异：仅在 advance_root 成功后才递增
        _step_count（旧实现在失败时也会错误递增）。

        Args:
            action: 落子位置 0-224

        Raises:
            RuntimeError: 子树推进失败（action 对应子节点不在树中）
        """
        success = self.searcher.advance_root(action)
        if not success:
            raise RuntimeError(
                f"PolicyValueAiEngine.after_move({action}): advance_root failed. "
                f"The child node for action {action} does not exist in the search tree. "
                f"This indicates an unexpected state — the action should have been visited during search."
            )
        # 仅在成功后才递增步数（修正旧 make_move 的缺陷）
        self._step_count += 1

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
