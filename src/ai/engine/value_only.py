"""价值头 AI 引擎模块

ValueOnlyAiEngine：基于价值头模型的枚举-评估落子引擎。
对每个候选空位模拟落子，用模型评估局面价值，再归一化为概率分布后决策。
不涉及 MCTS 搜索。
"""

import math
import random as _stdrandom
from typing import List, Optional, Tuple

from ai.base_model import BaseModel
from ai.constants import STRATEGY_ARGMAX, STRATEGY_SAMPLING, _SUPPORTED_STRATEGIES
from ai.engine.ai_engine import AIEngine, MoveDecision
from game.game import check_winner, is_board_full


class ValueOnlyAiEngine(AIEngine):
    """价值头 AI 引擎 - 枚举候选位置 + 模型评估 + 概率归一化

    负责：
    1. 棋盘预处理：排除孤立空位
    2. 对非孤立空位模拟落子，检测终局
    3. 非终局位置用模型评估价值
    4. masked softmax 归一化为概率分布
    5. 按策略（argmax / sampling）选择落子

    落子策略：
    - "argmax"：选择概率最大的合法位置；并列时随机打破
    - "sampling"：温度加权采样，委托给 ai.move_sampler.temperature_sample

    采样参数在构造时注入：
    - temperature / top_k / rng 一局对弈中不变。
    """

    def __init__(
        self,
        model: BaseModel,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        rng: Optional[_stdrandom.Random] = None,
    ):
        """初始化价值头 AI 引擎

        Args:
            model: AI 模型实例，必须继承 BaseModel
            temperature: 温度值 T > 0，仅在 strategy="sampling" 时生效
            top_k: 采样截断阈值，None 表示不截断
            rng: 可选随机源，None 时使用 random 模块全局源
        """
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.rng = rng

    # ---------- 公共 API ----------

    def get_move(
        self,
        board: List[List[int]],
        current_player: int,
        strategy: str = STRATEGY_ARGMAX,
    ) -> MoveDecision:
        """对给定棋盘状态做出落子决策

        执行 5 步算法：预处理 → 终局检测 → 模型评估 → 归一化 → 决策。

        Args:
            board: 15×15 棋盘矩阵，0=空, 1=黑, 2=白
            current_player: 当前落子方 (1=黑, 2=白)
            strategy: 落子策略，"argmax" 或 "sampling"

        Returns:
            MoveDecision 对象，含落子坐标和概率分布

        Raises:
            ValueError: 棋盘已满；strategy 取值未知
        """
        # 棋盘已满检查
        flat_board = [board[i][j] for i in range(15) for j in range(15)]
        if is_board_full(flat_board):
            raise ValueError("棋盘已满，无空位可落子")

        # 校验策略
        if strategy not in _SUPPORTED_STRATEGIES:
            raise ValueError(
                f"未知落子策略: {strategy!r}，支持的策略为 {_SUPPORTED_STRATEGIES}"
            )

        # === 步骤 1: 预处理——分离孤立空位与候选空位 ===
        evaluations = [0.0] * 225      # 每个位置的评估值
        candidate_positions: List[int] = []  # 非孤立空位的平坦索引
        occupied_mask = [False] * 225   # True = 已有棋子或孤立空位（不参与决策）

        for idx in range(225):
            if flat_board[idx] != 0:
                occupied_mask[idx] = True
                continue
            if self._is_isolated_empty(flat_board, idx):
                occupied_mask[idx] = True  # 孤立空位与已有棋子一样 mask 掉
            else:
                candidate_positions.append(idx)

        # === 步骤 2: 模拟落子，检测终局 ===
        opponent = 2 if current_player == 1 else 1
        need_model_eval: List[Tuple[int, List[int]]] = []  # (候选索引, 新局面 flat_board)

        # 兜底：如果所有空位都被 mask（如空棋盘），则开放全部空位
        if not candidate_positions:
            for idx in range(225):
                if flat_board[idx] == 0:
                    candidate_positions.append(idx)
                    occupied_mask[idx] = False

        for p in candidate_positions:
            # 模拟落子
            sim_board = flat_board.copy()
            sim_board[p] = current_player

            # 终局检测
            winner = check_winner(sim_board, p)
            if winner == current_player:
                evaluations[p] = 1.0   # 胜
            elif winner == opponent:
                evaluations[p] = -1.0  # 负（当前方落子后对方赢，不应发生，但兜底）
            elif is_board_full(sim_board):
                evaluations[p] = 0.0   # 平
            else:
                need_model_eval.append((p, sim_board))

        # === 步骤 3: 模型评估非终局位置 ===
        if need_model_eval:
            # 构建 board_vector（对手视角）
            board_vectors: List[List[int]] = []
            for _p, sim_board in need_model_eval:
                bv = sim_board.copy()
                bv.append(opponent)  # 对手视角
                board_vectors.append(bv)

            results = self.model.predict_batch(board_vectors)

            for i, (p, _) in enumerate(need_model_eval):
                v_i = results[i].value  # 对手视角价值
                evaluations[p] = -v_i   # 取反转回当前方视角

        # === 步骤 3.5: 当前局面价值评估（用于回放展示）===
        bv_current = flat_board.copy()
        bv_current.append(current_player)
        state_value = self.model.predict(bv_current).value

        # === 步骤 4: masked softmax 归一化 ===
        probs = self._masked_softmax(evaluations, occupied_mask)

        # === 步骤 5: 落子决策 ===
        # 构建合法位置列表
        valid_positions: List[Tuple[int, int]] = []
        valid_probs: List[float] = []
        for idx in range(225):
            if not occupied_mask[idx]:
                x = idx // 15
                y = idx % 15
                valid_positions.append((x, y))
                valid_probs.append(probs[idx])

        if strategy == STRATEGY_ARGMAX:
            x, y = self._pick_argmax(valid_positions, valid_probs)
        else:  # STRATEGY_SAMPLING
            from ai.move_sampler import temperature_sample
            x, y = temperature_sample(
                valid_positions, valid_probs,
                temperature=self.temperature,
                top_k=self.top_k,
                rng=self.rng,
            )

        return MoveDecision(x, y, probs=probs,
                          value_evaluations=evaluations,
                          state_value=state_value)

    def after_move(self, x: int, y: int):
        """落子后钩子：空操作（无搜索树管理需求）。

        Args:
            x: 落子行坐标 0-14（未使用）
            y: 落子列坐标 0-14（未使用）
        """
        pass

    # ---------- 静态工具方法 ----------

    @staticmethod
    def _is_isolated_empty(board_flat: List[int], idx: int) -> bool:
        """判断空位是否为孤立空位

        孤立空位：周围 2 格范围内（5×5 窗口）均无任何棋子。

        Args:
            board_flat: 225 维展平棋盘
            idx: 待检测位置索引

        Returns:
            True 表示孤立空位
        """
        x = idx // 15
        y = idx % 15

        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                nx = x + dx
                ny = y + dy
                if 0 <= nx < 15 and 0 <= ny < 15:
                    if board_flat[nx * 15 + ny] != 0:
                        return False  # 发现棋子，非孤立

        return True  # 周围 2 格内均无棋子

    @staticmethod
    def _masked_softmax(
        evaluations: List[float],
        occupied_mask: List[bool],
    ) -> List[float]:
        """对评估值做 masked softmax 归一化

        已有棋子的位置 mask 掉（概率 = 0），其余位置参与 softmax。
        使用数值稳定技巧（减去最大值）。

        Args:
            evaluations: 225 维评估值（-1/0/1 或模型输出）
            occupied_mask: 225 维 mask，True 表示已有棋子

        Returns:
            225 维概率分布，总和为 1
        """
        # 找出参与 softmax 的位置（非棋子位置）
        valid_indices = [i for i in range(225) if not occupied_mask[i]]

        if not valid_indices:
            # 无有效位置（不应发生，棋盘已满会提前抛异常）
            return [0.0] * 225

        # 提取有效位置的评估值
        valid_evals = [evaluations[i] for i in valid_indices]

        # 数值稳定：减去最大值
        max_eval = max(valid_evals)
        exp_values = [math.exp(e - max_eval) for e in valid_evals]

        # 归一化
        sum_exp = sum(exp_values)
        softmax_values = [ev / sum_exp for ev in exp_values]

        # 构建完整 225 维输出
        probs = [0.0] * 225
        for idx, prob in zip(valid_indices, softmax_values):
            probs[idx] = prob

        return probs

    @staticmethod
    def _pick_argmax(
        valid_positions: List[Tuple[int, int]],
        valid_probs: List[float],
    ) -> Tuple[int, int]:
        """argmax 选择：从概率最大的合法位置集合中随机选一个

        多个位置概率并列最大时，随机打破平局。

        Args:
            valid_positions: 合法（空位）坐标列表
            valid_probs: 与 valid_positions 一一对应的概率列表

        Returns:
            选中的 (x, y) 坐标
        """
        max_prob = max(valid_probs)
        candidates = [
            pos for pos, p in zip(valid_positions, valid_probs)
            if p == max_prob
        ]
        x, y = _stdrandom.choice(candidates)
        return (x, y)
