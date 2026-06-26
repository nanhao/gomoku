"""AI 引擎模块

连接 UI 和 AI 模型，处理输入输出，选择落子位置。
"""

import random
import math
from typing import List, Optional, Tuple
from ai.base_model import BaseModel


# 支持的落子策略常量
STRATEGY_ARGMAX = "argmax"
STRATEGY_SAMPLING = "sampling"
_SUPPORTED_STRATEGIES = (STRATEGY_ARGMAX, STRATEGY_SAMPLING)


def create_engine(ai_name: str) -> "AIEngine":
    """
    根据 AI 名称创建对应的 AIEngine 实例。

    AI 名称到模型的映射由 ai.registry 统一维护，外部调用者无需了解具体模型类。

    异常策略：
    - 未知模型名称、多代模型未指定代数等场景，直接上抛 ValueError，
      由 app 层捕获并转换为 ModelLoadError，向 GUI 层传达模型加载失败信息。
    - 不再回退到 RandomModel，避免静默降级导致用户误以为使用了指定模型。

    Args:
        ai_name: AI 名称，如 "random"、"fc_v1"、"cnn_v3.1"

    Returns:
        对应的 AIEngine 实例

    Raises:
        ValueError: 模型名称未知、多代模型未指定代数等
    """
    from ai.registry import get_model

    model = get_model(ai_name)
    return AIEngine(model)


class AIEngine:
    """AI 引擎 - 连接 UI 和 AI 模型

    负责：
    1. 将棋盘状态转换为模型输入向量
    2. 调用模型获取概率分布
    3. 根据指定策略（默认 argmax）在合法位置中选择落子

    落子策略（参见设计文档 design/engine_move_strategy_design.md、
    design/move_sampler_design.md）：
    - "argmax"（默认）：选择概率最大的合法位置；存在并列最大值时随机打破平局
    - "sampling"：温度加权采样，委托给 ai.move_sampler.temperature_sample 实现
    """

    def __init__(self, model: BaseModel):
        """
        初始化 AI 引擎

        Args:
            model: AI 模型实例，必须继承 BaseModel
        """
        self.model = model

    def get_move(
        self,
        board: List[List[int]],
        current_player: int,
        probs: List[float] = None,
        strategy: str = STRATEGY_ARGMAX,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ) -> Tuple[int, int]:
        """
        获取 AI 落子位置

        流程：
        1. 将棋盘转换为 226 维向量
        2. 调用模型获取 225 维概率分布（若已传入 probs 则跳过）
        3. 在合法位置（空位）中按 strategy 选择落子

        Args:
            board: 15x15 棋盘，0=空, 1=黑, 2=白
            current_player: 当前落子方，1=黑, 2=白
            probs: 预计算的概率分布（225 维），若提供则跳过 model.predict 调用
            strategy: 落子策略，取值见模块常量 STRATEGY_*。
                - "argmax"（默认）：选择概率最大的合法位置，并列时随机打破
                - "sampling"：温度采样（见 ai.move_sampler.temperature_sample）
            temperature: 仅在 strategy="sampling" 时生效，温度值 T > 0。
                默认 1.0（等价模型原始概率）。T → 0 趋近 argmax；
                T → ∞ 趋近均匀。argmax 分支完全不使用此参数。
            top_k: 仅在 strategy="sampling" 时生效，截断阈值；None 表示不截断。
            rng: 仅在 strategy="sampling" 时生效，可选随机源；None 时使用
                random 模块全局源。

        Returns:
            (x, y) 落子位置，0 <= x < 15, 0 <= y < 15

        Raises:
            ValueError: 棋盘已满，无空位可落子；strategy 取值未知；
                或 strategy="sampling" 时 temperature <= 0。
        """
        # 1. 转换棋盘为向量
        board_vector = self.board_to_vector(board, current_player)

        # 2. 获取模型预测（225 维概率分布 + 可选价值）
        # 性能优化：支持传入预计算的 probs，避免重复调用 model.predict
        if probs is None:
            result = self.model.predict(board_vector)
            # 统一通过 ModelPredict 取策略概率
            probs = result.policy

        # 兼容：如果 probs 是 ModelPredict 对象，提取 .policy
        if hasattr(probs, 'policy'):
            probs = probs.policy

        # 3. 构建位置-概率列表，排除已有棋子的位置
        valid_positions = []
        valid_probs = []
        for i in range(225):
            x = i // 15
            y = i % 15
            if 0 <= x < 15 and 0 <= y < 15 and board[x][y] == 0:
                valid_positions.append((x, y))
                valid_probs.append(probs[i])

        if not valid_positions:
            raise ValueError("棋盘已满，无空位可落子")

        # 4. 按策略选择位置
        if strategy == STRATEGY_ARGMAX:
            return self._pick_argmax(valid_positions, valid_probs)
        if strategy == STRATEGY_SAMPLING:
            # 委托给 ai.move_sampler.temperature_sample（本地导入避免循环依赖）
            from ai.move_sampler import temperature_sample
            return temperature_sample(
                valid_positions,
                valid_probs,
                temperature=temperature,
                top_k=top_k,
                rng=rng,
            )
        raise ValueError(
            f"未知落子策略: {strategy!r}，支持的策略为 {_SUPPORTED_STRATEGIES}"
        )

    @staticmethod
    def _pick_argmax(
        valid_positions: List[Tuple[int, int]],
        valid_probs: List[float],
    ) -> Tuple[int, int]:
        """argmax 选择：从概率最大的合法位置集合中随机选一个。

        设计要点：
        - 多个位置概率并列最大时，随机打破平局，避免固定取第一个导致
          自对弈路径退化为完全确定性序列。
        - 当 max_prob == 0（例如模型把所有概率压在已落子位置上）时，
          所有空位都并列，此函数退化为在全部空位中均匀随机挑一个；
          这是合理的兜底，不应抛异常。

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
        # 并列时随机挑一个；只有一个候选时 random.choice 等价于直接返回
        return random.choice(candidates)

    @staticmethod
    def board_to_vector(board: List[List[int]], current_player: int) -> List[int]:
        """
        将棋盘转换为 226 维向量

        转换规则：
        - 前 225 位：按行优先展开棋盘，board[0][0], board[0][1], ..., board[14][14]
        - 第 226 位：当前落子方（1=黑, 2=白）

        Args:
            board: 15x15 棋盘
            current_player: 当前落子方

        Returns:
            226 维向量
        """
        vector = []
        for row in board:
            vector.extend(row)
        vector.append(current_player)
        return vector
