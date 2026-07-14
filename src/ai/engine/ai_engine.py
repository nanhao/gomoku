"""AI 引擎抽象基类模块

定义 AIEngine 抽象接口、MoveDecision 数据结构、以及棋盘向量化工具。
所有具体引擎实现（SinglePolicyAiEngine、PolicyValueAiEngine）均继承此基类。
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class MoveDecision:
    """AI 引擎落子决策结果

    封装 get_move() 的返回值，替代原有的裸 tuple。
    设计为可扩展：后续可添加概率分布、价值评估等字段。

    Attributes:
        _position: (x, y) 落子坐标 tuple
        _probs: 225 维概率分布（模型原始输出），未传入时为 None
        _mcts_visit_counts: 225 维 MCTS 访问计数，未传入时为 None
        _mcts_root_value: MCTS 根节点价值评估（从当前玩家视角），未传入时为 None
    """

    def __init__(self, x: int, y: int, probs: List[float] = None, mcts_visit_counts: List[int] = None, mcts_root_value: float = None):
        """
        Args:
            x: 落子行坐标 (0-14)
            y: 落子列坐标 (0-14)
            probs: 可选的 225 维概率分布
            mcts_visit_counts: 可选的 225 维 MCTS 访问计数
            mcts_root_value: 可选的 MCTS 根节点价值，值域 [-1, 1]
        """
        self._position: Tuple[int, int] = (x, y)
        self._probs: Optional[List[float]] = probs
        self._mcts_visit_counts: Optional[List[int]] = mcts_visit_counts
        self._mcts_root_value: Optional[float] = mcts_root_value

    def get_position(self) -> Tuple[int, int]:
        """返回落子位置 (x, y) tuple"""
        return self._position

    def get_probs(self) -> Optional[List[float]]:
        """返回 225 维概率分布，未传入时返回 None"""
        return self._probs

    def get_mcts_visit_counts(self) -> Optional[List[int]]:
        """返回 225 维 MCTS 访问计数，未传入时返回 None"""
        return self._mcts_visit_counts

    def get_mcts_root_value(self) -> Optional[float]:
        """返回 MCTS 根节点价值，未传入时返回 None"""
        return self._mcts_root_value

    def __repr__(self) -> str:
        return f"MoveDecision(x={self._position[0]}, y={self._position[1]})"


class AIEngine(ABC):
    """AI 引擎抽象基类

    定义统一的引擎接口，两个具体实现：
    - SinglePolicyAiEngine：单策略引擎，直接使用模型预测 + 策略选择落子
    - PolicyValueAiEngine：策略-价值引擎，通过 MCTS 搜索落子

    子类自行管理构造参数和字段（两个子类的构造参数完全不同，
    无法在抽象基类共享 __init__）。
    """

    @abstractmethod
    def get_move(
        self,
        board: List[List[int]],
        current_player: int,
        strategy: str = "argmax",
    ) -> MoveDecision:
        """获取 AI 落子位置

        Args:
            board: 15x15 棋盘，0=空, 1=黑, 2=白
            current_player: 当前落子方，1=黑, 2=白
            strategy: 落子策略，"argmax" 或 "sampling"

        Returns:
            MoveDecision 对象，可通过 get_position() 获取 (x, y) 落子位置

        Raises:
            ValueError: 棋盘已满，无空位可落子；strategy 取值未知
        """
        ...

    @abstractmethod
    def after_move(self, x: int, y: int):
        """落子后钩子

        SinglePolicyAiEngine: no-op（无子树管理需求）。
        PolicyValueAiEngine: 调用 searcher.advance_root(action) 推进搜索树，
        其中 action = x * 15 + y。

        Args:
            x: 落子行坐标 0-14
            y: 落子列坐标 0-14

        Raises:
            RuntimeError: advance_root 失败时（仅 PolicyValueAiEngine）
        """
        ...

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
