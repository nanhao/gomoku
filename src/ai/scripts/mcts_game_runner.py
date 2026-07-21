"""MCTS 对局运行器模块

封装公用的 MCTS 对局循环：棋盘管理、MCTS 搜索、交替落子、终局判定。
通过回调机制让调用方注入差异化行为，供 self-play 和 battle 两场景共用。

设计文档：design/42/main_docs/mcts_game_runner_design.md
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

from ai.engine.policy_value import PolicyValueAiEngine
from ai.engine.ai_engine import MoveDecision
from ai.constants import STRATEGY_SAMPLING
from game.game import check_winner, is_board_full
from ai.scripts.mcts_config import MctsSelfPlayConfig


@dataclass
class GameResult:
    """单局对弈结果

    Attributes:
        winner: 获胜方（0=平局, 1=黑胜, 2=白胜）
        move_count: 本局总步数
    """
    winner: int
    move_count: int


class MctsGameRunner:
    """MCTS 对局运行器

    封装公用的 MCTS 对局循环，供 MctsSelfPlayWorker 和 MctsBattleWorker 共用。
    构造器不创建引擎，引擎由调用方在每局开始前创建并传入 run 方法。

    用法示例（self-play）：
        runner = MctsGameRunner(config)
        engine = build_policy_value(model, ...)
        def on_move(state, md, player):
            # 收集样本
            ...
        result = runner.run(engine, engine, run_id, game_idx, on_move=on_move)

    用法示例（battle）：
        runner = MctsGameRunner(config)
        result = runner.run(engine_black, engine_white, run_id, game_idx)
    """

    def __init__(self, config: MctsSelfPlayConfig):
        """初始化对局运行器

        Args:
            config: MCTS 超参数配置（复用 MctsSelfPlayConfig）
        """
        self._config = config

    # ---------- 公共 API ----------

    def run(
        self,
        engine_black: PolicyValueAiEngine,
        engine_white: PolicyValueAiEngine,
        run_id: str,
        game_idx: int,
        on_move: Optional[Callable[[List[int], MoveDecision, int], None]] = None,
    ) -> GameResult:
        """执行一局 MCTS 对弈

        初始化空棋盘后循环执行 MCTS 搜索、落子、终局判定。
        engine_black 和 engine_white 可以是同一实例（self-play）或不同实例（battle）。

        Args:
            engine_black: 执黑引擎（current_player=1 时使用）
            engine_white: 执白引擎（current_player=2 时使用）
            run_id: 本次运行唯一标识（时间戳后缀）
            game_idx: 局编号（用于 log_key 前缀）
            on_move: 每步回调，签名为 (state: 226维, decision: MoveDecision, current_player: int)。
                self-play 场景用于收集样本；battle 场景传 None。

        Returns:
            GameResult，包含 winner 和 move_count

        Raises:
            RuntimeError: engine.after_move 失败（子树推进异常）
        """
        # 初始化 225 维空棋盘，黑方先手
        board: List[int] = [0] * 225
        current_player: int = 1
        move_count: int = 0

        while True:
            # 根据当前玩家选择对应引擎
            engine = engine_black if current_player == 1 else engine_white

            # 构造 226 维 state（棋盘 + 当前玩家指示）
            state = board + [current_player]

            # 转换为 15×15 二维数组供 engine 使用
            board_2d = [[board[i * 15 + j] for j in range(15)] for i in range(15)]

            # 构造 log_key（与 MctsSelfPlayWorker 中一致）
            log_key_prefix = f"run{run_id}_game{game_idx}"
            # 设置 engine 的 log_key_prefix（支持 MCTS 日志输出）
            engine.log_key_prefix = log_key_prefix

            # MCTS 搜索 + 决策
            md = engine.get_move(board_2d, current_player, strategy=STRATEGY_SAMPLING)
            x, y = md.get_position()
            action = x * 15 + y

            # 回调：self-play 用于收集样本
            if on_move is not None:
                on_move(state, md, current_player)

            # 落子
            board[action] = current_player
            move_count += 1

            # 终局判定
            winner = check_winner(board, action)
            is_terminal = (winner != 0) or is_board_full(board)

            # 子树复用：推进搜索树根节点
            engine.after_move(x, y)

            if is_terminal:
                break

            # 切换玩家
            current_player = 2 if current_player == 1 else 1

        return GameResult(winner=winner, move_count=move_count)
