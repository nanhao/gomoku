"""MctsGameRunner 单元测试

测试 mcts_game_runner.py 中的 MctsGameRunner 类和 GameResult 数据类。
覆盖对局执行、回调机制、engine 选择、self-play 模式等。
"""

import os
import sys
import unittest
from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import MagicMock

# 添加项目根目录到路径
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ai.scripts.mcts_game_runner import MctsGameRunner, GameResult
from ai.scripts.mcts_self_play_worker import MctsSelfPlayConfig
from ai.engine.ai_engine import MoveDecision


class _MockPolicyValueEngine:
    """模拟 PolicyValueAiEngine，用于 MctsGameRunner 的单元测试。

    不依赖真实模型和 MCTS 搜索，直接按预设的落子序列返回 MoveDecision。
    """

    def __init__(self, moves: List[tuple], check_get_move_calls: bool = True):
        """初始化 mock engine

        Args:
            moves: 预设的落子序列 [(x, y), ...]，按调用顺序返回
            check_get_move_calls: 是否记录 get_move 调用（用于 TC1.5 engine 选择验证）
        """
        self._moves = moves
        self._call_count = 0
        self._after_move_count = 0
        self._get_move_calls = []  # 记录每次 get_move 的 current_player
        self._check_calls = check_get_move_calls

    def get_move(self, board, current_player, strategy=None):
        if self._check_calls:
            self._get_move_calls.append(current_player)
        if self._call_count >= len(self._moves):
            raise RuntimeError("mock engine 预设落子耗尽")
        x, y = self._moves[self._call_count]
        self._call_count += 1
        # 返回 MoveDecision，mcts_visit_counts 给 225 维全 1 列表（确保不被 visit_counts 过滤）
        return MoveDecision(x, y, mcts_visit_counts=[1] * 225, mcts_q_values=[0.0] * 225)

    def after_move(self, x, y):
        self._after_move_count += 1


def _build_quick_win_moves_black() -> List[tuple]:
    """构建黑方快速获胜的落子序列（黑在 row 7 连五）"""
    # 黑(7,7) 白(0,0) 黑(7,8) 白(0,1) 黑(7,9) 白(0,2) 黑(7,10) 白(0,3) 黑(7,11)
    return [(7, 7), (0, 0), (7, 8), (0, 1), (7, 9), (0, 2), (7, 10), (0, 3), (7, 11)]


def _build_quick_win_moves_white() -> List[tuple]:
    """构建白方快速获胜的落子序列（白在 row 7 连五）

    黑方落在 column 0 的分散位置（不形成五连），白方在 row 7 连五获胜。
    """
    return [(0, 0), (7, 7), (2, 0), (7, 8), (4, 0), (7, 9), (6, 0), (7, 10), (8, 0), (7, 11)]


def _build_draw_moves() -> List[tuple]:
    """构建填满棋盘但不产生五连的序列。用交替填充但不形成五连的模式。"""
    moves = []
    # 用蛇形填充，每行填满但避免五连：让黑白交替填入相邻位置
    for row in range(15):
        for col in range(15):
            moves.append((row, col))
    return moves


class TestGameResult(unittest.TestCase):
    """GameResult 数据类测试"""

    def test_default_values(self):
        result = GameResult(winner=1, move_count=45)
        self.assertEqual(result.winner, 1)
        self.assertEqual(result.move_count, 45)

    def test_draw(self):
        result = GameResult(winner=0, move_count=225)
        self.assertEqual(result.winner, 0)


class TestMctsGameRunner(unittest.TestCase):
    """MctsGameRunner 类单元测试 — TC1.1 ~ TC1.7"""

    def setUp(self):
        self._config = MctsSelfPlayConfig(simulations=2)
        self._runner = MctsGameRunner(self._config)

    # ---------- TC1.1: 黑胜场景 ----------

    def test_run_black_wins(self):
        """TC1.1: 黑方按五连序列落子，返回 winner=1"""
        moves = _build_quick_win_moves_black()
        engine_black = _MockPolicyValueEngine(moves[::2])  # 黑方步: 索引 0,2,4,...
        engine_white = _MockPolicyValueEngine(moves[1::2])  # 白方步: 索引 1,3,5,...
        result = self._runner.run(engine_black, engine_white, "test", 0)
        self.assertEqual(result.winner, 1)
        self.assertGreater(result.move_count, 0)

    # ---------- TC1.2: 白胜场景 ----------

    def test_run_white_wins(self):
        """TC1.2: 白方按五连序列落子，返回 winner=2"""
        moves = _build_quick_win_moves_white()
        engine_black = _MockPolicyValueEngine(moves[::2])
        engine_white = _MockPolicyValueEngine(moves[1::2])
        result = self._runner.run(engine_black, engine_white, "test", 0)
        self.assertEqual(result.winner, 2)
        self.assertGreater(result.move_count, 0)

    # ---------- TC1.4: on_move 回调 ----------

    def test_run_on_move_call_count(self):
        """TC1.4: on_move 回调调用次数等于对局步数"""
        moves = _build_quick_win_moves_black()
        engine_black = _MockPolicyValueEngine(moves[::2])
        engine_white = _MockPolicyValueEngine(moves[1::2])

        call_records = []

        def on_move(state, decision, current_player):
            call_records.append((len(state), current_player))

        result = self._runner.run(
            engine_black, engine_white, "test", 0, on_move=on_move
        )
        self.assertEqual(len(call_records), result.move_count)
        # 每步 call 中 state 应为 226 维（225 棋盘 + 1 当前玩家）
        for state_len, _ in call_records:
            self.assertEqual(state_len, 226)

    # ---------- TC1.5: engine 选择正确 ----------

    def test_run_engine_selection(self):
        """TC1.5: engine_black 仅在 current_player=1 时被调用；
        engine_white 仅在 current_player=2 时被调用"""
        moves = _build_quick_win_moves_black()
        engine_black = _MockPolicyValueEngine(moves[::2], check_get_move_calls=True)
        engine_white = _MockPolicyValueEngine(moves[1::2], check_get_move_calls=True)

        self._runner.run(engine_black, engine_white, "test", 0)

        self.assertGreater(len(engine_black._get_move_calls), 0)
        self.assertGreater(len(engine_white._get_move_calls), 0)
        for cp in engine_black._get_move_calls:
            self.assertEqual(cp, 1, "engine_black 仅在 current_player=1 时调用")
        for cp in engine_white._get_move_calls:
            self.assertEqual(cp, 2, "engine_white 仅在 current_player=2 时调用")

    # ---------- TC1.6: after_move 调用 ----------

    def test_run_after_move_call_count(self):
        """TC1.6: after_move 调用次数等于 get_move 调用次数"""
        moves = _build_quick_win_moves_black()
        engine_black = _MockPolicyValueEngine(moves[::2])
        engine_white = _MockPolicyValueEngine(moves[1::2])

        result = self._runner.run(engine_black, engine_white, "test", 0)
        move_count = result.move_count
        self.assertEqual(engine_black._after_move_count + engine_white._after_move_count, move_count)

    # ---------- TC1.7: self-play 模式 ----------

    def test_run_self_play_mode_same_instance(self):
        """TC1.7: 同一 engine 实例同时作为 black 和 white 传入，对局正常完成"""
        moves = _build_quick_win_moves_black()
        engine = _MockPolicyValueEngine(moves, check_get_move_calls=True)
        result = self._runner.run(engine, engine, "test", 0)
        self.assertIn(result.winner, (0, 1, 2))
        self.assertGreater(result.move_count, 0)


if __name__ == "__main__":
    unittest.main()
