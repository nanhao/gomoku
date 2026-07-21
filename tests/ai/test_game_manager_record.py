"""AIGameManager record_move 温度和采样概率记录测试

从 test_game_manager_temp.py 拆分出的子文件，覆盖：
- TestRecordMoveTempInfo：验证 play_single_game 在温度采样时向 recorder 传递正确信息
- TestSamplingProbsOccupiedPositions：验证 sampling_probs 覆盖全部 225 位
- TestSingleGameResult：SingleGameResult NamedTuple 及 play_single_game 返回值测试
- TestStartAiVsAiResultFormat：start_ai_vs_ai 结果列表测试
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# 添加项目根目录到路径
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from game import GomokuController
from ai.engine import AIEngine
from ai.constants import STRATEGY_ARGMAX, STRATEGY_SAMPLING
from ai.game_manager import AIGameManager
from ai.move_sampler import (
    TempConfig,
    TEMP_SIDE_BLACK,
    TEMP_SIDE_WHITE,
    TEMP_SIDE_BOTH,
    TEMP_SIDE_RANDOM,
)

from tests.ai.test_game_manager_temp import (
    _SpyEngine,
    _make_recorder,
    _TempSaveDirTestBase,
)


# ---------------------------------------------------------------------------
# 第四十一次追加：验证 record_move 记录了温度采样信息
# ---------------------------------------------------------------------------

class _SpyRecorder:
    """记录 record_move 每次调用参数的桩 recorder。

    用于验证 play_single_game 在温度采样启用时是否正确向
    DataRecorder.record_move() 传递 strategy / temperature / sampling_probs。
    """

    def __init__(self):
        self.current_game_id = 1
        self.move_calls: list = []

    def new_game(self):
        self.current_game_id += 1

    def record_move(self, game_id, input_vector, output_vector, position,
                    strategy=None, temperature=None, top_k=None,
                    sampling_probs=None, mcts_probs=None, mcts_value=None, mcts_q_values=None):
        self.move_calls.append({
            "game_id": game_id,
            "strategy": strategy,
            "temperature": temperature,
            "top_k": top_k,
            "has_sampling_probs": sampling_probs is not None,
            "sampling_probs_len": len(sampling_probs) if sampling_probs is not None else 0,
            "has_mcts_probs": mcts_probs is not None,
        })

    def save_game_result(self, game_id, winner, end_board,
                         temp_side_this_game=None):
        pass

    def get_round_dir(self):
        import tempfile as _tempfile
        import os as _os
        return _tempfile.mkdtemp()


class TestRecordMoveTempInfo(_TempSaveDirTestBase):
    """验证 play_single_game 在温度采样时向 recorder 传递正确信息。"""

    def test_record_move_with_temp_config_passes_strategy_and_temperature(self):
        """temp_config 启用时，record_move 收到 strategy="sampling" 和温度值。"""
        engine = _SpyEngine()
        recorder = _SpyRecorder()
        cfg = TempConfig(
            temperature=1.5, temp_moves=99, temp_side=TEMP_SIDE_BOTH, seed=1,
        )

        with patch('ai.game_manager.create_engine', return_value=engine):
            AIGameManager.start_ai_vs_ai(
                "random", "random", num_games=1, recorder=recorder, temp_config=cfg,
            )

        sampling_moves = [
            m for m in recorder.move_calls
            if m["strategy"] == "sampling"
        ]
        self.assertGreater(
            len(sampling_moves), 0,
            "启用 temp_config 后应至少有一次 sampling 记录",
        )
        for m in sampling_moves:
            self.assertEqual(m["temperature"], 1.5)
            self.assertIsNone(m["top_k"])
            self.assertTrue(
                m["has_sampling_probs"],
                "sampling 步应传入 sampling_probs",
            )

    def test_record_move_without_temp_config_uses_argmax(self):
        """不传 temp_config 时，record_move 收到 strategy="argmax"。"""
        engine = _SpyEngine()
        recorder = _SpyRecorder()
        cfg = TempConfig(
            temp_moves=3, temp_side=TEMP_SIDE_BOTH, seed=1,
        )

        with patch('ai.game_manager.create_engine', return_value=engine):
            AIGameManager.start_ai_vs_ai(
                "random", "random", num_games=1, recorder=recorder, temp_config=cfg,
            )

        # 阈值（3 步）之后应为 argmax
        argmax_moves = [
            m for m in recorder.move_calls
            if m["strategy"] == "argmax"
        ]
        self.assertGreater(
            len(argmax_moves), 0,
            "超过阈值后应有 argmax 记录",
        )
        for m in argmax_moves:
            self.assertIsNone(m["temperature"])
            self.assertFalse(m["has_sampling_probs"])

    def test_record_move_sampling_probs_present(self):
        """sampling_probs 长度与空位数一致（>=1）。"""
        engine = _SpyEngine()
        recorder = _SpyRecorder()
        cfg = TempConfig(
            temperature=1.0, temp_moves=99, temp_side=TEMP_SIDE_BOTH, seed=1,
        )

        with patch('ai.game_manager.create_engine', return_value=engine):
            AIGameManager.start_ai_vs_ai(
                "random", "random", num_games=1, recorder=recorder, temp_config=cfg,
            )

        sampling_moves = [
            m for m in recorder.move_calls
            if m["strategy"] == "sampling"
        ]
        self.assertGreater(len(sampling_moves), 0)
        for m in sampling_moves:
            self.assertTrue(m["has_sampling_probs"])

    def test_record_move_sampling_probs_always_225(self):
        """sampling_probs 应始终为 225 维数组（已落子位填 0），而非仅空位长度。

        回归防护：此前 sampling_probs 仅包含空位值（长度 = 空位数），
        导致第 2 步起长度 < 225，ReplayViewer 的 Sampling 热力图无法渲染。
        """
        engine = _SpyEngine()
        recorder = _SpyRecorder()
        cfg = TempConfig(
            temperature=1.0, temp_moves=99, temp_side=TEMP_SIDE_BOTH, seed=1,
        )

        with patch('ai.game_manager.create_engine', return_value=engine):
            AIGameManager.start_ai_vs_ai(
                "random", "random", num_games=1, recorder=recorder, temp_config=cfg,
            )

        sampling_moves = [
            m for m in recorder.move_calls
            if m["strategy"] == "sampling"
        ]
        self.assertGreater(
            len(sampling_moves), 5,
            "应对弈超过 5 步以验证非满棋盘场景",
        )
        for i, m in enumerate(sampling_moves):
            self.assertEqual(
                m["sampling_probs_len"], 225,
                f"第 {i+1} 个 sampling 步的 sampling_probs 长度应为 225，"
                f"实际为 {m['sampling_probs_len']}"
            )


class _SpyRecorderFull(_SpyRecorder):
    """扩展版 spy recorder，额外保存 sampling_probs 实际数值列表。

    用于验证 sampling_probs 的具体值（不仅仅是长度/存在性）。
    """

    def record_move(self, game_id, input_vector, output_vector, position,
                    strategy=None, temperature=None, top_k=None,
                    sampling_probs=None, mcts_probs=None, mcts_value=None, mcts_q_values=None):
        super().record_move(
            game_id, input_vector, output_vector, position,
            strategy=strategy, temperature=temperature,
            top_k=top_k, sampling_probs=sampling_probs,
            mcts_probs=mcts_probs, mcts_value=mcts_value, mcts_q_values=mcts_q_values,
        )
        # 额外保存实际数值，供值级别断言使用
        self.move_calls[-1]["sampling_probs_values"] = (
            list(sampling_probs) if sampling_probs is not None else None
        )


class TestSamplingProbsOccupiedPositions(_TempSaveDirTestBase):
    """sampling_probs 应覆盖全部 225 个位置，包括已落子位（不被强制置 0）。

    Bug：此前已落子位在 sampling_probs 构建时被 `if board[x][y] == 0`
    过滤，导致 sampling_probs[occupied] = 0.0，而 AI output 的同位置
    值非零。log 归一化的 max_r 不同 → 两个热力图颜色比例不一致。

    修复后：温度变换对全部 225 个位置统一执行，已落子位同样记录
    p^(1/T)，视觉上 T=1.0 时 Sampling 热力图与 AI 热力图完全一致。
    """

    def test_sampling_probs_covers_occupied_positions(self):
        """T=1.0、均匀分布模型下，sampling_probs 各步之和应均为 1.0。

        SpyModel 返回均匀分布 [1/225]*225，T=1.0 → p^(1/1)=p=1/225。
        全部 225 格均应记录非零值，sum=1.0。

        若已落子位被置 0（Bug 状态）：第 2 步起 sum = (225-k)/225 < 1.0，
        其中 k 为已落子数，测试失败。
        """
        engine = _SpyEngine()
        recorder = _SpyRecorderFull()
        cfg = TempConfig(
            temperature=1.0, temp_moves=99, temp_side=TEMP_SIDE_BOTH, seed=1,
        )

        with patch('ai.game_manager.create_engine', return_value=engine):
            AIGameManager.start_ai_vs_ai(
                "random", "random", num_games=1, recorder=recorder, temp_config=cfg,
            )

        sampling_moves = [
            m for m in recorder.move_calls
            if m["strategy"] == "sampling"
        ]
        self.assertGreater(len(sampling_moves), 5, "应对弈超过 5 步")

        for i, m in enumerate(sampling_moves):
            sp = m["sampling_probs_values"]
            self.assertIsNotNone(sp, f"第 {i+1} 步 sampling_probs 不应为 None")
            self.assertEqual(len(sp), 225, f"第 {i+1} 步长度应为 225")
            actual_sum = sum(sp)
            self.assertAlmostEqual(
                actual_sum, 1.0, places=6,
                msg=(
                    f"第 {i+1} 步 sampling_probs 之和应为 1.0"
                    f"（T=1.0 时 p^1=p，全部 225 位应有非零值），"
                    f"实际 {actual_sum:.8f}。"
                    f"已落子位被置 0 会导致此值小于 1.0。"
                ),
            )


class TestSingleGameResult(_TempSaveDirTestBase):
    """SingleGameResult NamedTuple 及 play_single_game 返回值测试"""

    def test_namedtuple_attribute_access(self):
        """SN-01/SN-02: 属性访问 winner 和 temp_side_this_game"""
        from ai.game_manager import SingleGameResult
        result = SingleGameResult(winner=1, temp_side_this_game="black")
        self.assertEqual(result.winner, 1)
        self.assertEqual(result.temp_side_this_game, "black")

    def test_namedtuple_attribute_error(self):
        """SN-03: 错误属性抛出 AttributeError"""
        from ai.game_manager import SingleGameResult
        result = SingleGameResult(winner=1, temp_side_this_game=None)
        with self.assertRaises(AttributeError):
            _ = result.typo_field

    def testplay_single_game_returns_result_with_temp_side_black(self):
        """GM-01: play_single_game 真实执行，temp_side="black" 时
        返回 SingleGameResult 且 temp_side_this_game="black" """
        from ai.game_manager import SingleGameResult
        engine = _SpyEngine()
        recorder = _make_recorder()
        result = AIGameManager.play_single_game(
            engine, engine, recorder, 1,
            temp_config=TempConfig(temperature=1.0, temp_side="black",
                                   temp_moves=999, seed=42),
        )
        self.assertIsInstance(result, SingleGameResult)
        self.assertIn(result.winner, [1, 2, 3])
        self.assertEqual(result.temp_side_this_game, "black")

    def testplay_single_game_returns_result_with_temp_side_white(self):
        """GM-02: play_single_game 真实执行，temp_side="white" 时
        返回 SingleGameResult 且 temp_side_this_game="white" """
        from ai.game_manager import SingleGameResult
        engine = _SpyEngine()
        recorder = _make_recorder()
        result = AIGameManager.play_single_game(
            engine, engine, recorder, 1,
            temp_config=TempConfig(temperature=1.0, temp_side="white",
                                   temp_moves=999, seed=42),
        )
        self.assertIsInstance(result, SingleGameResult)
        self.assertIn(result.winner, [1, 2, 3])
        self.assertEqual(result.temp_side_this_game, "white")

    def testplay_single_game_returns_result_with_temp_side_both(self):
        """GM-03: play_single_game 真实执行，temp_side="both" 时
        返回 SingleGameResult 且 temp_side_this_game="both" """
        from ai.game_manager import SingleGameResult
        engine = _SpyEngine()
        recorder = _make_recorder()
        result = AIGameManager.play_single_game(
            engine, engine, recorder, 1,
            temp_config=TempConfig(temperature=1.0, temp_side="both",
                                   temp_moves=999, seed=42),
        )
        self.assertIsInstance(result, SingleGameResult)
        self.assertIn(result.winner, [1, 2, 3])
        self.assertEqual(result.temp_side_this_game, "both")

    def testplay_single_game_no_temp_config(self):
        """GM-04: 无 temp_config 时 temp_side_this_game 为 None"""
        from ai.game_manager import SingleGameResult
        engine = _SpyEngine()
        recorder = _make_recorder()
        result = AIGameManager.play_single_game(
            engine, engine, recorder, 1,
        )
        self.assertIsInstance(result, SingleGameResult)
        self.assertIn(result.winner, [1, 2, 3])
        self.assertIsNone(result.temp_side_this_game)

    def testplay_single_game_temp_side_random(self):
        """GM-05: play_single_game 真实执行，temp_side="random" 时
        返回 SingleGameResult 且 temp_side_this_game 为 "black" 或 "white" """
        from ai.game_manager import SingleGameResult
        engine = _SpyEngine()
        recorder = _make_recorder()
        result = AIGameManager.play_single_game(
            engine, engine, recorder, 1,
            temp_config=TempConfig(temperature=1.0, temp_side="random",
                                   temp_moves=999, seed=42),
        )
        self.assertIsInstance(result, SingleGameResult)
        self.assertIn(result.winner, [1, 2, 3])
        self.assertIn(result.temp_side_this_game, ("black", "white"))


class TestStartAiVsAiResultFormat(_TempSaveDirTestBase):
    """start_ai_vs_ai 结果列表测试（真实执行）"""

    def test_start_ai_vs_ai_result_has_temp_side_black(self):
        """SA-01: start_ai_vs_ai 真实执行，temp_side="black" →
        每项含 temp_side_this_game="black" """
        engine = _SpyEngine()
        recorder = _make_recorder()
        with patch('ai.game_manager.create_engine', return_value=engine):
            results = AIGameManager.start_ai_vs_ai(
                "random", "random", 2, recorder,
                temp_config=TempConfig(temperature=1.0, temp_side="black",
                                       temp_moves=999, seed=42),
            )
        for r in results:
            self.assertIn("temp_side_this_game", r)
            self.assertEqual(r["temp_side_this_game"], "black")

    def test_start_ai_vs_ai_no_temp_result_has_none(self):
        """SA-02: 无 temp_config 时结果 temp_side_this_game 为 None"""
        engine = _SpyEngine()
        recorder = _make_recorder()
        with patch('ai.game_manager.create_engine', return_value=engine):
            results = AIGameManager.start_ai_vs_ai(
                "random", "random", 2, recorder,
            )
        for r in results:
            self.assertIn("temp_side_this_game", r)
            self.assertIsNone(r["temp_side_this_game"])

    def test_start_ai_vs_ai_result_has_temp_side_both(self):
        """SA-03: temp_side="both" → 每项含 temp_side_this_game="both" """
        engine = _SpyEngine()
        recorder = _make_recorder()
        with patch('ai.game_manager.create_engine', return_value=engine):
            results = AIGameManager.start_ai_vs_ai(
                "random", "random", 2, recorder,
                temp_config=TempConfig(temperature=1.0, temp_side="both",
                                       temp_moves=999, seed=42),
            )
        for r in results:
            self.assertIn("temp_side_this_game", r)
            self.assertEqual(r["temp_side_this_game"], "both")

    def test_start_ai_vs_ai_result_has_temp_side_random(self):
        """SA-04: temp_side="random" → 每项含 temp_side_this_game="black"/"white" """
        engine = _SpyEngine()
        recorder = _make_recorder()
        with patch('ai.game_manager.create_engine', return_value=engine):
            results = AIGameManager.start_ai_vs_ai(
                "random", "random", 5, recorder,
                temp_config=TempConfig(temperature=1.0, temp_side="random",
                                       temp_moves=999, seed=2026),
            )
        temp_sides = {r["temp_side_this_game"] for r in results}
        # 每项应为 "black" 或 "white"
        for ts in temp_sides:
            self.assertIn(ts, ("black", "white"))


if __name__ == "__main__":
    unittest.main()
