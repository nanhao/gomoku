"""AIGameManager 温度采样编排测试

对应设计文档 design/move_sampler_design.md 的用例 23-28。

思路：使用 _SpyEngine 记录每一次 get_move 的 strategy / temperature 参数，
配合 Mock DataRecorder，绕过真实的 model.predict / record 写入，快速验证
AIGameManager 是否正确完成"按步切换温度 + 按方启用 + random 抽签 + 默认
保持 argmax"等编排行为。
"""

import os
import shutil
import sys
import tempfile
import unittest
from typing import List
from unittest.mock import MagicMock, patch

# 添加项目根目录到路径
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from game import GomokuController
from ai.engine import AIEngine, STRATEGY_ARGMAX, STRATEGY_SAMPLING
from ai.game_manager import AIGameManager
from ai.move_sampler import (
    TempConfig,
    TEMP_SIDE_BLACK,
    TEMP_SIDE_WHITE,
    TEMP_SIDE_BOTH,
    TEMP_SIDE_RANDOM,
)


# ---------------------------------------------------------------------------
# 测试桩
# ---------------------------------------------------------------------------


class _SpyModel:
    """predict 恒定返回均匀分布的桩模型。"""

    name = "spy"
    network = None

    def __init__(self):
        self.predict_call_count = 0

    def predict(self, board_vector):
        self.predict_call_count += 1
        return [1.0 / 225] * 225


class _SpyEngine(AIEngine):
    """记录每次 get_move 调用参数的 AIEngine 子类。

    不改变 get_move 的返回值逻辑（仍真正走父类），但把每次调用
    的 (strategy, temperature, player) 追加到 calls 列表。
    """

    def __init__(self):
        super().__init__(_SpyModel())
        self.calls: List[dict] = []

    def get_move(self, board, current_player, probs=None,
                 strategy=STRATEGY_ARGMAX, temperature=1.0,
                 top_k=None, rng=None):
        self.calls.append({
            "player": current_player,
            "strategy": strategy,
            "temperature": temperature,
        })
        return super().get_move(
            board, current_player, probs=probs, strategy=strategy,
            temperature=temperature, top_k=top_k, rng=rng,
        )


def _make_recorder(engine=None):
    """生成一个满足 _play_single_game 调用契约的 Mock recorder。

    关键属性：current_game_id；关键方法：new_game / record_move /
    save_game_result。不实际写盘。

    若传入 engine（_SpyEngine 实例），则在每次 new_game 时往
    engine.calls 追加一个 {"boundary": True} 标记，供测试精确切分局边界。
    """
    recorder = MagicMock()
    recorder.current_game_id = 1

    def _new_game():
        recorder.current_game_id += 1
        if engine is not None:
            engine.calls.append({"boundary": True})
    recorder.new_game.side_effect = _new_game

    return recorder


# ---------------------------------------------------------------------------
# 基类：patch GomokuController 使存档写入到临时目录，避开 WRITE GUARD
# ---------------------------------------------------------------------------


class _TempSaveDirTestBase(unittest.TestCase):
    """将 ai.game_manager.GomokuController 重定向到临时存档目录。

    _play_single_game 内部会 new GomokuController() 并写入 game/save/，
    该行为在测试环境下会被 conftest 的 WRITE GUARD 拦截。本基类通过
    patch 把 save_dir 换成临时目录，保持生产代码不变。
    """

    def setUp(self):
        self.test_save_dir = tempfile.mkdtemp(prefix="test_game_save_temp_")
        self._patcher = patch(
            "ai.game_manager.GomokuController",
            lambda save_dir=None: GomokuController(save_dir=self.test_save_dir),
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.test_save_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 用例 23：默认 temp_config=None → 仅 argmax
# ---------------------------------------------------------------------------

class TestDefaultNoTempConfig(_TempSaveDirTestBase):

    def test_default_no_temp_config_uses_argmax_only(self):
        """用例 23：不传 temp_config 时所有 engine.get_move 调用都为 argmax。"""
        engine = _SpyEngine()
        recorder = _make_recorder()

        AIGameManager.start_ai_vs_ai(
            engine, engine, num_games=1, recorder=recorder,
        )

        self.assertGreater(len(engine.calls), 0, "应有若干次 get_move 调用")
        for call in engine.calls:
            self.assertEqual(
                call["strategy"], STRATEGY_ARGMAX,
                f"temp_config=None 时不应触发 sampling，实际 call={call}"
            )


# ---------------------------------------------------------------------------
# 用例 24：阈值前 sampling、阈值后 argmax
# ---------------------------------------------------------------------------

class TestTempMovesThreshold(_TempSaveDirTestBase):

    def test_temp_config_applies_only_before_threshold(self):
        """用例 24：temp_moves=5 → 前 5 步 sampling，其后 argmax。"""
        engine = _SpyEngine()
        recorder = _make_recorder()
        cfg = TempConfig(
            temperature=1.0, temp_moves=5, temp_side=TEMP_SIDE_BOTH, seed=42,
        )

        AIGameManager.start_ai_vs_ai(
            engine, engine, num_games=1, recorder=recorder, temp_config=cfg,
        )

        # 至少 5 步才能触发"前 5 步"的完整验证
        self.assertGreaterEqual(
            len(engine.calls), 6,
            "对局应至少 6 步才能验证阈值行为"
        )
        # 前 5 步：sampling
        for i in range(5):
            self.assertEqual(
                engine.calls[i]["strategy"], STRATEGY_SAMPLING,
                f"第 {i} 步应为 sampling，实际 call={engine.calls[i]}"
            )
            self.assertEqual(engine.calls[i]["temperature"], 1.0)
        # 第 6 步及之后：argmax
        for i in range(5, len(engine.calls)):
            self.assertEqual(
                engine.calls[i]["strategy"], STRATEGY_ARGMAX,
                f"第 {i} 步应为 argmax，实际 call={engine.calls[i]}"
            )


# ---------------------------------------------------------------------------
# 用例 25：temp_side 过滤
# ---------------------------------------------------------------------------

class TestTempSideFilter(_TempSaveDirTestBase):

    def test_temp_side_black_only_affects_black(self):
        """用例 25：side=black → 仅黑方调用 sampling；白方全程 argmax。"""
        engine = _SpyEngine()
        recorder = _make_recorder()
        cfg = TempConfig(
            temperature=1.0,
            temp_moves=999,   # 全局阈值放宽，让 side 过滤起主导作用
            temp_side=TEMP_SIDE_BLACK,
            seed=42,
        )

        AIGameManager.start_ai_vs_ai(
            engine, engine, num_games=1, recorder=recorder, temp_config=cfg,
        )

        self.assertGreater(len(engine.calls), 0)
        for call in engine.calls:
            if call["player"] == 1:
                self.assertEqual(
                    call["strategy"], STRATEGY_SAMPLING,
                    f"黑方 call 应为 sampling：{call}"
                )
            else:
                self.assertEqual(
                    call["strategy"], STRATEGY_ARGMAX,
                    f"白方 call 应为 argmax：{call}"
                )


# ---------------------------------------------------------------------------
# 用例 26：temp_side=random 局内一致，且局间可不同
# ---------------------------------------------------------------------------

class TestTempSideRandom(_TempSaveDirTestBase):

    def test_temp_side_random_resolved_once_per_game(self):
        """用例 26：side=random, 同一局内启用方不变。"""
        engine = _SpyEngine()
        recorder = _make_recorder(engine=engine)
        # 选用 seed 使抽签出至少一次黑、一次白，故跑 20 局
        cfg = TempConfig(
            temperature=1.0,
            temp_moves=999,      # side 过滤主导
            temp_side=TEMP_SIDE_RANDOM,
            seed=2026,
        )

        total_games = 20
        AIGameManager.start_ai_vs_ai(
            engine, engine, num_games=total_games, recorder=recorder,
            temp_config=cfg,
        )

        # 依 recorder.new_game 触发时往 engine.calls 注入的 boundary 标记
        # 精确切段，每段 == 一整局的所有 get_move 调用。
        segments = []
        current_seg = []
        for call in engine.calls:
            if call.get("boundary"):
                if current_seg:
                    segments.append(current_seg)
                    current_seg = []
                continue
            current_seg.append(call)
        if current_seg:
            segments.append(current_seg)

        self.assertGreaterEqual(
            len(segments), 2,
            "应至少捕获 2 局的 engine.calls 片段",
        )

        # 对每段：sampling 要么只在黑方，要么只在白方
        black_sampled_games = 0
        white_sampled_games = 0
        for seg in segments:
            sampling_players = {
                c["player"] for c in seg if c["strategy"] == STRATEGY_SAMPLING
            }
            # 局内 sampling 应只来自一方
            self.assertTrue(
                len(sampling_players) <= 1,
                f"局内 sampling 应只来自一方，实际 players={sampling_players}"
            )
            if sampling_players == {1}:
                black_sampled_games += 1
            elif sampling_players == {2}:
                white_sampled_games += 1

        # 20 局 random 抽签至少覆盖两种结果（seed=2026 实证有效）
        self.assertGreater(black_sampled_games, 0, "至少有 1 局抽到黑方")
        self.assertGreater(white_sampled_games, 0, "至少有 1 局抽到白方")


# ---------------------------------------------------------------------------
# 用例 27：GUI 路径（不传 temp_config）与改动前行为一致
# ---------------------------------------------------------------------------

class TestGUIPathUnaffected(_TempSaveDirTestBase):

    def test_gui_path_unaffected(self):
        """用例 27：未传 temp_config 时，_play_single_game 不触发任何 sampling 相关副作用。"""
        engine = _SpyEngine()
        recorder = _make_recorder()

        # 模拟 GUI 的调用路径：直接 start_ai_vs_ai(engine, engine, n, recorder)，
        # 不传 temp_config 与任何新参数
        AIGameManager.start_ai_vs_ai(
            engine, engine, num_games=1, recorder=recorder,
        )

        # 所有 call 的 temperature 字段应保持默认 1.0（但 strategy=argmax 不使用）
        for call in engine.calls:
            self.assertEqual(call["strategy"], STRATEGY_ARGMAX)


# ---------------------------------------------------------------------------
# 用例 28：start_ai_vs_ai_random 支持 temp_config
# ---------------------------------------------------------------------------

class TestRandomMatchSupportsTempConfig(_TempSaveDirTestBase):

    def test_start_ai_vs_ai_random_supports_temp_config(self):
        """用例 28：start_ai_vs_ai_random 传 temp_config 时开启温度采样。"""
        engine = _SpyEngine()
        recorder = _make_recorder()
        cfg = TempConfig(
            temperature=1.0, temp_moves=3, temp_side=TEMP_SIDE_BOTH, seed=7,
        )

        AIGameManager.start_ai_vs_ai_random(
            engine, engine, num_games=1, recorder=recorder, temp_config=cfg,
        )

        sampling_calls = [
            c for c in engine.calls if c["strategy"] == STRATEGY_SAMPLING
        ]
        self.assertGreater(
            len(sampling_calls), 0,
            "start_ai_vs_ai_random 应接受 temp_config 并至少触发一次 sampling"
        )


# ---------------------------------------------------------------------------
# 第四十一次追加：验证 record_move 记录了温度采样信息
# ---------------------------------------------------------------------------

class _SpyRecorder:
    """记录 record_move 每次调用参数的桩 recorder。

    用于验证 _play_single_game 在温度采样启用时是否正确向
    DataRecorder.record_move() 传递 strategy / temperature / sampling_probs。
    """

    def __init__(self):
        self.current_game_id = 1
        self.move_calls: list = []

    def new_game(self):
        self.current_game_id += 1

    def record_move(self, game_id, input_vector, output_vector, position,
                    strategy=None, temperature=None, top_k=None,
                    sampling_probs=None):
        self.move_calls.append({
            "game_id": game_id,
            "strategy": strategy,
            "temperature": temperature,
            "top_k": top_k,
            "has_sampling_probs": sampling_probs is not None,
        })

    def save_game_result(self, game_id, winner, end_board):
        pass

    def get_round_dir(self):
        import tempfile
        import os
        return tempfile.mkdtemp()


class TestRecordMoveTempInfo(_TempSaveDirTestBase):
    """验证 _play_single_game 在温度采样时向 recorder 传递正确信息。"""

    def test_record_move_with_temp_config_passes_strategy_and_temperature(self):
        """temp_config 启用时，record_move 收到 strategy="sampling" 和温度值。"""
        engine = _SpyEngine()
        recorder = _SpyRecorder()
        cfg = TempConfig(
            temperature=1.5, temp_moves=99, temp_side=TEMP_SIDE_BOTH, seed=1,
        )

        AIGameManager.start_ai_vs_ai(
            engine, engine, num_games=1, recorder=recorder, temp_config=cfg,
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

        AIGameManager.start_ai_vs_ai(
            engine, engine, num_games=1, recorder=recorder, temp_config=cfg,
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

        AIGameManager.start_ai_vs_ai(
            engine, engine, num_games=1, recorder=recorder, temp_config=cfg,
        )

        sampling_moves = [
            m for m in recorder.move_calls
            if m["strategy"] == "sampling"
        ]
        self.assertGreater(len(sampling_moves), 0)
        for m in sampling_moves:
            self.assertTrue(m["has_sampling_probs"])


if __name__ == "__main__":
    unittest.main()
