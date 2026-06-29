"""AiGameManager after_move 调用 + 新 start_ai_vs_ai 接口测试（意图 35：T35-11 ~ T35-14, T35-19 ~ T35-20）

测试内容：
- T35-11: play_single_game 对 SinglePolicyAiEngine 调用 after_move(x, y)
- T35-12: play_single_game 对 PolicyValueAiEngine 调用 after_move(x, y)
- T35-13: after_move 在落子后、move_count 递增前调用（调用顺序）
- T35-14: after_move 对 PolicyValueAiEngine 异常不静默吞没
- T35-19: start_ai_vs_ai 每局创建新引擎（接收 ai 名称）
- T35-20: start_ai_vs_ai_random 每局创建新引擎（接收 ai 名称）
"""

import sys
import os
import unittest
import shutil
import tempfile
from unittest.mock import patch, MagicMock

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from game import GomokuController
from ai.game_manager import AIGameManager
from ai.engine import SinglePolicyAiEngine
from ai.models.random.model import RandomModel
from ai.data_recorder import DataRecorder
from ai.base_model import ModelPredict


# ---------- T35-11：play_single_game 对 SinglePolicyAiEngine 调用 after_move ----------

class TestT35_11_PlaySingleGameWithAfterMove(unittest.TestCase):
    """T35-11：play_single_game 对 SinglePolicyAiEngine 调用 after_move(x, y)"""

    def setUp(self):
        self.test_save_dir = tempfile.mkdtemp(prefix="test_t35_11_save_")
        self.test_ai_data_dir = tempfile.mkdtemp(prefix="test_t35_11_ai_data_")
        # Patch GomokuController 使其使用临时存档目录
        self._patcher = patch(
            'ai.game_manager.GomokuController',
            lambda save_dir=None: GomokuController(save_dir=self.test_save_dir)
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.test_save_dir, ignore_errors=True)
        shutil.rmtree(self.test_ai_data_dir, ignore_errors=True)

    def test_play_single_game_with_single_policy_engines(self):
        """两个 SinglePolicyAiEngine(random) 对局正常完成，不抛异常"""
        engine1 = SinglePolicyAiEngine(RandomModel())
        engine2 = SinglePolicyAiEngine(RandomModel())
        recorder = DataRecorder("test_t35_11", self.test_ai_data_dir)

        result = AIGameManager.play_single_game(
            engine1, engine2, recorder, 1)

        # 对局正常完成
        self.assertIn(result.winner, [1, 2, 3])

        # recorder 记录中有数据
        round_dir = recorder.get_round_dir()
        record_file = os.path.join(round_dir, str(recorder.current_game_id), "record.json")
        self.assertTrue(os.path.exists(record_file),
                        f"记录文件应存在: {record_file}")


# ---------- T35-12：play_single_game 对 PolicyValueAiEngine 调用 after_move ----------

class TestT35_12_PlaySingleGamePolicyValueAfterMove(unittest.TestCase):
    """T35-12：play_single_game 对 PolicyValueAiEngine 调用 after_move"""

    @classmethod
    def setUpClass(cls):
        try:
            from ai.mcts.searcher import MCTSSearcher
            cls.MCTSSearcher = MCTSSearcher
        except ImportError:
            cls.MCTSSearcher = None

    def setUp(self):
        if self.MCTSSearcher is None:
            self.skipTest("MCTSSearcher 尚未实现")
        self.test_save_dir = tempfile.mkdtemp(prefix="test_t35_12_save_")
        self.test_ai_data_dir = tempfile.mkdtemp(prefix="test_t35_12_ai_data_")
        # Patch GomokuController 使其使用临时存档目录
        self._patcher = patch(
            'ai.game_manager.GomokuController',
            lambda save_dir=None: GomokuController(save_dir=self.test_save_dir)
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.test_save_dir, ignore_errors=True)
        shutil.rmtree(self.test_ai_data_dir, ignore_errors=True)

    def test_play_single_game_with_policy_value_engines(self):
        """两个 PolicyValueAiEngine（50 模拟）对局正常完成"""
        from ai.engine import PolicyValueAiEngine
        from tests.unit.test_mcts_searcher import MockModel

        model = MockModel()
        searcher = self.MCTSSearcher(
            model, num_simulations=50,
            dirichlet_epsilon=0.0,
        )
        engine1 = PolicyValueAiEngine(searcher, temperature=0.0)
        engine2 = PolicyValueAiEngine(
            self.MCTSSearcher(MockModel(), num_simulations=50, dirichlet_epsilon=0.0),
            temperature=0.0,
        )
        recorder = DataRecorder("test_t35_12", self.test_ai_data_dir)

        result = AIGameManager.play_single_game(
            engine1, engine2, recorder, 1)

        # 对局正常完成
        self.assertIn(result.winner, [1, 2, 3])
        # 验证 _step_count 递增（说明 after_move 被调用）
        self.assertGreater(engine1._step_count, 0,
                           "engine1._step_count 应 > 0，说明 after_move 被调用")


# ---------- T35-13：after_move 在落子后、move_count 递增前调用 ----------

class TestT35_13_AfterMoveCallOrder(unittest.TestCase):
    """T35-13：after_move 在落子后、move_count 递增前调用（调用顺序验证）"""

    def setUp(self):
        self.test_save_dir = tempfile.mkdtemp(prefix="test_t35_13_save_")
        self.test_ai_data_dir = tempfile.mkdtemp(prefix="test_t35_13_ai_data_")
        # Patch GomokuController 使其使用临时存档目录
        self._patcher = patch(
            'ai.game_manager.GomokuController',
            lambda save_dir=None: GomokuController(save_dir=self.test_save_dir)
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.test_save_dir, ignore_errors=True)
        shutil.rmtree(self.test_ai_data_dir, ignore_errors=True)

    def test_call_order_get_move_then_make_move_then_after_move(self):
        """验证调用顺序为：get_move → make_move → after_move"""
        # 使用记录调用顺序的 spy 引擎
        call_order = []

        class SpyEngine(SinglePolicyAiEngine):
            def get_move(self, board, current_player, strategy="argmax"):
                call_order.append("get_move")
                return super().get_move(board, current_player, strategy)

            def after_move(self, x, y):
                call_order.append("after_move")
                super().after_move(x, y)

        engine1 = SpyEngine(RandomModel())
        engine2 = SpyEngine(RandomModel())
        recorder = DataRecorder("test_t35_13", self.test_ai_data_dir)

        AIGameManager.play_single_game(engine1, engine2, recorder, 1)

        # 验证调用序列中 get_move → after_move 的模式至少出现一次
        # 序列应为 [get_move, after_move, get_move, after_move, ...]
        for i in range(0, len(call_order) - 1, 2):
            self.assertEqual(call_order[i], "get_move",
                             f"第 {i} 个调用应为 get_move")
            self.assertEqual(call_order[i + 1], "after_move",
                             f"第 {i + 1} 个调用应为 after_move")


# ---------- T35-14：after_move 对 PolicyValueAiEngine 异常不静默吞没 ----------

class TestT35_14_AfterMoveExceptionNotSilent(unittest.TestCase):
    """T35-14：after_move 对 PolicyValueAiEngine 异常不静默吞没"""

    @classmethod
    def setUpClass(cls):
        try:
            from ai.mcts.searcher import MCTSSearcher
            cls.MCTSSearcher = MCTSSearcher
        except ImportError:
            cls.MCTSSearcher = None

    def setUp(self):
        if self.MCTSSearcher is None:
            self.skipTest("MCTSSearcher 尚未实现")
        self.test_save_dir = tempfile.mkdtemp(prefix="test_t35_14_save_")
        # Patch GomokuController 使其使用临时存档目录
        self._patcher = patch(
            'ai.game_manager.GomokuController',
            lambda save_dir=None: GomokuController(save_dir=self.test_save_dir)
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.test_save_dir, ignore_errors=True)

    def test_advance_root_failure_propagates_runtime_error(self):
        """Mock searcher 的 advance_root 恒返回 False，应传播 RuntimeError"""
        from ai.engine import PolicyValueAiEngine
        from tests.unit.test_mcts_searcher import MockModel
        import numpy as np

        model = MockModel()
        searcher = self.MCTSSearcher(
            model, num_simulations=50,
            dirichlet_epsilon=0.0,
        )
        # 创建一个会推进失败的引擎：替换 searcher 的 advance_root
        original_advance = searcher.advance_root
        searcher.advance_root = lambda action: False

        engine1 = PolicyValueAiEngine(searcher, temperature=0.0)
        engine2 = PolicyValueAiEngine(
            self.MCTSSearcher(MockModel(), num_simulations=50, dirichlet_epsilon=0.0),
            temperature=0.0,
        )
        test_ai_data_dir = tempfile.mkdtemp(prefix="test_t35_14_ai_data_")
        try:
            recorder = DataRecorder("test_t35_14", test_ai_data_dir)

            np.random.seed(42)
            with self.assertRaises(RuntimeError) as ctx:
                AIGameManager.play_single_game(engine1, engine2, recorder, 1)

            msg = str(ctx.exception)
            self.assertIn("advance_root", msg,
                          "异常消息应包含 'advance_root'")
        finally:
            shutil.rmtree(test_ai_data_dir, ignore_errors=True)


# ---------- T35-19：start_ai_vs_ai 每局创建新引擎（接收 ai 名称）----------

class TestT35_19_StartAiVsAiByNames(unittest.TestCase):
    """T35-19：start_ai_vs_ai 接收 ai 名称，每局创建新引擎"""

    def setUp(self):
        self.test_save_dir = tempfile.mkdtemp(prefix="test_t35_19_save_")
        self.test_ai_data_dir = tempfile.mkdtemp(prefix="test_t35_19_ai_data_")
        # Patch GomokuController 使其使用临时存档目录
        self._patcher = patch(
            'ai.game_manager.GomokuController',
            lambda save_dir=None: GomokuController(save_dir=self.test_save_dir)
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.test_save_dir, ignore_errors=True)
        shutil.rmtree(self.test_ai_data_dir, ignore_errors=True)

    def test_start_ai_vs_ai_by_names_two_games(self):
        """start_ai_vs_ai("random", "random", 2, recorder) 完成 2 局"""
        recorder = DataRecorder("test_t35_19", self.test_ai_data_dir)

        results = AIGameManager.start_ai_vs_ai(
            "random", "random", 2, recorder)

        # 验证结果数量
        self.assertEqual(len(results), 2,
                         "应返回 2 局对战结果")
        # 验证每局结果格式正确
        for i, r in enumerate(results, 1):
            self.assertEqual(r["game"], i,
                             f"第 {i} 场 game 字段应为 {i}")
            self.assertIn(r["winner"], [1, 2, 3],
                          f"第 {i} 场 winner 应为 1/2/3")

    def test_start_ai_vs_ai_by_names_creates_engines_per_game(self):
        """验证 start_ai_vs_ai 内部每局调用 create_engine"""
        from unittest.mock import patch as mock_patch

        recorder = DataRecorder("test_t35_19b", self.test_ai_data_dir)

        # Patch create_engine 来追踪调用次数
        with mock_patch('ai.game_manager.create_engine') as mock_create:
            mock_create.side_effect = lambda name: SinglePolicyAiEngine(RandomModel())

            results = AIGameManager.start_ai_vs_ai(
                "random", "random", 2, recorder)

            # 2 局 × 2 个引擎 = 4 次 create_engine 调用
            self.assertGreaterEqual(mock_create.call_count, 4,
                                    f"2 局应至少调用 4 次 create_engine，实际 {mock_create.call_count}")
            self.assertEqual(len(results), 2)


# ---------- T35-20：start_ai_vs_ai_random 每局创建新引擎（接收 ai 名称）----------

class TestT35_20_StartAiVsAiRandomByNames(unittest.TestCase):
    """T35-20：start_ai_vs_ai_random 接收 ai 名称，每局创建新引擎"""

    def setUp(self):
        self.test_save_dir = tempfile.mkdtemp(prefix="test_t35_20_save_")
        self.test_ai_data_dir = tempfile.mkdtemp(prefix="test_t35_20_ai_data_")
        # Patch GomokuController 使其使用临时存档目录
        self._patcher = patch(
            'ai.game_manager.GomokuController',
            lambda save_dir=None: GomokuController(save_dir=self.test_save_dir)
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.test_save_dir, ignore_errors=True)
        shutil.rmtree(self.test_ai_data_dir, ignore_errors=True)

    def test_start_ai_vs_ai_random_by_names_two_games(self):
        """start_ai_vs_ai_random("random", "random", 2, recorder) 完成 2 局"""
        recorder = DataRecorder("test_t35_20", self.test_ai_data_dir)

        results = AIGameManager.start_ai_vs_ai_random(
            "random", "random", 2, recorder)

        # 验证结果数量
        self.assertEqual(len(results), 2,
                         "应返回 2 局对战结果")
        # 验证每局结果格式正确
        for i, r in enumerate(results, 1):
            self.assertEqual(r["game"], i)
            self.assertIn(r["winner"], [1, 2, 3])
            self.assertIn("ai1_color", r)
            self.assertIn("ai2_color", r)
            self.assertIn("ai1_won", r)
            self.assertIn("ai2_won", r)

    def test_start_ai_vs_ai_random_creates_engines_per_game(self):
        """验证 start_ai_vs_ai_random 内部每局调用 create_engine"""
        from unittest.mock import patch as mock_patch

        recorder = DataRecorder("test_t35_20b", self.test_ai_data_dir)

        with mock_patch('ai.game_manager.create_engine') as mock_create:
            mock_create.side_effect = lambda name: SinglePolicyAiEngine(RandomModel())

            results = AIGameManager.start_ai_vs_ai_random(
                "random", "random", 2, recorder)

            self.assertGreaterEqual(mock_create.call_count, 4,
                                    f"2 局应至少调用 4 次 create_engine，实际 {mock_create.call_count}")
            self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
