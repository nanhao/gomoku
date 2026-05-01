"""AI 对战脚本测试

测试 ai/battle.py 及 AIGameManager.start_ai_vs_ai_random 的相关功能。
分为主流程测试（primary）和边缘测试（edge）。
"""

import sys
import os
import unittest
import shutil
import tempfile
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径（支持从任何目录运行）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.game_manager import AIGameManager
from ai.engine import AIEngine
from ai.models.random.model import RandomModel
from ai.models.fc_v1.model import FCv1Model
from ai.models.cnn_v1.model import CNNV1Model
from ai.data_recorder import DataRecorder
import ai.battle as battle_module


# ============================================================
# 主流程测试（Primary Tests）
# 严格按照设计文档编写，测试核心业务主流程
# ============================================================

class TestStartAiVsAiRandom(unittest.TestCase):
    """AIGameManager.start_ai_vs_ai_random 主流程测试"""

    def setUp(self):
        """测试前准备：创建临时目录"""
        self.test_ai_data_dir = tempfile.mkdtemp(prefix="test_battle_ai_data_")
        self.test_save_dir = tempfile.mkdtemp(prefix="test_battle_save_")
        # Patch GomokuController 使其使用临时存档目录，避免写入 game/save/
        self._patcher = patch(
            'ai.game_manager.GomokuController',
            lambda save_dir=None: __import__('game').GomokuController(save_dir=self.test_save_dir)
        )
        self._patcher.start()

    def tearDown(self):
        """测试后清理：删除临时目录"""
        self._patcher.stop()
        if os.path.exists(self.test_ai_data_dir):
            shutil.rmtree(self.test_ai_data_dir)
        if os.path.exists(self.test_save_dir):
            shutil.rmtree(self.test_save_dir)

    def test_returns_correct_number_of_results(self):
        """主流程：返回结果列表长度等于对战局数"""
        engine1 = AIEngine(RandomModel())
        engine2 = AIEngine(RandomModel())
        recorder = DataRecorder("test_rnd", self.test_ai_data_dir)

        num_games = 3
        results = AIGameManager.start_ai_vs_ai_random(engine1, engine2, num_games, recorder)

        self.assertEqual(len(results), num_games)

    def test_result_contains_required_fields(self):
        """主流程：每个结果包含必要字段"""
        engine1 = AIEngine(RandomModel())
        engine2 = AIEngine(RandomModel())
        recorder = DataRecorder("test_rnd", self.test_ai_data_dir)

        results = AIGameManager.start_ai_vs_ai_random(engine1, engine2, 1, recorder)

        r = results[0]
        # 验证所有必要字段存在
        self.assertIn("game", r)
        self.assertIn("winner", r)
        self.assertIn("ai1_color", r)
        self.assertIn("ai2_color", r)
        self.assertIn("ai1_won", r)
        self.assertIn("ai2_won", r)

    def test_game_numbers_are_sequential(self):
        """主流程：局数编号从1开始递增"""
        engine1 = AIEngine(RandomModel())
        engine2 = AIEngine(RandomModel())
        recorder = DataRecorder("test_rnd", self.test_ai_data_dir)

        results = AIGameManager.start_ai_vs_ai_random(engine1, engine2, 4, recorder)

        for i, r in enumerate(results, 1):
            self.assertEqual(r["game"], i)

    def test_ai1_and_ai2_colors_are_complementary(self):
        """主流程：每局中 ai1_color 和 ai2_color 互为黑白（一黑一白）"""
        engine1 = AIEngine(RandomModel())
        engine2 = AIEngine(RandomModel())
        recorder = DataRecorder("test_rnd", self.test_ai_data_dir)

        results = AIGameManager.start_ai_vs_ai_random(engine1, engine2, 5, recorder)

        for r in results:
            # ai1 和 ai2 的颜色之和始终为 3（1+2=3）
            self.assertEqual(r["ai1_color"] + r["ai2_color"], 3)
            # 颜色只能是 1 或 2
            self.assertIn(r["ai1_color"], [1, 2])
            self.assertIn(r["ai2_color"], [1, 2])

    def test_winner_is_valid(self):
        """主流程：获胜方只能是 1（黑胜）、2（白胜）或 3（平局）"""
        engine1 = AIEngine(RandomModel())
        engine2 = AIEngine(RandomModel())
        recorder = DataRecorder("test_rnd", self.test_ai_data_dir)

        results = AIGameManager.start_ai_vs_ai_random(engine1, engine2, 3, recorder)

        for r in results:
            self.assertIn(r["winner"], [1, 2, 3])

    def test_ai_won_flags_consistent_with_winner_and_color(self):
        """主流程：ai1_won 和 ai2_won 与 winner/color 一致"""
        engine1 = AIEngine(RandomModel())
        engine2 = AIEngine(RandomModel())
        recorder = DataRecorder("test_rnd", self.test_ai_data_dir)

        results = AIGameManager.start_ai_vs_ai_random(engine1, engine2, 5, recorder)

        for r in results:
            winner = r["winner"]
            ai1_color = r["ai1_color"]
            ai2_color = r["ai2_color"]

            expected_ai1_won = (winner == ai1_color)
            expected_ai2_won = (winner == ai2_color)

            self.assertEqual(r["ai1_won"], expected_ai1_won)
            self.assertEqual(r["ai2_won"], expected_ai2_won)

    def test_random_side_assignment(self):
        """主流程：多局后执方分配具有随机性（不是全黑或全白）"""
        engine1 = AIEngine(RandomModel())
        engine2 = AIEngine(RandomModel())
        recorder = DataRecorder("test_rnd", self.test_ai_data_dir)

        # 运行较多局以确保随机性可被观察到
        num_games = 20
        results = AIGameManager.start_ai_vs_ai_random(engine1, engine2, num_games, recorder)

        ai1_as_black = sum(1 for r in results if r["ai1_color"] == 1)
        ai1_as_white = sum(1 for r in results if r["ai1_color"] == 2)

        # 验证两种执方都出现过（极低概率全是同色）
        self.assertGreater(ai1_as_black, 0, "ai1 从未执黑，随机性异常")
        self.assertGreater(ai1_as_white, 0, "ai1 从未执白，随机性异常")

    def test_record_files_created(self):
        """主流程：对战后记录文件被正确创建"""
        engine1 = AIEngine(RandomModel())
        engine2 = AIEngine(RandomModel())
        recorder = DataRecorder("test_rnd", self.test_ai_data_dir)

        num_games = 2
        AIGameManager.start_ai_vs_ai_random(engine1, engine2, num_games, recorder)

        round_dir = recorder.get_round_dir()
        for i in range(1, num_games + 1):
            record_file = os.path.join(round_dir, str(i), "record.json")
            self.assertTrue(os.path.exists(record_file), f"第 {i} 局记录文件不存在")


# ============================================================
# 边缘测试（Edge Tests）
# 测试非主流程的分支、边界条件
# ============================================================

class TestStartAiVsAiRandomEdge(unittest.TestCase):
    """AIGameManager.start_ai_vs_ai_random 边缘测试"""

    def setUp(self):
        """测试前准备"""
        self.test_ai_data_dir = tempfile.mkdtemp(prefix="test_battle_ai_data_edge_")
        self.test_save_dir = tempfile.mkdtemp(prefix="test_battle_save_edge_")
        # Patch GomokuController 使其使用临时存档目录，避免写入 game/save/
        self._patcher = patch(
            'ai.game_manager.GomokuController',
            lambda save_dir=None: __import__('game').GomokuController(save_dir=self.test_save_dir)
        )
        self._patcher.start()

    def tearDown(self):
        """测试后清理"""
        self._patcher.stop()
        if os.path.exists(self.test_ai_data_dir):
            shutil.rmtree(self.test_ai_data_dir)
        if os.path.exists(self.test_save_dir):
            shutil.rmtree(self.test_save_dir)

    def test_single_game(self):
        """边缘：只对战 1 局"""
        engine1 = AIEngine(RandomModel())
        engine2 = AIEngine(RandomModel())
        recorder = DataRecorder("test_rnd_edge", self.test_ai_data_dir)

        results = AIGameManager.start_ai_vs_ai_random(engine1, engine2, 1, recorder)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["game"], 1)

    def test_draw_game_both_not_won(self):
        """边缘：平局时两方都不算获胜"""
        engine1 = AIEngine(RandomModel())
        engine2 = AIEngine(RandomModel())
        recorder = DataRecorder("test_rnd_edge", self.test_ai_data_dir)

        results = AIGameManager.start_ai_vs_ai_random(engine1, engine2, 10, recorder)

        for r in results:
            if r["winner"] == 3:  # 平局
                self.assertFalse(r["ai1_won"], "平局时 ai1_won 应为 False")
                self.assertFalse(r["ai2_won"], "平局时 ai2_won 应为 False")


class TestBattleModule(unittest.TestCase):
    """ai/battle.py 脚本功能测试"""

    def setUp(self):
        """测试前准备"""
        self.test_ai_data_dir = tempfile.mkdtemp(prefix="test_battle_script_data_")

    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.test_ai_data_dir):
            shutil.rmtree(self.test_ai_data_dir)

    def test_load_model_random(self):
        """主流程：加载 random 模型（通过 registry.get_model）"""
        model = battle_module.get_model("random")
        self.assertIsInstance(model, RandomModel)
        self.assertEqual(model.name, "random")
        probs = model.predict([0] * 225 + [1])
        self.assertEqual(len(probs), 225)

    def test_load_model_fc_v1(self):
        """主流程：加载 fc_v1 模型（通过 registry.get_model）"""
        model = battle_module.get_model("fc_v1")
        self.assertIsInstance(model, FCv1Model)
        self.assertEqual(model.name, "fc_v1")
        probs = model.predict([0] * 225 + [1])
        self.assertEqual(len(probs), 225)

    def test_load_model_cnn(self):
        """主流程：加载 cnn_v1 模型（通过 registry.get_model）"""
        model = battle_module.get_model("cnn_v1")
        self.assertIsInstance(model, CNNV1Model)
        self.assertEqual(model.name, "cnn_v1")
        probs = model.predict([0] * 225 + [1])
        self.assertEqual(len(probs), 225)

    def test_load_model_unknown_raises_value_error(self):
        """边缘：加载未知 AI 名称时抛出 ValueError（通过 registry.get_model）"""
        with self.assertRaises(ValueError) as ctx:
            battle_module.get_model("nonexistent_ai")
        self.assertIn("nonexistent_ai", str(ctx.exception))

    def test_get_color_name(self):
        """主流程：颜色名称转换正确"""
        self.assertEqual(battle_module.get_color_name(1), "黑")
        self.assertEqual(battle_module.get_color_name(2), "白")

    def test_get_winner_name(self):
        """主流程：获胜方描述转换正确"""
        self.assertEqual(battle_module.get_winner_name(1), "黑方胜")
        self.assertEqual(battle_module.get_winner_name(2), "白方胜")
        self.assertEqual(battle_module.get_winner_name(3), "平局")

    def test_print_battle_results_output(self):
        """主流程：胜率统计输出正确"""
        results = [
            {"ai1_won": True,  "ai2_won": False, "winner": 1},
            {"ai1_won": False, "ai2_won": True,  "winner": 2},
            {"ai1_won": False, "ai2_won": False, "winner": 3},  # 平局
        ]
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            battle_module.print_battle_results("random", "fc_v1", results)
        output = f.getvalue()

        # 验证输出包含统计信息
        self.assertIn("总局数: 3", output)
        self.assertIn("random 胜: 1 局", output)
        self.assertIn("fc_v1 胜: 1 局", output)
        self.assertIn("平局: 1 局", output)

    def test_print_battle_results_zero_games(self):
        """边缘：空结果列表不出错，胜率为 0.0%"""
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            battle_module.print_battle_results("ai1", "ai2", [])
        output = f.getvalue()
        self.assertIn("总局数: 0", output)

    def test_run_battle_full_flow(self):
        """主流程：run_battle 完整流程（指定 random vs random，运行 2 局）"""
        # 直接 patch battle_module 中使用 DataRecorder 的 ai_data_dir
        # 通过替换 DataRecorder 类，将数据写入临时目录
        original_data_recorder = battle_module.DataRecorder

        class FakeDataRecorder(original_data_recorder):
            """将数据写入测试临时目录的 DataRecorder"""
            def __init__(self_, ai_name, ai_data_dir=None):
                super().__init__(ai_name, self.test_ai_data_dir)

        battle_module.DataRecorder = FakeDataRecorder
        # 同时 patch GomokuController 使其使用临时存档目录
        test_save_dir = self.test_ai_data_dir + "_save"
        os.makedirs(test_save_dir, exist_ok=True)
        with patch('ai.game_manager.GomokuController',
                   lambda _save_dir=None: __import__('game').GomokuController(save_dir=test_save_dir)):
            try:
                import io
                from contextlib import redirect_stdout
                f = io.StringIO()
                with redirect_stdout(f):
                    results = battle_module.run_battle("random", "random", 2)
            finally:
                # 恢复原始 DataRecorder
                battle_module.DataRecorder = original_data_recorder

        # 验证对战完成，结果数量正确
        self.assertEqual(len(results), 2)

        output = f.getvalue()
        # 输出应包含对战信息
        self.assertIn("第 1 局", output)
        self.assertIn("第 2 局", output)
        self.assertIn("=== 对战结果 ===", output)


if __name__ == "__main__":
    unittest.main()
