"""AIGameManager.play_single_game 单元测试

测试公开静态方法 play_single_game 的核心逻辑，
包括主流程（正常完成一局）和边界条件（棋盘已满时的 ValueError 处理）。
"""

import sys
import os
import unittest
import shutil
import tempfile
from unittest.mock import patch

# 添加项目根目录到路径（支持从任何目录运行）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from game import GomokuController
from ai.game_manager import AIGameManager
from ai.engine import AIEngine
from ai.models.random.model import RandomModel
from ai.data_recorder import DataRecorder


class TestPlaySingleGameMainFlow(unittest.TestCase):
    """play_single_game 主流程测试"""

    def setUp(self):
        """测试前准备：创建临时目录"""
        self.temp_dir = tempfile.mkdtemp(prefix="test_play_single_")
        self.test_ai_data_dir = os.path.join(self.temp_dir, "ai_data")
        os.makedirs(self.test_ai_data_dir, exist_ok=True)
        
        self.test_save_dir = tempfile.mkdtemp(prefix="test_save_single_")
        # Patch GomokuController 使其使用临时目录，避免写入 game/save/
        self._patcher = patch(
            'ai.game_manager.GomokuController',
            lambda save_dir=None: GomokuController(save_dir=self.test_save_dir)
        )
        self._patcher.start()

    def tearDown(self):
        """测试后清理：删除临时目录"""
        self._patcher.stop()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        shutil.rmtree(self.test_save_dir, ignore_errors=True)

    def test_normal_game_returns_winner(self):
        """正常完成一局，返回有效的 SingleGameResult"""
        engine_black = AIEngine(RandomModel())
        engine_white = AIEngine(RandomModel())
        recorder = DataRecorder("test_unit", self.test_ai_data_dir)

        result = AIGameManager.play_single_game(
            engine_black, engine_white, recorder, game_num=1
        )

        # result.winner 必须是有效值
        self.assertIn(result.winner, [1, 2, 3])
        # temp_side_this_game 无 temp_config 时为 None
        self.assertIsNone(result.temp_side_this_game)

    def test_normal_game_records_data(self):
        """正常完成一局，recorder 记录了数据文件"""
        engine_black = AIEngine(RandomModel())
        engine_white = AIEngine(RandomModel())
        recorder = DataRecorder("test_unit_rec", self.test_ai_data_dir)

        AIGameManager.play_single_game(
            engine_black, engine_white, recorder, game_num=1
        )

        # 验证 record.json 存在
        round_dir = recorder.get_round_dir()
        record_path = os.path.join(round_dir, str(recorder.current_game_id), "record.json")
        self.assertTrue(os.path.exists(record_path), "record.json 应该存在")

    def test_normal_game_record_has_required_fields(self):
        """正常完成一局，record.json 包含必要字段"""
        import json
        engine_black = AIEngine(RandomModel())
        engine_white = AIEngine(RandomModel())
        recorder = DataRecorder("test_unit_fields", self.test_ai_data_dir)

        AIGameManager.play_single_game(
            engine_black, engine_white, recorder, game_num=1
        )

        # 读取 record.json 验证字段
        round_dir = recorder.get_round_dir()
        game_id = recorder.current_game_id
        record_path = os.path.join(round_dir, str(game_id), "record.json")
        with open(record_path, "r", encoding="utf-8") as f:
            record = json.load(f)

        # 必须包含 history、winner、end_board
        self.assertIn("history", record)
        self.assertIn("winner", record)
        self.assertIn("end_board", record)

        # history 应该非空（至少有一步落子）
        self.assertGreater(len(record["history"]), 0)

    def test_normal_game_end_board_dimension(self):
        """正常完成一局，end_board 为 226 维"""
        import json
        engine_black = AIEngine(RandomModel())
        engine_white = AIEngine(RandomModel())
        recorder = DataRecorder("test_unit_dim", self.test_ai_data_dir)

        AIGameManager.play_single_game(
            engine_black, engine_white, recorder, game_num=1
        )

        round_dir = recorder.get_round_dir()
        game_id = recorder.current_game_id
        record_path = os.path.join(round_dir, str(game_id), "record.json")
        with open(record_path, "r", encoding="utf-8") as f:
            record = json.load(f)

        # end_board 应为 226 维（225 棋盘 + 1 落子方）
        self.assertEqual(len(record["end_board"]), 226,
                         f"end_board 应为 226 维，实际为 {len(record['end_board'])} 维")

    def test_multiple_games_independent(self):
        """多局游戏独立运行，每局都成功完成"""
        engine_black = AIEngine(RandomModel())
        engine_white = AIEngine(RandomModel())
        recorder = DataRecorder("test_unit_multi", self.test_ai_data_dir)

        winners = []
        for game_num in range(1, 4):
            result = AIGameManager.play_single_game(
                engine_black, engine_white, recorder, game_num=game_num
            )
            winners.append(result.winner)

        # 每局都返回有效 winner
        for winner in winners:
            self.assertIn(winner, [1, 2, 3])

        # 验证每局都有对应的 record.json
        round_dir = recorder.get_round_dir()
        for game_id in range(1, 4):
            record_path = os.path.join(round_dir, str(game_id), "record.json")
            self.assertTrue(os.path.exists(record_path),
                            f"第 {game_id} 局的 record.json 应该存在")


class TestPlaySingleGameEdgeCases(unittest.TestCase):
    """play_single_game 边界条件测试"""

    def setUp(self):
        """测试前准备：创建临时目录"""
        self.temp_dir = tempfile.mkdtemp(prefix="test_play_edge_")
        self.test_ai_data_dir = os.path.join(self.temp_dir, "ai_data")
        os.makedirs(self.test_ai_data_dir, exist_ok=True)
        
        self.test_save_dir = tempfile.mkdtemp(prefix="test_save_edge_")
        # Patch GomokuController 使其使用临时目录，避免写入 game/save/
        self._patcher = patch(
            'ai.game_manager.GomokuController',
            lambda save_dir=None: GomokuController(save_dir=self.test_save_dir)
        )
        self._patcher.start()

    def tearDown(self):
        """测试后清理：删除临时目录"""
        self._patcher.stop()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        shutil.rmtree(self.test_save_dir, ignore_errors=True)

    def test_game_without_extra_params(self):
        """仅传入必需参数时，正常运行不报错"""
        engine_black = AIEngine(RandomModel())
        engine_white = AIEngine(RandomModel())
        recorder = DataRecorder("test_unit_no_cb", self.test_ai_data_dir)

        result = AIGameManager.play_single_game(
            engine_black, engine_white, recorder, game_num=1
        )
        self.assertIn(result.winner, [1, 2, 3])

    def test_recorder_game_id_incremented(self):
        """游戏完成后，recorder 的 current_game_id 应正确递增"""
        engine_black = AIEngine(RandomModel())
        engine_white = AIEngine(RandomModel())
        recorder = DataRecorder("test_unit_gid", self.test_ai_data_dir)

        # 初始 current_game_id 为 0（DataRecorder 初始化时）
        initial_id = recorder.current_game_id

        AIGameManager.play_single_game(
            engine_black, engine_white, recorder, game_num=1
        )

        # play_single_game 内部调用 recorder.new_game()，current_game_id 应该递增
        self.assertEqual(recorder.current_game_id, initial_id + 1)

    def test_recorder_records_moves_with_correct_game_id(self):
        """recorder 使用 current_game_id 记录数据，而非 game_num 参数"""
        import json
        engine_black = AIEngine(RandomModel())
        engine_white = AIEngine(RandomModel())

        # 创建一个已经有一局记录的 recorder
        recorder = DataRecorder("test_unit_gid_match", self.test_ai_data_dir)
        # 先手动递增 game_id
        recorder.new_game()  # current_game_id = 1
        initial_id = recorder.current_game_id

        # 再调用 play_single_game，应该使用 new_game 递增后的 id
        AIGameManager.play_single_game(
            engine_black, engine_white, recorder, game_num=1
        )

        # current_game_id 应该再递增一次
        self.assertEqual(recorder.current_game_id, initial_id + 1)

        # 数据文件应该在正确的 game_id 目录下
        round_dir = recorder.get_round_dir()
        record_path = os.path.join(round_dir, str(initial_id + 1), "record.json")
        self.assertTrue(os.path.exists(record_path),
                        f"record.json 应在 game_id={initial_id + 1} 的目录下")


class TestPlaySingleGameBoardFull(unittest.TestCase):
    """play_single_game 棋盘已满时的 ValueError 处理测试

    使用模拟引擎构造棋盘已满的场景，
    验证 play_single_game 能正确处理 ValueError 并返回结果。
    """

    def setUp(self):
        """测试前准备：创建临时目录"""
        self.temp_dir = tempfile.mkdtemp(prefix="test_play_full_")
        self.test_ai_data_dir = os.path.join(self.temp_dir, "ai_data")
        os.makedirs(self.test_ai_data_dir, exist_ok=True)
        
        self.test_save_dir = tempfile.mkdtemp(prefix="test_save_full_")
        # Patch GomokuController 使其使用临时目录，避免写入 game/save/
        self._patcher = patch(
            'ai.game_manager.GomokuController',
            lambda save_dir=None: GomokuController(save_dir=self.test_save_dir)
        )
        self._patcher.start()

    def tearDown(self):
        """测试后清理：删除临时目录"""
        self._patcher.stop()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        shutil.rmtree(self.test_save_dir, ignore_errors=True)

    def test_board_full_value_error_handled(self):
        """当引擎抛出 ValueError（棋盘已满）时，应正常结束并返回 winner"""
        from unittest.mock import MagicMock, patch
        import numpy as np

        # 创建一个会在 get_move 时抛出 ValueError 的 mock 引擎
        engine_black = AIEngine(RandomModel())
        engine_white = AIEngine(RandomModel())

        # 构造一个几乎满的棋盘场景：
        # 使用 mock 让 engine_black.get_move 在特定步数抛出 ValueError
        original_get_move = engine_black.get_move
        call_count = [0]

        def mock_get_move(board, current_player, probs=None):
            call_count[0] += 1
            # 第一次调用后抛出 ValueError（模拟棋盘已满）
            if call_count[0] > 1:
                raise ValueError("棋盘已满，无空位可落子")
            return original_get_move(board, current_player, probs)

        engine_black.get_move = mock_get_move

        recorder = DataRecorder("test_unit_full", self.test_ai_data_dir)

        # 应该正常返回，不抛异常
        result = AIGameManager.play_single_game(
            engine_black, engine_white, recorder, game_num=1
        )

        # result.winner 应该是有效值
        self.assertIn(result.winner, [0, 1, 2, 3])

    def test_board_full_still_saves_result(self):
        """棋盘已满时，仍然保存游戏结果"""
        from unittest.mock import MagicMock

        engine_black = AIEngine(RandomModel())
        engine_white = AIEngine(RandomModel())

        # 让 get_move 第一次就抛出 ValueError
        engine_black.get_move = lambda board, player, probs=None: (
            _ for _ in ()
        ).throw(ValueError("棋盘已满，无空位可落子"))

        recorder = DataRecorder("test_unit_full_save", self.test_ai_data_dir)

        AIGameManager.play_single_game(
            engine_black, engine_white, recorder, game_num=1
        )

        # 验证结果仍然被保存
        round_dir = recorder.get_round_dir()
        game_id = recorder.current_game_id
        record_path = os.path.join(round_dir, str(game_id), "record.json")
        self.assertTrue(os.path.exists(record_path),
                        "棋盘已满时仍然应该保存 record.json")


if __name__ == "__main__":
    unittest.main()
