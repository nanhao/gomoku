"""游戏管理器测试

测试 AIGameManager 类的所有公共方法。
"""

import sys
import os
import unittest
import shutil
import tempfile
from unittest.mock import patch

# 添加项目根目录到路径（支持从任何目录运行）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from game import GomokuController
from ai.game_manager import AIGameManager
from ai.engine import AIEngine
from ai.models.random.model import RandomModel
from ai.data_recorder import DataRecorder


class TestAIGameManager(unittest.TestCase):
    """AI 游戏管理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建测试用的存档目录
        self.test_save_dir = "test_game_save"
        if os.path.exists(self.test_save_dir):
            shutil.rmtree(self.test_save_dir)
        os.makedirs(self.test_save_dir)
        
        # 创建 controller 和引擎
        self.controller = GomokuController(self.test_save_dir)
        self.engine = AIEngine(RandomModel())
    
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.test_save_dir):
            shutil.rmtree(self.test_save_dir)
    
    def test_init(self):
        """测试初始化"""
        manager = AIGameManager(self.controller, self.engine, ai_player=2)
        
        self.assertEqual(manager.controller, self.controller)
        self.assertEqual(manager.engine, self.engine)
        self.assertEqual(manager.ai_player, 2)
    
    def test_start_pve_game_ai_white(self):
        """测试启动人机对战，AI 执白"""
        manager = AIGameManager(self.controller, self.engine, ai_player=2)
        result = manager.start_pve_game("test_game")
        
        # 验证游戏创建成功
        self.assertEqual(result["status"], "success")
        # AI 执白，不自动落子，当前玩家是黑方
        self.assertEqual(self.controller.get_current_player(), 1)
        # 历史记录为空
        self.assertEqual(len(self.controller.game.history), 0)
    
    def test_start_pve_game_ai_black(self):
        """测试启动人机对战，AI 执黑"""
        manager = AIGameManager(self.controller, self.engine, ai_player=1)
        result = manager.start_pve_game("test_game")
        
        # AI 执黑，自动执行第一步
        # 验证有落子记录
        self.assertEqual(len(self.controller.game.history), 1)
        # 当前玩家切换到白方
        self.assertEqual(self.controller.get_current_player(), 2)
    
    def test_player_move_no_ai_turn(self):
        """测试玩家落子，不触发 AI（轮到玩家）"""
        # AI 执白，玩家执黑
        manager = AIGameManager(self.controller, self.engine, ai_player=2)
        manager.start_pve_game("test_game")
        
        # 玩家落子
        result = manager.player_move(7, 7)
        
        # 验证返回结构
        self.assertIn("player_result", result)
        self.assertIn("ai_result", result)
        
        # 验证玩家落子成功
        self.assertEqual(result["player_result"]["status"], "success")
        
        # 验证 AI 自动落子
        self.assertEqual(result["ai_result"]["status"], "success")
        
        # 验证总共 2 步
        self.assertEqual(len(self.controller.game.history), 2)
    
    def test_player_move_game_over(self):
        """测试玩家落子后游戏结束，不触发 AI"""
        # AI 执白
        manager = AIGameManager(self.controller, self.engine, ai_player=2)
        manager.start_pve_game("test_game")
        
        # 模拟快结束的棋盘（黑方即将获胜）
        # 在 (7,3), (7,4), (7,5), (7,6) 放黑子
        for y in range(3, 7):
            self.controller.game.board[7][y] = 1
        
        # 玩家落子获胜
        result = manager.player_move(7, 7)
        
        # 验证只有玩家落子，没有 AI 落子
        self.assertIn("player_result", result)
        self.assertNotIn("ai_result", result)
        
        # 验证游戏结束
        self.assertTrue(self.controller.is_over())
        self.assertEqual(self.controller.get_winner(), 1)
    
    def test_player_move_error(self):
        """测试玩家落子错误"""
        manager = AIGameManager(self.controller, self.engine, ai_player=2)
        manager.start_pve_game("test_game")
        
        # 先落一子
        manager.player_move(7, 7)
        
        # 再次落同一位置（非法）
        result = manager.player_move(7, 7)
        
        # 验证返回错误
        self.assertIn("error", result["player_result"])
        # 不触发 AI
        self.assertNotIn("ai_result", result)
    
    def test_ai_move(self):
        """测试 AI 直接落子"""
        manager = AIGameManager(self.controller, self.engine, ai_player=1)
        manager.start_pve_game("test_game")  # AI 执黑，已自动落子
        
        # 当前是白方
        self.assertEqual(self.controller.get_current_player(), 2)
        
        # 手动触发 AI 落子（虽然通常不需要）
        result = manager.ai_move()
        
        # 验证落子成功
        self.assertEqual(result["status"], "success")
        # 总共 2 步（第一步自动 + 这次手动）
        self.assertEqual(len(self.controller.game.history), 2)


class TestAIGameManagerAIvsAI(unittest.TestCase):
    """AI 内战测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.test_save_dir = tempfile.mkdtemp(prefix="test_game_save_")
        self.test_ai_data_dir = "test_ai_data_vs_ai"
        
        if os.path.exists(self.test_ai_data_dir):
            shutil.rmtree(self.test_ai_data_dir)
        
        # Patch GomokuController 使其使用临时目录，避免写入 game/save/
        self._patcher = patch(
            'ai.game_manager.GomokuController',
            lambda save_dir=None: GomokuController(save_dir=self.test_save_dir)
        )
        self._patcher.start()
    
    def tearDown(self):
        """测试后清理"""
        self._patcher.stop()
        shutil.rmtree(self.test_save_dir, ignore_errors=True)
        if os.path.exists(self.test_ai_data_dir):
            shutil.rmtree(self.test_ai_data_dir)
    
    def test_start_ai_vs_ai_single_game(self):
        """测试 AI 内战，1 局"""
        engine1 = AIEngine(RandomModel())
        engine2 = AIEngine(RandomModel())
        recorder = DataRecorder("random", self.test_ai_data_dir)
        
        results = AIGameManager.start_ai_vs_ai(
            engine1, engine2, num_games=1, recorder=recorder,
            project_root=os.path.dirname(os.path.dirname(__file__))
        )
        
        # 验证结果
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["game"], 1)
        # 验证有获胜方（1, 2, 或 3 平局）
        self.assertIn(results[0]["winner"], [1, 2, 3])
        
        # 验证记录文件存在
        round_dir = recorder.get_round_dir()
        record_file = os.path.join(round_dir, "1", "record.json")
        self.assertTrue(os.path.exists(record_file))
    
    def test_start_ai_vs_ai_multiple_games(self):
        """测试 AI 内战，多局"""
        engine1 = AIEngine(RandomModel())
        engine2 = AIEngine(RandomModel())
        recorder = DataRecorder("random", self.test_ai_data_dir)
        
        num_games = 3
        results = AIGameManager.start_ai_vs_ai(
            engine1, engine2, num_games=num_games, recorder=recorder,
            project_root=os.path.dirname(os.path.dirname(__file__))
        )
        
        # 验证结果数量
        self.assertEqual(len(results), num_games)
        
        # 验证每局都有结果
        for i, result in enumerate(results, 1):
            self.assertEqual(result["game"], i)
            self.assertIn(result["winner"], [1, 2, 3])
        
        # 验证所有记录文件存在
        round_dir = recorder.get_round_dir()
        for i in range(1, num_games + 1):
            record_file = os.path.join(round_dir, str(i), "record.json")
            self.assertTrue(os.path.exists(record_file))
    
    def test_start_ai_vs_ai_unique_save_names(self):
        """测试 AI 内战多次调用时存档名称唯一性
        
        这是回归测试，确保修复了使用 id(controller) 导致存档名冲突的问题。
        训练周期会多次调用 start_ai_vs_ai，每轮都会创建新的对局，
        存档名称必须唯一，不能冲突。
        """
        engine1 = AIEngine(RandomModel())
        engine2 = AIEngine(RandomModel())
        
        project_root = os.path.dirname(os.path.dirname(__file__))
        
        # 模拟训练周期：连续调用两次 start_ai_vs_ai
        # 第一次调用（第1轮）
        recorder1 = DataRecorder("test_round", self.test_ai_data_dir)
        results1 = AIGameManager.start_ai_vs_ai(
            engine1, engine2, num_games=3, recorder=recorder1,
            project_root=project_root
        )
        
        # 第二次调用（第2轮）
        recorder2 = DataRecorder("test_round", self.test_ai_data_dir)
        results2 = AIGameManager.start_ai_vs_ai(
            engine1, engine2, num_games=3, recorder=recorder2,
            project_root=project_root
        )
        
        # 验证两轮都成功完成，没有存档冲突错误
        self.assertEqual(len(results1), 3)
        self.assertEqual(len(results2), 3)
        
        # 验证两轮的数据都正确记录
        round_dir1 = recorder1.get_round_dir()
        round_dir2 = recorder2.get_round_dir()
        
        # 两轮应该有不同的轮次目录
        self.assertNotEqual(round_dir1, round_dir2)
        
        # 验证每轮的记录文件都存在
        for i in range(1, 4):
            record_file1 = os.path.join(round_dir1, str(i), "record.json")
            record_file2 = os.path.join(round_dir2, str(i), "record.json")
            self.assertTrue(os.path.exists(record_file1), f"第一轮第{i}局记录不存在")
            self.assertTrue(os.path.exists(record_file2), f"第二轮第{i}局记录不存在")


class TestEndBoardDimension(unittest.TestCase):
    """回归测试：Bug M1 - end_board 应为 226 维而非 227 维

    board_to_vector 已返回 226 维（225棋盘+1落子方），
    多余的 append(0) 会使长度变为 227。
    """

    def setUp(self):
        """测试前准备"""
        self.test_save_dir = tempfile.mkdtemp(prefix="test_game_save_eb_")
        self.test_ai_data_dir = "test_ai_data_end_board"
        
        if os.path.exists(self.test_ai_data_dir):
            shutil.rmtree(self.test_ai_data_dir)
        
        # Patch GomokuController 使其使用临时目录，避免写入 game/save/
        self._patcher = patch(
            'ai.game_manager.GomokuController',
            lambda save_dir=None: GomokuController(save_dir=self.test_save_dir)
        )
        self._patcher.start()

    def tearDown(self):
        """测试后清理"""
        self._patcher.stop()
        shutil.rmtree(self.test_save_dir, ignore_errors=True)
        if os.path.exists(self.test_ai_data_dir):
            shutil.rmtree(self.test_ai_data_dir)

    def test_start_ai_vs_ai_end_board_dimension(self):
        """Bug M1: start_ai_vs_ai 保存的 end_board 应为 226 维"""
        engine1 = AIEngine(RandomModel())
        engine2 = AIEngine(RandomModel())
        recorder = DataRecorder("test_dim", self.test_ai_data_dir)

        AIGameManager.start_ai_vs_ai(
            engine1, engine2, num_games=1, recorder=recorder,
            project_root=os.path.dirname(os.path.dirname(__file__))
        )

        # 读取 record.json 验证 end_board 维度
        import json
        round_dir = recorder.get_round_dir()
        record_path = os.path.join(round_dir, "1", "record.json")
        with open(record_path, "r", encoding="utf-8") as f:
            record = json.load(f)

        # end_board 应为 226 维（225 棋盘格 + 1 落子方），而非 227
        self.assertEqual(len(record["end_board"]), 226,
                         f"end_board 应为 226 维，实际为 {len(record['end_board'])} 维")

    def test_start_ai_vs_ai_random_end_board_dimension(self):
        """Bug M1: start_ai_vs_ai_random 保存的 end_board 应为 226 维"""
        engine1 = AIEngine(RandomModel())
        engine2 = AIEngine(RandomModel())
        recorder = DataRecorder("test_dim_rnd", self.test_ai_data_dir)

        AIGameManager.start_ai_vs_ai_random(
            engine1, engine2, num_games=1, recorder=recorder
        )

        # 读取 record.json 验证 end_board 维度
        import json
        round_dir = recorder.get_round_dir()
        record_path = os.path.join(round_dir, "1", "record.json")
        with open(record_path, "r", encoding="utf-8") as f:
            record = json.load(f)

        # end_board 应为 226 维
        self.assertEqual(len(record["end_board"]), 226,
                         f"end_board 应为 226 维，实际为 {len(record['end_board'])} 维")


class TestGameIdConsistency(unittest.TestCase):
    """回归测试：Bug L3 - start_ai_vs_ai_random 应使用 recorder.current_game_id

    而非循环变量 game_num 作为 game_id，与 start_ai_vs_ai 保持一致。
    """

    def setUp(self):
        """测试前准备"""
        self.test_save_dir = tempfile.mkdtemp(prefix="test_game_save_id_")
        self.test_ai_data_dir = "test_ai_data_id"
        
        if os.path.exists(self.test_ai_data_dir):
            shutil.rmtree(self.test_ai_data_dir)
        
        # Patch GomokuController 使其使用临时目录，避免写入 game/save/
        self._patcher = patch(
            'ai.game_manager.GomokuController',
            lambda save_dir=None: GomokuController(save_dir=self.test_save_dir)
        )
        self._patcher.start()

    def tearDown(self):
        """测试后清理"""
        self._patcher.stop()
        shutil.rmtree(self.test_save_dir, ignore_errors=True)
        if os.path.exists(self.test_ai_data_dir):
            shutil.rmtree(self.test_ai_data_dir)

    def test_ai_vs_ai_random_uses_recorder_game_id(self):
        """Bug L3: start_ai_vs_ai_random 使用 recorder.current_game_id 而非 game_num"""
        engine1 = AIEngine(RandomModel())
        engine2 = AIEngine(RandomModel())

        # 使用已有的 recorder（模拟 recorder 被复用的情况）
        recorder = DataRecorder("test_gid", self.test_ai_data_dir)
        # 先调用一次 start_ai_vs_ai 使 recorder.current_game_id 递增
        AIGameManager.start_ai_vs_ai(
            engine1, engine2, num_games=1, recorder=recorder,
            project_root=os.path.dirname(os.path.dirname(__file__))
        )

        # 此时 recorder.current_game_id = 1
        # 再调用 start_ai_vs_ai_random，新游戏应从 id=2 开始
        results = AIGameManager.start_ai_vs_ai_random(
            engine1, engine2, num_games=1, recorder=recorder
        )

        # 验证对战成功完成（如果使用了错误的 game_num=1，会触发 game_id 不匹配异常）
        self.assertEqual(len(results), 1)
        # 验证 record.json 存在于正确的目录
        round_dir = recorder.get_round_dir()
        import json
        # 游戏应该在 id=2 的子目录中
        record_path = os.path.join(round_dir, "2", "record.json")
        self.assertTrue(os.path.exists(record_path),
                        "start_ai_vs_ai_random 应使用 recorder.current_game_id")


if __name__ == "__main__":
    unittest.main()
