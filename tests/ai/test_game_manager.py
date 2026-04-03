"""游戏管理器测试

测试 AIGameManager 类的所有公共方法。
"""

import sys
import os
import unittest
import shutil

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
        self.test_save_dir = "test_game_save"
        self.test_ai_data_dir = "test_ai_data_vs_ai"
        
        for d in [self.test_save_dir, self.test_ai_data_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
    
    def tearDown(self):
        """测试后清理"""
        for d in [self.test_save_dir, self.test_ai_data_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
    
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


if __name__ == "__main__":
    unittest.main()
