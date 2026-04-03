"""数据记录器测试

测试 DataRecorder 类的所有公共方法。
"""

import sys
import os
import unittest
import shutil
import json

# 添加项目根目录到路径（支持从任何目录运行）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.data_recorder import DataRecorder


class TestDataRecorder(unittest.TestCase):
    """数据记录器测试类"""
    
    def setUp(self):
        """测试前准备：创建临时测试目录"""
        self.test_dir = "test_ai_data"
        # 清理可能存在的测试目录
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def tearDown(self):
        """测试后清理：删除测试目录"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_create_round_dir(self):
        """测试轮次目录创建"""
        recorder = DataRecorder("random", self.test_dir)
        round_dir = recorder.get_round_dir()
        
        # 验证目录存在
        self.assertTrue(os.path.exists(round_dir))
        # 验证目录名格式
        self.assertTrue(round_dir.endswith("random_1"))
    
    def test_create_multiple_round_dirs(self):
        """测试创建多个轮次目录"""
        recorder1 = DataRecorder("random", self.test_dir)
        recorder2 = DataRecorder("random", self.test_dir)
        
        round_dir1 = recorder1.get_round_dir()
        round_dir2 = recorder2.get_round_dir()
        
        # 验证两个目录不同
        self.assertNotEqual(round_dir1, round_dir2)
        # 验证第二个目录名递增
        self.assertTrue(round_dir2.endswith("random_2"))
    
    def test_new_game(self):
        """测试开始新游戏"""
        recorder = DataRecorder("random", self.test_dir)
        game_id = recorder.new_game()
        
        # 验证游戏编号从 1 开始
        self.assertEqual(game_id, 1)
        
        # 验证可以创建多个游戏
        game_id2 = recorder.new_game()
        self.assertEqual(game_id2, 2)
    
    def test_record_move(self):
        """测试记录落子"""
        recorder = DataRecorder("random", self.test_dir)
        game_id = recorder.new_game()
        
        input_vector = [0] * 226
        output_vector = [0] * 225
        position = (7, 7)
        
        # 记录落子
        recorder.record_move(game_id, input_vector, output_vector, position)
        
        # 验证不抛出异常
        # 实际数据在 save_game_result 时才写入文件
    
    def test_record_move_wrong_game_id(self):
        """测试记录落子时游戏编号错误"""
        recorder = DataRecorder("random", self.test_dir)
        recorder.new_game()  # 创建游戏 1
        
        with self.assertRaises(ValueError) as context:
            recorder.record_move(999, [0] * 226, [0] * 225, (7, 7))
        
        self.assertIn("游戏编号不匹配", str(context.exception))
    
    def test_save_game_result(self):
        """测试保存游戏结果"""
        recorder = DataRecorder("random", self.test_dir)
        game_id = recorder.new_game()
        
        # 记录一些落子
        recorder.record_move(game_id, [0] * 226, [0] * 225, (7, 7))
        recorder.record_move(game_id, [1] * 226, [1] * 225, (7, 8))
        
        # 保存结果
        end_board = [0] * 226
        recorder.save_game_result(game_id, 1, end_board)
        
        # 验证文件存在
        round_dir = recorder.get_round_dir()
        game_dir = os.path.join(round_dir, str(game_id))
        record_file = os.path.join(game_dir, "record.json")
        
        self.assertTrue(os.path.exists(record_file))
        
        # 验证文件内容
        with open(record_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.assertEqual(len(data["history"]), 2)
        self.assertEqual(data["winner"], 1)
        self.assertEqual(data["end_board"], end_board)
        
        # 验证历史记录内容
        first_move = data["history"][0]
        self.assertEqual(first_move["input"], [0] * 226)
        self.assertEqual(first_move["output"], [0] * 225)
        self.assertEqual(first_move["position"], [7, 7])
    
    def test_save_game_result_wrong_game_id(self):
        """测试保存结果时游戏编号错误"""
        recorder = DataRecorder("random", self.test_dir)
        recorder.new_game()
        
        with self.assertRaises(ValueError) as context:
            recorder.save_game_result(999, 1, [0] * 226)
        
        self.assertIn("游戏编号不匹配", str(context.exception))
    
    def test_full_game_record(self):
        """测试完整游戏记录流程"""
        recorder = DataRecorder("random", self.test_dir)
        
        # 第一局
        game1_id = recorder.new_game()
        self.assertEqual(game1_id, 1)
        
        # 模拟 5 步落子
        for i in range(5):
            recorder.record_move(game1_id, [i] * 226, [i] * 225, (i, i))
        
        recorder.save_game_result(game1_id, 1, [5] * 226)
        
        # 第二局
        game2_id = recorder.new_game()
        self.assertEqual(game2_id, 2)
        
        recorder.record_move(game2_id, [10] * 226, [10] * 225, (3, 3))
        recorder.save_game_result(game2_id, 2, [20] * 226)
        
        # 验证两个记录文件都存在
        round_dir = recorder.get_round_dir()
        
        file1 = os.path.join(round_dir, "1", "record.json")
        file2 = os.path.join(round_dir, "2", "record.json")
        
        self.assertTrue(os.path.exists(file1))
        self.assertTrue(os.path.exists(file2))
        
        # 验证内容
        with open(file1, 'r') as f:
            data1 = json.load(f)
        self.assertEqual(len(data1["history"]), 5)
        self.assertEqual(data1["winner"], 1)
        
        with open(file2, 'r') as f:
            data2 = json.load(f)
        self.assertEqual(len(data2["history"]), 1)
        self.assertEqual(data2["winner"], 2)


if __name__ == "__main__":
    unittest.main()
