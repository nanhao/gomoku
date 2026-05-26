"""数据记录器测试

测试 DataRecorder 类的所有公共方法。
"""

import sys
import os
import unittest
import shutil
import json
import tempfile

# 添加项目根目录到路径（支持从任何目录运行）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.data_recorder import DataRecorder


class TestDataRecorder(unittest.TestCase):
    """数据记录器测试类"""
    
    def setUp(self):
        """测试前准备：创建临时测试目录"""
        self.test_dir = tempfile.mkdtemp(prefix="test_ai_data_")
    
    def tearDown(self):
        """测试后清理：删除临时测试目录"""
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
        """测试记录落子时游戏编号错误会被自动纠正"""
        recorder = DataRecorder("random", self.test_dir)
        recorder.new_game()  # 创建游戏 1
        
        # CR-003 修复：不再抛异常，改为自动纠正
        recorder.record_move(999, [0] * 226, [0] * 225, (7, 7))
        # 验证数据已被记录（在 save_game_result 时写入文件）
        self.assertEqual(len(recorder.current_game_data["history"]), 1)
    
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
        """测试保存结果时游戏编号错误会被自动纠正"""
        recorder = DataRecorder("random", self.test_dir)
        recorder.new_game()
        
        # CR-003 修复：不再抛异常，改为自动纠正并保存
        recorder.save_game_result(999, 1, [0] * 226)
        # 验证文件已创建
        round_dir = recorder.get_round_dir()
        game_dir = os.path.join(round_dir, str(recorder.current_game_id))
        record_file = os.path.join(game_dir, "record.json")
        self.assertTrue(os.path.exists(record_file))
    
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


    # ====================================================================
    # 第四十一次追加：温度采样字段记录测试
    # ====================================================================

    def test_record_move_with_sampling_strategy(self):
        """record_move 记录 sampling 策略时写入全部温度字段"""
        recorder = DataRecorder("test", self.test_dir)
        game_id = recorder.new_game()

        sampling_probs = [0.001] * 225
        sampling_probs[112] = 0.5  # (7,7)

        recorder.record_move(
            game_id,
            input_vector=[0] * 226,
            output_vector=[0.001] * 225,
            position=(7, 7),
            strategy="sampling",
            temperature=1.0,
            top_k=None,
            sampling_probs=sampling_probs,
        )

        recorder.save_game_result(game_id, 1, [0] * 226)

        # 验证 JSON 文件内容
        round_dir = recorder.get_round_dir()
        record_file = os.path.join(round_dir, "1", "record.json")
        with open(record_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        move = data["history"][0]
        self.assertEqual(move["strategy"], "sampling")
        self.assertEqual(move["temperature"], 1.0)
        self.assertIsNone(move["top_k"])
        self.assertEqual(len(move["sampling_probs"]), 225)
        self.assertEqual(move["sampling_probs"][112], 0.5)

    def test_record_move_with_argmax_strategy(self):
        """record_move 记录 argmax 策略时仅写入 strategy 字段"""
        recorder = DataRecorder("test", self.test_dir)
        game_id = recorder.new_game()

        recorder.record_move(
            game_id,
            input_vector=[0] * 226,
            output_vector=[0.001] * 225,
            position=(7, 7),
            strategy="argmax",
        )

        recorder.save_game_result(game_id, 1, [0] * 226)

        round_dir = recorder.get_round_dir()
        record_file = os.path.join(round_dir, "1", "record.json")
        with open(record_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        move = data["history"][0]
        self.assertEqual(move["strategy"], "argmax")
        self.assertNotIn("temperature", move)
        self.assertNotIn("sampling_probs", move)

    def test_record_move_backward_compatible(self):
        """record_move 不传新参数时保持旧格式兼容（无 strategy 等字段）"""
        recorder = DataRecorder("test", self.test_dir)
        game_id = recorder.new_game()

        # 使用旧签名调用（不传 strategy 等新参数）
        recorder.record_move(
            game_id,
            input_vector=[0] * 226,
            output_vector=[0.001] * 225,
            position=(7, 7),
        )

        recorder.save_game_result(game_id, 1, [0] * 226)

        round_dir = recorder.get_round_dir()
        record_file = os.path.join(round_dir, "1", "record.json")
        with open(record_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        move = data["history"][0]
        # 旧格式：不应有 strategy 字段（向后兼容）
        self.assertNotIn("strategy", move)
        self.assertNotIn("temperature", move)
        self.assertNotIn("sampling_probs", move)

    def test_record_move_sampling_mixed_with_argmax(self):
        """同一局中 sampling 和 argmax 步穿插记录"""
        recorder = DataRecorder("test", self.test_dir)
        game_id = recorder.new_game()

        # 第 1 步：sampling
        recorder.record_move(
            game_id,
            input_vector=[0] * 226,
            output_vector=[0.001] * 225,
            position=(7, 7),
            strategy="sampling",
            temperature=1.0,
            top_k=None,
            sampling_probs=[0.001] * 225,
        )
        # 第 2 步：argmax
        recorder.record_move(
            game_id,
            input_vector=[1] * 226,
            output_vector=[0.002] * 225,
            position=(7, 8),
            strategy="argmax",
        )
        # 第 3 步：sampling（不同温度）
        recorder.record_move(
            game_id,
            input_vector=[2] * 226,
            output_vector=[0.003] * 225,
            position=(8, 7),
            strategy="sampling",
            temperature=2.0,
            top_k=5,
            sampling_probs=[0.002] * 225,
        )

        recorder.save_game_result(game_id, 1, [0] * 226)

        round_dir = recorder.get_round_dir()
        record_file = os.path.join(round_dir, "1", "record.json")
        with open(record_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.assertEqual(len(data["history"]), 3)

        # 第 1 步：sampling, T=1.0, top_k=None
        m1 = data["history"][0]
        self.assertEqual(m1["strategy"], "sampling")
        self.assertEqual(m1["temperature"], 1.0)
        self.assertIsNone(m1["top_k"])

        # 第 2 步：argmax
        m2 = data["history"][1]
        self.assertEqual(m2["strategy"], "argmax")
        self.assertNotIn("temperature", m2)

        # 第 3 步：sampling, T=2.0, top_k=5
        m3 = data["history"][2]
        self.assertEqual(m3["strategy"], "sampling")
        self.assertEqual(m3["temperature"], 2.0)
        self.assertEqual(m3["top_k"], 5)

    def test_record_move_sampling_probs_length(self):
        """sampling_probs 长度校验：应为 225（满棋盘时为空位数）"""
        recorder = DataRecorder("test", self.test_dir)
        game_id = recorder.new_game()

        # 正常情况：225 个元素的 sampling_probs
        recorder.record_move(
            game_id,
            input_vector=[0] * 226,
            output_vector=[0.001] * 225,
            position=(7, 7),
            strategy="sampling",
            temperature=1.0,
            top_k=None,
            sampling_probs=[0.0] * 225,
        )

        recorder.save_game_result(game_id, 1, [0] * 226)

        round_dir = recorder.get_round_dir()
        record_file = os.path.join(round_dir, "1", "record.json")
        with open(record_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.assertEqual(len(data["history"][0]["sampling_probs"]), 225)


if __name__ == "__main__":
    unittest.main()
