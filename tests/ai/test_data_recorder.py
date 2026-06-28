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
from ai.constants import STRATEGY_ARGMAX, STRATEGY_SAMPLING


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
            strategy=STRATEGY_SAMPLING,
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
            strategy=STRATEGY_ARGMAX
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
            strategy=STRATEGY_SAMPLING,
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
            strategy=STRATEGY_ARGMAX,
        )
        # 第 3 步：sampling（不同温度）
        recorder.record_move(
            game_id,
            input_vector=[2] * 226,
            output_vector=[0.003] * 225,
            position=(8, 7),
            strategy=STRATEGY_SAMPLING,
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
            strategy=STRATEGY_SAMPLING,
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

    # ====================================================================
    # step_id 字段测试（step_id_matching_design.md）
    # ====================================================================

    def test_step_id_present_in_record(self):
        """record_move 自动生成 step_id，保存后 JSON 中包含该字段"""
        recorder = DataRecorder("test", self.test_dir)
        game_id = recorder.new_game()

        recorder.record_move(game_id, [0] * 226, [0] * 225, (7, 7))
        recorder.save_game_result(game_id, 1, [0] * 226)

        round_dir = recorder.get_round_dir()
        record_file = os.path.join(round_dir, "1", "record.json")
        with open(record_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        move = data["history"][0]
        self.assertIn("step_id", move)
        self.assertIsInstance(move["step_id"], str)
        self.assertGreater(len(move["step_id"]), 0)

    def test_step_id_uuid4_format(self):
        """step_id 为 UUID4 格式：8-4-4-4-12 连字符分隔"""
        recorder = DataRecorder("test", self.test_dir)
        game_id = recorder.new_game()

        recorder.record_move(game_id, [0] * 226, [0] * 225, (7, 7))
        recorder.save_game_result(game_id, 1, [0] * 226)

        round_dir = recorder.get_round_dir()
        record_file = os.path.join(round_dir, "1", "record.json")
        with open(record_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        step_id = data["history"][0]["step_id"]
        # UUID4 格式：xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
        # 简单校验：5 段用连字符分隔，总长度 36
        self.assertEqual(len(step_id), 36)
        parts = step_id.split('-')
        self.assertEqual(len(parts), 5)
        self.assertEqual(len(parts[0]), 8)
        self.assertEqual(len(parts[1]), 4)
        self.assertEqual(len(parts[2]), 4)
        self.assertEqual(len(parts[3]), 4)
        self.assertEqual(len(parts[4]), 12)
        # UUID4 的第 3 段首字符为 '4'
        self.assertEqual(parts[2][0], '4')

    def test_step_id_unique_across_moves(self):
        """不同步的 step_id 互不相同"""
        recorder = DataRecorder("test", self.test_dir)
        game_id = recorder.new_game()

        recorder.record_move(game_id, [0] * 226, [0] * 225, (7, 7))
        recorder.record_move(game_id, [1] * 226, [1] * 225, (7, 8))
        recorder.record_move(game_id, [2] * 226, [2] * 225, (8, 7))
        recorder.save_game_result(game_id, 1, [0] * 226)

        round_dir = recorder.get_round_dir()
        record_file = os.path.join(round_dir, "1", "record.json")
        with open(record_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        step_ids = [step["step_id"] for step in data["history"]]
        self.assertEqual(len(step_ids), 3)
        self.assertEqual(len(set(step_ids)), 3, f"step_id 不唯一: {step_ids}")

    def test_step_id_unique_across_games(self):
        """跨局 step_id 互不相同"""
        recorder = DataRecorder("test", self.test_dir)

        game1 = recorder.new_game()
        recorder.record_move(game1, [0] * 226, [0] * 225, (7, 7))
        recorder.save_game_result(game1, 1, [0] * 226)

        game2 = recorder.new_game()
        recorder.record_move(game2, [1] * 226, [1] * 225, (7, 8))
        recorder.save_game_result(game2, 2, [0] * 226)

        round_dir = recorder.get_round_dir()

        with open(os.path.join(round_dir, "1", "record.json"), 'r') as f:
            data1 = json.load(f)
        with open(os.path.join(round_dir, "2", "record.json"), 'r') as f:
            data2 = json.load(f)

        sid1 = data1["history"][0]["step_id"]
        sid2 = data2["history"][0]["step_id"]
        self.assertNotEqual(sid1, sid2, "跨局 step_id 不应相同")

    def test_step_id_first_field_in_json(self):
        """step_id 在 JSON entry 中为第一个字段"""
        recorder = DataRecorder("test", self.test_dir)
        game_id = recorder.new_game()

        recorder.record_move(game_id, [0] * 226, [0] * 225, (7, 7))
        recorder.save_game_result(game_id, 1, [0] * 226)

        round_dir = recorder.get_round_dir()
        record_file = os.path.join(round_dir, "1", "record.json")
        with open(record_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        move = data["history"][0]
        # Python 3.7+ dict 保证插入顺序
        first_key = next(iter(move))
        self.assertEqual(first_key, "step_id")


class TestDataRecorderTempSideThisGame(unittest.TestCase):
    """temp_side_this_game 字段写入测试"""

    def setUp(self):
        """测试前准备：创建临时测试目录"""
        self.test_dir = tempfile.mkdtemp(prefix="test_ai_data_")

    def tearDown(self):
        """测试后清理：删除临时测试目录"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _get_record_data(self, recorder, game_id):
        """辅助：读取指定 game_id 的 record.json"""
        game_dir = os.path.join(recorder.round_dir, str(game_id))
        record_file = os.path.join(game_dir, "record.json")
        with open(record_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def test_new_game_initializes_temp_side_this_game(self):
        """DR-01: new_game() 初始化 temp_side_this_game 为 None"""
        recorder = DataRecorder(ai_name="test_init", ai_data_dir=self.test_dir)
        game_id = recorder.new_game()
        self.assertIsNone(recorder.current_game_data["temp_side_this_game"])

    def test_save_game_result_temp_side_black(self):
        """DR-02: 单方采样 temp_side_this_game="black" 写入 record.json"""
        recorder = DataRecorder(ai_name="test_black", ai_data_dir=self.test_dir)
        game_id = recorder.new_game()
        board = [0] * 226
        recorder.save_game_result(game_id, 1, board, temp_side_this_game="black")
        data = self._get_record_data(recorder, game_id)
        self.assertEqual(data["temp_side_this_game"], "black")

    def test_save_game_result_temp_side_white(self):
        """DR-03: 单方采样 temp_side_this_game="white" 写入 record.json"""
        recorder = DataRecorder(ai_name="test_white", ai_data_dir=self.test_dir)
        game_id = recorder.new_game()
        board = [0] * 226
        recorder.save_game_result(game_id, 2, board, temp_side_this_game="white")
        data = self._get_record_data(recorder, game_id)
        self.assertEqual(data["temp_side_this_game"], "white")

    def test_save_game_result_temp_side_both(self):
        """DR-04: 双方采样 temp_side_this_game="both" 写入 record.json"""
        recorder = DataRecorder(ai_name="test_both", ai_data_dir=self.test_dir)
        game_id = recorder.new_game()
        board = [0] * 226
        recorder.save_game_result(game_id, 1, board, temp_side_this_game="both")
        data = self._get_record_data(recorder, game_id)
        self.assertEqual(data["temp_side_this_game"], "both")

    def test_save_game_result_temp_side_none(self):
        """DR-05: 无温度采样 temp_side_this_game=None 写入 record.json 为 null"""
        recorder = DataRecorder(ai_name="test_none", ai_data_dir=self.test_dir)
        game_id = recorder.new_game()
        board = [0] * 226
        recorder.save_game_result(game_id, 1, board, temp_side_this_game=None)
        data = self._get_record_data(recorder, game_id)
        self.assertIsNone(data["temp_side_this_game"])

    def test_save_game_result_default_no_temp_side(self):
        """DR-06: 不传 temp_side_this_game 时，默认值为 None，record.json 含 null"""
        recorder = DataRecorder(ai_name="test_default", ai_data_dir=self.test_dir)
        game_id = recorder.new_game()
        board = [0] * 226
        # 不传 temp_side_this_game，使用默认值 None
        recorder.save_game_result(game_id, 1, board)
        data = self._get_record_data(recorder, game_id)
        self.assertIsNone(data["temp_side_this_game"])


if __name__ == "__main__":
    unittest.main()
