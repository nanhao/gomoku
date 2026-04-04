"""评价器测试

测试 Evaluator 类的所有功能。
"""

import sys
import os
import unittest
import json
import shutil

# 添加项目根目录到路径（支持从任何目录运行）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.evaluator import Evaluator


class TestEvaluator(unittest.TestCase):
    """评价器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.test_dir = "test_evaluator_data"
        self.ai_data_dir = os.path.join(self.test_dir, "ai_data")
        self.train_data_dir = os.path.join(self.test_dir, "train_data")
        
        # 清理并创建测试目录
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.ai_data_dir)
        os.makedirs(self.train_data_dir)
        
        # 创建评价器
        self.evaluator = Evaluator(self.ai_data_dir, self.train_data_dir)
    
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_constants(self):
        """测试常量定义"""
        self.assertEqual(Evaluator.MAX_REWARD, 1000000)
        self.assertEqual(Evaluator.MIN_REWARD, -1000000)
        self.assertEqual(Evaluator.DECAY_RATE, 0.9)
    
    def test_get_sample_path(self):
        """测试生成 sample 路径"""
        record_file = os.path.join(self.ai_data_dir, "random_1", "1", "record.json")
        sample_file = self.evaluator._get_sample_path(record_file)
        
        expected = os.path.join(self.train_data_dir, "random_1", "1", "sample.json")
        self.assertEqual(sample_file, expected)
    
    def test_initialize_sample(self):
        """测试初始化 sample"""
        # 构造测试数据
        history = [
            {
                "input": [0] * 225 + [1],  # 空棋盘，黑方落子
                "output": [0] * 225,       # 225 维 reward
                "position": [7, 7]
            }
        ]
        
        # 在 (7, 7) 位置放置黑子
        history[0]["input"][7 * 15 + 7] = 1
        
        sample_dict = self.evaluator._initialize_sample(history)
        
        # 验证结果
        self.assertEqual(len(sample_dict), 1)
        self.assertEqual(len(sample_dict[0]["mask"]), 225)
        self.assertEqual(len(sample_dict[0]["output"]), 225)
        
        # 验证 (7, 7) 位置的 mask 为 1（当前版本 mask 全为 1）
        pos_7_7_idx = 7 * 15 + 7
        self.assertEqual(sample_dict[0]["mask"][pos_7_7_idx], 1)
        
        # 验证 (7, 7) 位置的 reward 为 MIN_REWARD
        self.assertEqual(sample_dict[0]["output"][pos_7_7_idx], Evaluator.MIN_REWARD)
        
        # 验证其他位置的 mask 也为 1（当前版本 mask 全部为 1）
        self.assertEqual(sample_dict[0]["mask"][0], 1)
    
    def test_compute_rewards_black_wins(self):
        """测试黑方获胜时的 reward 计算（新算法：只给五子连线的5步分配 reward）"""
        # 构造测试数据：黑方形成五子连线获胜
        history = []
        
        # 黑方形成垂直五子连线：(3,7), (4,7), (5,7), (6,7), (7,7)
        # 中间夹杂白方落子
        moves = [
            ([3, 7], 1),   # 黑 a1
            ([0, 0], 2),   # 白 b1
            ([4, 7], 1),   # 黑 a2
            ([0, 1], 2),   # 白 b2
            ([5, 7], 1),   # 黑 a3
            ([0, 2], 2),   # 白 b3
            ([6, 7], 1),   # 黑 a4
            ([0, 3], 2),   # 白 b4
            ([7, 7], 1),   # 黑 a5（获胜）
        ]
        
        board = [0] * 225
        for pos, player in moves:
            # 先更新棋盘再记录（确保最后一步的棋盘状态正确）
            board[pos[0] * 15 + pos[1]] = player
            input_vec = board.copy() + [player]
            output = [0] * 225
            
            history.append({
                "input": input_vec,
                "output": output,
                "position": pos
            })
        
        # 初始化
        sample_dict = self.evaluator._initialize_sample(history)
        
        # 计算 reward（黑方获胜）
        self.evaluator._compute_rewards(sample_dict, history, winner=1)
        
        max_r = Evaluator.MAX_REWARD
        r = Evaluator.DECAY_RATE
        
        # 验证：最后一步（黑 a5, step 8, position [7,7]）应该是 MAX_REWARD
        # 这是五子连线的最后一步（j5）
        last_step = sample_dict[8]
        reward_idx = 7 * 15 + 7  # [7, 7]
        self.assertEqual(last_step["output"][reward_idx], max_r)
        self.assertEqual(last_step["mask"][reward_idx], 1)
        
        # 验证：黑方倒数第二步（a4, step 6, position [6,7]）应该是 MAX_REWARD * r
        # 这是五子连线的 j4
        step_a4 = sample_dict[6]
        reward_idx_a4 = 6 * 15 + 7  # [6, 7]
        self.assertEqual(step_a4["output"][reward_idx_a4], int(max_r * r))
    
    def test_evaluate_record(self):
        """测试完整评价流程"""
        # 创建测试记录文件
        round_dir = os.path.join(self.ai_data_dir, "test_round")
        game_dir = os.path.join(round_dir, "1")
        os.makedirs(game_dir)
        
        record_file = os.path.join(game_dir, "record.json")
        
        # 构造简单的记录数据
        history = []
        board = [0] * 225
        
        # 黑方获胜：3 步
        moves = [
            ([7, 7], 1),
            ([7, 8], 2),
            ([6, 6], 1),  # 黑方获胜
        ]
        
        for pos, player in moves:
            input_vec = board.copy() + [player]
            output = [0] * 225
            
            history.append({
                "input": input_vec,
                "output": output,
                "position": pos
            })
            
            board[pos[0] * 15 + pos[1]] = player
        
        record = {
            "history": history,
            "winner": 1,
            "end_board": board + [0]
        }
        
        with open(record_file, 'w', encoding='utf-8') as f:
            json.dump(record, f)
        
        # 执行评价
        sample_file = self.evaluator.evaluate_record(record_file)
        
        # 验证文件生成
        self.assertTrue(os.path.exists(sample_file))
        
        # 验证内容
        with open(sample_file, 'r', encoding='utf-8') as f:
            sample = json.load(f)
        
        self.assertEqual(len(sample), 3)  # 3 步
        self.assertIn("input", sample[0])
        self.assertIn("output", sample[0])
        self.assertIn("mask", sample[0])
        self.assertEqual(len(sample[0]["mask"]), 225)
        self.assertEqual(len(sample[0]["output"]), 225)


class TestEvaluatorDecayRate(unittest.TestCase):
    """Evaluator decay_rate 参数测试"""
    
    def setUp(self):
        self.test_dir = "test_evaluator_rate_data"
        self.ai_data_dir = os.path.join(self.test_dir, "ai_data")
        self.train_data_dir = os.path.join(self.test_dir, "train_data")
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.ai_data_dir)
        os.makedirs(self.train_data_dir)
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def _make_history(self):
        """constructing simple game history with 5-in-a-row (black wins)"""
        board = [0] * 225
        history = []
        # 黑方形成水平五子连线：(7,3), (7,4), (7,5), (7,6), (7,7)
        moves = [
            ([7, 3], 1),   # 黑 a1
            ([0, 0], 2),   # 白 b1
            ([7, 4], 1),   # 黑 a2
            ([0, 1], 2),   # 白 b2
            ([7, 5], 1),   # 黑 a3
            ([0, 2], 2),   # 白 b3
            ([7, 6], 1),   # 黑 a4
            ([0, 3], 2),   # 白 b4
            ([7, 7], 1),   # 黑 a5（获胜）
        ]
        for pos, player in moves:
            # 先更新棋盘再记录
            board[pos[0] * 15 + pos[1]] = player
            history.append({
                "input": board.copy() + [player],
                "output": [0] * 225,
                "position": pos
            })
        return history
    
    def test_default_decay_rate_is_0_9(self):
        """默认 decay_rate 为 0.9"""
        evaluator = Evaluator(self.ai_data_dir, self.train_data_dir)
        self.assertAlmostEqual(evaluator.decay_rate, 0.9, places=9)
    
    def test_custom_decay_rate_is_used(self):
        """自定义 decay_rate 被使用"""
        evaluator = Evaluator(self.ai_data_dir, self.train_data_dir, decay_rate=0.5)
        self.assertAlmostEqual(evaluator.decay_rate, 0.5, places=9)
    
    def test_custom_decay_rate_affects_reward_calculation(self):
        """自定义 decay_rate 影响 reward 计算"""
        # rate=0.5: 默认 0.9 vs 0.5 的计算结果应不同
        evaluator_default = Evaluator(self.ai_data_dir, self.train_data_dir, decay_rate=0.9)
        evaluator_custom = Evaluator(self.ai_data_dir, self.train_data_dir, decay_rate=0.5)
        
        history = self._make_history()
        sample_default = evaluator_default._initialize_sample(history)
        evaluator_default._compute_rewards(sample_default, history, winner=1)
        
        sample_custom = evaluator_custom._initialize_sample(history)
        evaluator_custom._compute_rewards(sample_custom, history, winner=1)
        
        # 验证 decay_rate 影响 reward 计算
        # 五子连线最后一步（step 8, position [7,7]）是 MAX_REWARD
        # 五子连线第一步（step 0, position [7,3]）是 MAX_REWARD * decay^4
        # rate=0.9: 1000000 * 0.9^4 = 656100
        # rate=0.5: 1000000 * 0.5^4 = 62500
        first_step_idx_default = 7 * 15 + 3  # [7, 3]
        reward_09 = sample_default[0]["output"][first_step_idx_default]
        reward_05 = sample_custom[0]["output"][first_step_idx_default]
        
        self.assertEqual(reward_09, int(1000000 * 0.9 ** 4))
        self.assertEqual(reward_05, int(1000000 * 0.5 ** 4))
    
    def test_class_constant_decay_rate_unchanged(self):
        """DECAY_RATE 类常量保持 0.9，不受实例影响"""
        Evaluator(self.ai_data_dir, self.train_data_dir, decay_rate=0.5)
        # 类常量不应被修改
        self.assertEqual(Evaluator.DECAY_RATE, 0.9)


class TestEvaluatorMaskAlwaysOne(unittest.TestCase):
    """Evaluator mask 恒为 1 行为验证测试
    
    验证 mask 全面初始化为 1 的语义变更。
    """
    
    def setUp(self):
        self.test_dir = "test_evaluator_mask_data"
        self.ai_data_dir = os.path.join(self.test_dir, "ai_data")
        self.train_data_dir = os.path.join(self.test_dir, "train_data")
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.ai_data_dir)
        os.makedirs(self.train_data_dir)
        self.evaluator = Evaluator(self.ai_data_dir, self.train_data_dir)
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_mask_all_ones_empty_board(self):
        """空棋盘下 mask 全为 1"""
        # 空棋盘无棋子
        history = [{
            "input": [0] * 225 + [1],
            "output": [0] * 225,
            "position": [7, 7]
        }]
        sample_dict = self.evaluator._initialize_sample(history)
        
        # 所有位置 mask 应全为 1
        for i in range(225):
            self.assertEqual(sample_dict[0]["mask"][i], 1,
                             msg=f"位置 {i} 的 mask 应为 1，实际为 {sample_dict[0]['mask'][i]}")
    
    def test_mask_all_ones_occupied_board(self):
        """有棋子时 mask 仍全为 1"""
        # 在 (7, 7) 放置黑子
        board = [0] * 225
        board[7 * 15 + 7] = 1
        history = [{
            "input": board + [2],
            "output": [0] * 225,
            "position": [7, 8]
        }]
        sample_dict = self.evaluator._initialize_sample(history)
        
        # 所有位置 mask 应全为 1（包括已有棋子的位置）
        for i in range(225):
            self.assertEqual(sample_dict[0]["mask"][i], 1,
                             msg=f"位置 {i} 的 mask 应为 1，实际为 {sample_dict[0]['mask'][i]}")
    
    def test_mask_length_is_225(self):
        """mask 长度为 225"""
        history = [{
            "input": [0] * 225 + [1],
            "output": [0] * 225,
            "position": [7, 7]
        }]
        sample_dict = self.evaluator._initialize_sample(history)
        self.assertEqual(len(sample_dict[0]["mask"]), 225)
    
    def test_mask_preserved_after_compute_rewards(self):
        """compute_rewards 后 mask 仍全为 1"""
        # 构造 3 步帟史（黑方获胜）
        board = [0] * 225
        history = []
        moves = [([7, 7], 1), ([7, 8], 2), ([6, 6], 1)]
        for pos, player in moves:
            history.append({
                "input": board.copy() + [player],
                "output": [0] * 225,
                "position": pos
            })
            board[pos[0] * 15 + pos[1]] = player
        
        sample_dict = self.evaluator._initialize_sample(history)
        self.evaluator._compute_rewards(sample_dict, history, winner=1)
        
        # 应用 _compute_rewards 后，所有步 mask 应全为 1
        for step_idx, sample in enumerate(sample_dict):
            for i in range(225):
                self.assertEqual(sample["mask"][i], 1,
                                 msg=f"步骤 {step_idx} 位置 {i} 的 mask 应为 1")
    
    def test_sample_json_mask_all_ones(self):
        """sample.json 中写入的 mask 全为 1"""
        # 创建测试 record.json
        round_dir = os.path.join(self.ai_data_dir, "test_round")
        game_dir = os.path.join(round_dir, "1")
        os.makedirs(game_dir)
        record_file = os.path.join(game_dir, "record.json")
        
        board = [0] * 225
        history = []
        moves = [([7, 7], 1), ([7, 8], 2), ([6, 6], 1)]
        for pos, player in moves:
            history.append({
                "input": board.copy() + [player],
                "output": [0] * 225,
                "position": pos
            })
            board[pos[0] * 15 + pos[1]] = player
        
        record = {"history": history, "winner": 1, "end_board": board + [0]}
        with open(record_file, 'w', encoding='utf-8') as f:
            json.dump(record, f)
        
        sample_file = self.evaluator.evaluate_record(record_file)
        with open(sample_file, 'r', encoding='utf-8') as f:
            sample = json.load(f)
        
        # 所有步骤的 mask 应全为 1
        for step_idx, step_sample in enumerate(sample):
            for i in range(225):
                self.assertEqual(step_sample["mask"][i], 1,
                                 msg=f"步骤 {step_idx} 位置 {i} 的 mask 应为 1")


if __name__ == "__main__":
    unittest.main()
