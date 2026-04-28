"""Evaluator 阶段1+2 瘦身重构单元测试

验证阶段1+2的变更：
1. 6 个私有委托方法已删除
2. 5 个公开委托方法已彻底删除
3. 3 个重复常量已删除
4. 内部调用已直连底层组件，evaluate_record 能正常完成
"""

import json
import os
import shutil
import sys
import unittest

# 添加项目根目录到路径
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.evaluator import Evaluator


class TestPrivateDelegateMethodsRemoved(unittest.TestCase):
    """验证 6 个私有委托方法已从 Evaluator 中删除"""

    def setUp(self):
        """创建 Evaluator 实例"""
        self.evaluator = Evaluator()

    # ---------- 6 个私有委托方法应不再存在 ----------

    def test_compute_position_reward_removed(self):
        """_compute_position_reward 不再是 Evaluator 的方法"""
        self.assertFalse(
            hasattr(self.evaluator, '_compute_position_reward'),
            "_compute_position_reward 应已被删除"
        )

    def test_detect_shape_in_direction_removed(self):
        """_detect_shape_in_direction 不再是 Evaluator 的方法"""
        self.assertFalse(
            hasattr(self.evaluator, '_detect_shape_in_direction'),
            "_detect_shape_in_direction 应已被删除"
        )

    def test_compute_four_reward_removed(self):
        """_compute_four_reward 不再是 Evaluator 的方法"""
        self.assertFalse(
            hasattr(self.evaluator, '_compute_four_reward'),
            "_compute_four_reward 应已被删除"
        )

    def test_find_four_defense_removed(self):
        """_find_four_defense 不再是 Evaluator 的方法"""
        self.assertFalse(
            hasattr(self.evaluator, '_find_four_defense'),
            "_find_four_defense 应已被删除"
        )

    def test_shape_reward_value_removed(self):
        """_shape_reward_value 不再是 Evaluator 的方法"""
        self.assertFalse(
            hasattr(self.evaluator, '_shape_reward_value'),
            "_shape_reward_value 应已被删除"
        )

    def test_softmax_removed(self):
        """_softmax 不再是 Evaluator 的方法"""
        self.assertFalse(
            hasattr(self.evaluator, '_softmax'),
            "_softmax 应已被删除"
        )


class TestPublicDelegateMethodsRemoved(unittest.TestCase):
    """验证 5 个公开委托方法已从 Evaluator 中彻底删除

    阶段1：这些方法曾发出 DeprecationWarning。
    阶段2：调用方已全部迁移，方法彻底删除。
    """

    def setUp(self):
        """创建 Evaluator 实例"""
        self.evaluator = Evaluator()

    def test_detect_shapes_removed(self):
        """detect_shapes 不再是 Evaluator 的方法"""
        self.assertFalse(
            hasattr(self.evaluator, 'detect_shapes'),
            "detect_shapes 应已被删除，请直接使用 ai.shape_detector.detect_shapes"
        )

    def test_shape_to_reward_removed(self):
        """shape_to_reward 不再是 Evaluator 的方法"""
        self.assertFalse(
            hasattr(self.evaluator, 'shape_to_reward'),
            "shape_to_reward 应已被删除，请直接使用 RewardCalculator.shape_to_reward"
        )

    def test_find_five_in_a_row_removed(self):
        """find_five_in_a_row 不再是 Evaluator 的方法"""
        self.assertFalse(
            hasattr(self.evaluator, 'find_five_in_a_row'),
            "find_five_in_a_row 应已被删除，请直接使用 LineTracer.find_five_in_a_row"
        )

    def test_trace_winning_line_removed(self):
        """trace_winning_line 不再是 Evaluator 的方法"""
        self.assertFalse(
            hasattr(self.evaluator, 'trace_winning_line'),
            "trace_winning_line 应已被删除，请直接使用 LineTracer.trace_winning_line"
        )

    def test_blend_rewards_removed(self):
        """blend_rewards 不再是 Evaluator 的方法"""
        self.assertFalse(
            hasattr(self.evaluator, 'blend_rewards'),
            "blend_rewards 应已被删除，请直接使用 LineTracer.blend_rewards"
        )


class TestDuplicateConstantsRemoved(unittest.TestCase):
    """验证 3 个重复常量已从 Evaluator 类中删除"""

    def test_directions_constant_removed(self):
        """Evaluator.DIRECTIONS 应已被删除"""
        with self.assertRaises(AttributeError):
            _ = Evaluator.DIRECTIONS

    def test_direction_names_constant_removed(self):
        """Evaluator.DIRECTION_NAMES 应已被删除"""
        with self.assertRaises(AttributeError):
            _ = Evaluator.DIRECTION_NAMES

    def test_softmax_scale_constant_removed(self):
        """Evaluator.SOFTMAX_SCALE 应已被删除"""
        with self.assertRaises(AttributeError):
            _ = Evaluator.SOFTMAX_SCALE


class TestInternalCallsDirectlyConnected(unittest.TestCase):
    """验证内部调用已直连底层组件，evaluate_record 能正常完成"""

    def setUp(self):
        """创建临时目录和 Evaluator 实例"""
        self.test_dir = "test_evaluator_phase1_integration_data"
        self.ai_data_dir = os.path.join(self.test_dir, "ai_data")
        self.train_data_dir = os.path.join(self.test_dir, "train_data")

        # 清理可能残留的旧测试目录
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.ai_data_dir)
        os.makedirs(self.train_data_dir)

        self.evaluator = Evaluator(self.ai_data_dir, self.train_data_dir)

    def tearDown(self):
        """清理临时目录"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_evaluate_record_completes_successfully(self):
        """evaluate_record 正常完成——内部调用直连 _reward_calculator 和 _line_tracer

        构造一个有胜者的对局记录，验证：
        - _compute_all_rewards 内部直连调用不再走已删除的委托方法
        - softmax 走 probability_utils.masked_softmax
        - blend_rewards 走 _line_tracer.blend_rewards
        """
        # 构造对局历史：黑方横向五连
        history = []
        board = [0] * 225

        # 黑方 5 步形成五连
        for i in range(5):
            step = {
                "input": board[:] + [1],  # 黑方回合
                "output": [],
                "position": [7, 7 + i]
            }
            history.append(step)
            board[7 * 15 + (7 + i)] = 1  # 黑方落子

            # 白方在其他位置落子
            if i < 4:
                step_white = {
                    "input": board[:] + [2],  # 白方回合
                    "output": [],
                    "position": [0, i]
                }
                history.append(step_white)
                board[0 * 15 + i] = 2  # 白方落子

        record = {
            "history": history,
            "winner": 1,
            "end_board": board[:]
        }

        # 保存 record.json
        record_path = os.path.join(self.ai_data_dir, "test_game", "record.json")
        os.makedirs(os.path.dirname(record_path), exist_ok=True)
        with open(record_path, 'w', encoding='utf-8') as f:
            json.dump(record, f)

        # 执行评价——不应抛出异常
        sample_path = self.evaluator.evaluate_record(record_path)

        # 验证 sample.json 存在
        self.assertTrue(os.path.exists(sample_path))

        # 读取并验证结果结构
        with open(sample_path, 'r', encoding='utf-8') as f:
            sample = json.load(f)

        self.assertEqual(len(sample), len(history))

        # 验证每个 step 的结构和输出合法性
        for step in sample:
            self.assertIn("input", step)
            self.assertIn("output", step)
            self.assertIn("mask", step)
            self.assertEqual(len(step["output"]), 225)
            self.assertEqual(len(step["mask"]), 225)
            # output 应为概率分布：mask=1 的位置概率和约为 1
            masked_sum = sum(
                step["output"][i]
                for i in range(225)
                if step["mask"][i] == 1
            )
            self.assertAlmostEqual(masked_sum, 1.0, places=5,
                                   msg="Softmax 输出概率和应为 1")

    def test_evaluate_record_no_winner(self):
        """无胜者时 evaluate_record 正常完成（不走五连追溯分支）"""
        history = [
            {
                "input": [0] * 225 + [1],
                "output": [],
                "position": [7, 7]
            }
        ]
        record = {
            "history": history,
            "winner": 0,
            "end_board": []
        }

        # 保存 record.json
        record_path = os.path.join(self.ai_data_dir, "game_no_win", "record.json")
        os.makedirs(os.path.dirname(record_path), exist_ok=True)
        with open(record_path, 'w', encoding='utf-8') as f:
            json.dump(record, f)

        # 此记录无效（winner=0），evaluate_record 应抛出 ValueError
        with self.assertRaises(ValueError):
            self.evaluator.evaluate_record(record_path)


if __name__ == '__main__':
    unittest.main()
