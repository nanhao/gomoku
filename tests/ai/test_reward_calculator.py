"""RewardCalculator 奖励计算测试

测试各棋型的 reward 计算、冲四递归路径、双向检测。
"""

import sys
import os
import unittest

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.reward_calculator import RewardCalculator, RewardConfig


class TestRewardConfig(unittest.TestCase):
    """测试 RewardConfig 默认值"""

    def test_default_max_reward(self):
        """默认 max_reward = 1_000_000"""
        config = RewardConfig()
        self.assertEqual(config.max_reward, 1_000_000)

    def test_default_min_reward(self):
        """默认 min_reward = -1_000_000"""
        config = RewardConfig()
        self.assertEqual(config.min_reward, -1_000_000)

    def test_default_values_contain_key_shapes(self):
        """默认 values 应包含关键棋型"""
        config = RewardConfig()
        self.assertIn("连五", config.values)
        self.assertIn("活四", config.values)
        self.assertIn("冲四", config.values)
        self.assertIn("活三", config.values)
        self.assertIn("活二", config.values)

    def test_custom_config(self):
        """自定义配置应生效"""
        config = RewardConfig(max_reward=500000, min_reward=-500000)
        self.assertEqual(config.max_reward, 500000)
        self.assertEqual(config.min_reward, -500000)


class TestShapeToReward(unittest.TestCase):
    """测试 shape_to_reward 各棋型映射"""

    def setUp(self):
        self.calculator = RewardCalculator()
        self.max_reward = self.calculator.config.max_reward

    def test_live_five_reward(self):
        """连五 reward = MAX_REWARD"""
        shape_info = {"shape": "连五", "directions": ["横向"]}
        reward = self.calculator.shape_to_reward([0] * 225, shape_info, 1, True, (7, 7))
        self.assertEqual(reward, self.max_reward)

    def test_live_four_reward(self):
        """活四 reward = 0.9 * MAX_REWARD"""
        shape_info = {"shape": "活四", "directions": ["横向"]}
        reward = self.calculator.shape_to_reward([0] * 225, shape_info, 1, True, (7, 7))
        self.assertEqual(reward, int(0.9 * self.max_reward))

    def test_multi_four_reward(self):
        """多冲四 reward = 0.9 * MAX_REWARD"""
        shape_info = {"shape": "多冲四", "directions": ["横向", "纵向"]}
        reward = self.calculator.shape_to_reward([0] * 225, shape_info, 1, True, (7, 7))
        self.assertEqual(reward, int(0.9 * self.max_reward))

    def test_live_three_reward(self):
        """活三 reward = 0.5 * MAX_REWARD"""
        shape_info = {"shape": "活三", "directions": ["横向"]}
        reward = self.calculator.shape_to_reward([0] * 225, shape_info, 1, True, (7, 7))
        self.assertEqual(reward, int(0.5 * self.max_reward))

    def test_multi_live_three_reward(self):
        """多活三 reward = 0.8 * MAX_REWARD"""
        shape_info = {"shape": "多活三", "directions": ["横向", "纵向"]}
        reward = self.calculator.shape_to_reward([0] * 225, shape_info, 1, True, (7, 7))
        self.assertEqual(reward, int(0.8 * self.max_reward))

    def test_opponent_factor(self):
        """敌方棋型 reward 乘以 0.95 因子"""
        shape_info = {"shape": "活四", "directions": ["横向"]}
        reward_ally = self.calculator.shape_to_reward([0] * 225, shape_info, 1, True, (7, 7))
        reward_enemy = self.calculator.shape_to_reward([0] * 225, shape_info, 1, False, (7, 7))
        self.assertEqual(reward_enemy, int(0.9 * self.max_reward * 0.95))
        self.assertLess(reward_enemy, reward_ally)

    def test_sleep_three_dynamic_ratio(self):
        """眠三 reward = (0.1 + count_directions * 0.1) * MAX_REWARD"""
        # 1 个方向
        shape1 = {"shape": "眠三", "directions": ["横向"]}
        reward1 = self.calculator.shape_to_reward([0] * 225, shape1, 1, True, (7, 7))
        self.assertEqual(reward1, int((0.1 + 1 * 0.1) * self.max_reward))

        # 2 个方向
        shape2 = {"shape": "眠三", "directions": ["横向", "纵向"]}
        reward2 = self.calculator.shape_to_reward([0] * 225, shape2, 1, True, (7, 7))
        self.assertEqual(reward2, int((0.1 + 2 * 0.1) * self.max_reward))

    def test_live_two_dynamic_ratio(self):
        """活二 reward = (0.3 + count_directions * 0.1) * MAX_REWARD"""
        shape1 = {"shape": "活二", "directions": ["横向"]}
        reward1 = self.calculator.shape_to_reward([0] * 225, shape1, 1, True, (7, 7))
        self.assertEqual(reward1, int((0.3 + 1 * 0.1) * self.max_reward))

    def test_sleep_two_zero_reward(self):
        """眠二 reward = 0"""
        shape_info = {"shape": "眠二", "directions": ["横向"]}
        reward = self.calculator.shape_to_reward([0] * 225, shape_info, 1, True, (7, 7))
        self.assertEqual(reward, 0)

    def test_single_piece_zero_reward(self):
        """独一 reward = 0"""
        shape_info = {"shape": "独一", "directions": ["横向"]}
        reward = self.calculator.shape_to_reward([0] * 225, shape_info, 1, True, (7, 7))
        self.assertEqual(reward, 0)


class TestFourReward(unittest.TestCase):
    """测试冲四递归 reward 计算"""

    def setUp(self):
        self.calculator = RewardCalculator()
        self.max_reward = self.calculator.config.max_reward

    def test_four_without_position_fallback(self):
        """冲四不传 position 时回退到简化计算"""
        shape_info = {"shape": "冲四", "directions": ["横向"]}
        reward = self.calculator.shape_to_reward([0] * 225, shape_info, 1, True, position=None)
        # 回退值 = 0.5 * MAX_REWARD
        self.assertEqual(reward, int(0.5 * self.max_reward))

    def test_four_with_no_subsequent_shape(self):
        """冲四防守后无后续棋型时，reward 较低"""
        # 构造一个简单冲四棋盘：黑方 (7,5)(7,6)(7,7)(7,8)，白方阻挡一端 (7,4)
        board = [0] * 225
        board[7 * 15 + 5] = 1  # 黑
        board[7 * 15 + 6] = 1  # 黑
        board[7 * 15 + 7] = 1  # 黑
        board[7 * 15 + 8] = 1  # 黑
        board[7 * 15 + 4] = 2  # 白方阻挡

        shape_info = {"shape": "冲四", "directions": ["横向"]}
        reward = self.calculator.shape_to_reward(board, shape_info, 1, True, position=(7, 9))

        # 防守后无后续棋型，应返回低值
        self.assertGreater(reward, 0)
        self.assertLessEqual(reward, self.max_reward)


class TestComputePositionReward(unittest.TestCase):
    """测试 compute_position_reward 双向检测"""

    def setUp(self):
        self.calculator = RewardCalculator()

    def test_empty_board_center_reward(self):
        """空棋盘中心位置 reward = 0（只有独一棋型）"""
        board = [0] * 225
        reward = self.calculator.compute_position_reward(board, 7, 7, 1)
        # 空棋盘上只有独一棋型，reward = 0
        self.assertEqual(reward, 0)

    def test_with_existing_pieces_has_reward(self):
        """有棋子的棋盘上，某个位置应产生非零 reward"""
        board = [0] * 225
        # 放置一些黑子，形成活二/活三
        board[7 * 15 + 5] = 1
        board[7 * 15 + 6] = 1
        board[7 * 15 + 7] = 1
        # 在 (7,8) 落子可形成活四
        reward = self.calculator.compute_position_reward(board, 7, 8, 1)
        self.assertGreater(reward, 0)

    def test_bidirectional_detection(self):
        """双向检测：敌方活三位置 reward 应考虑防守价值"""
        # 构造白方活三
        board = [0] * 225
        board[7 * 15 + 5] = 2  # 白
        board[7 * 15 + 6] = 2  # 白
        board[7 * 15 + 7] = 2  # 白

        # 黑方在此位置落子无好棋型，但防守白方活三
        reward = self.calculator.compute_position_reward(board, 7, 8, 1)
        # 应大于 0（因为防守了白方活三）
        self.assertGreater(reward, 0)

    def test_max_of_both_players(self):
        """双向检测应取两方 reward 的较大值"""
        # 构造白方活三的棋盘
        board = [0] * 225
        board[7 * 15 + 5] = 2  # 白
        board[7 * 15 + 6] = 2  # 白
        board[7 * 15 + 7] = 2  # 白

        # 在 (7, 8) 位置：黑方落子有独一（reward=0），但防守白方活三有价值
        reward = self.calculator.compute_position_reward(board, 7, 8, 1)
        self.assertGreater(reward, 0)


class TestGetShapeRatio(unittest.TestCase):
    """测试 _get_shape_ratio 棋型比例系数"""

    def setUp(self):
        self.calculator = RewardCalculator()

    def test_live_five_ratio(self):
        """连五比例 = 1.0"""
        ratio = self.calculator._get_shape_ratio("连五", 1)
        self.assertEqual(ratio, 1.0)

    def test_live_four_ratio(self):
        """活四比例 = 0.9"""
        ratio = self.calculator._get_shape_ratio("活四", 1)
        self.assertEqual(ratio, 0.9)

    def test_four_ratio(self):
        """冲四比例 = 0.5"""
        ratio = self.calculator._get_shape_ratio("冲四", 1)
        self.assertEqual(ratio, 0.5)

    def test_live_three_ratio(self):
        """活三比例 = 0.5"""
        ratio = self.calculator._get_shape_ratio("活三", 1)
        self.assertEqual(ratio, 0.5)

    def test_sleep_three_dynamic(self):
        """眠三动态比例 = 0.1 + count_directions * 0.1"""
        ratio1 = self.calculator._get_shape_ratio("眠三", 1)
        self.assertAlmostEqual(ratio1, 0.2)  # 0.1 + 1*0.1

        ratio2 = self.calculator._get_shape_ratio("眠三", 2)
        self.assertAlmostEqual(ratio2, 0.3)  # 0.1 + 2*0.1

    def test_live_two_dynamic(self):
        """活二动态比例 = 0.3 + count_directions * 0.1"""
        ratio1 = self.calculator._get_shape_ratio("活二", 1)
        self.assertAlmostEqual(ratio1, 0.4)  # 0.3 + 1*0.1

        ratio2 = self.calculator._get_shape_ratio("活二", 2)
        self.assertAlmostEqual(ratio2, 0.5)  # 0.3 + 2*0.1

    def test_unknown_shape_zero(self):
        """未知棋型比例 = 0"""
        ratio = self.calculator._get_shape_ratio("未知棋型", 1)
        self.assertEqual(ratio, 0.0)


class TestShapeRewardValue(unittest.TestCase):
    """测试 _shape_reward_value 非递归奖励值"""

    def setUp(self):
        self.calculator = RewardCalculator()

    def test_live_five_value(self):
        """连五非递归值 = 1.0 * MAX_REWARD * factor"""
        value = self.calculator._shape_reward_value("连五", 1, 1.0)
        self.assertEqual(value, self.calculator.config.max_reward)

    def test_factor_applied(self):
        """factor 应乘以最终值"""
        value = self.calculator._shape_reward_value("活四", 1, 0.95)
        self.assertEqual(value, int(0.9 * self.calculator.config.max_reward * 0.95))


if __name__ == '__main__':
    unittest.main()
