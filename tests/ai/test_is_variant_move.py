"""is_variant_move 函数单元测试

覆盖意图 9 新增的变异步判定函数（create_samples.is_variant_move）：
- 基本判定逻辑（IV-01 ~ IV-06）
- 边界条件（EDGE-01 ~ EDGE-04）
"""

import math
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.create_samples import is_variant_move


class TestIsVariantMoveBasic(unittest.TestCase):
    """is_variant_move 基本判定逻辑（IV-01 ~ IV-06）"""

    def test_unique_max_not_variant(self):
        """IV-01: 落子位是唯一最大值 → False（非变异步）"""
        output = [0.0] * 225
        output[10] = 0.9  # max=0.9，唯一
        step = {"output": output, "position": [0, 10]}
        self.assertFalse(is_variant_move(step))

    def test_tied_max_not_variant(self):
        """IV-02: 落子位是并列最大值之一 → False（非变异步）"""
        output = [0.0] * 225
        output[10] = 0.5
        output[20] = 0.5  # 并列 max
        step = {"output": output, "position": [0, 10]}  # idx=10，∈ argmax
        self.assertFalse(is_variant_move(step))

    def test_not_max_variant(self):
        """IV-03: 落子位不是最大值 → True（变异步）"""
        output = [0.0] * 225
        output[10] = 0.1  # 落子位=0.1
        output[0] = 0.8   # max=0.8
        step = {"output": output, "position": [0, 10]}
        self.assertTrue(is_variant_move(step))

    def test_all_zeros_not_variant(self):
        """IV-04: 全零分布 → False（落子位 ∈ argmax）"""
        output = [0.0] * 225
        step = {"output": output, "position": [7, 7]}
        self.assertFalse(is_variant_move(step))

    def test_position_oob_row_20(self):
        """IV-05: 越界坐标 row=20 → False（兜底）"""
        output = [0.0] * 225
        output[0] = 0.9
        step = {"output": output, "position": [20, 20]}
        self.assertFalse(is_variant_move(step))

    def test_position_oob_negative(self):
        """IV-06: 越界坐标 row=-1 → False（兜底）"""
        output = [0.0] * 225
        step = {"output": output, "position": [-1, 5]}
        self.assertFalse(is_variant_move(step))


class TestIsVariantMoveEdgeCases(unittest.TestCase):
    """is_variant_move 边界条件（EDGE-01 ~ EDGE-04）"""

    def test_missing_output_keyerror(self):
        """EDGE-01: step 缺少 output 字段 → KeyError"""
        step = {"position": [7, 7]}
        with self.assertRaises(KeyError):
            is_variant_move(step)

    def test_missing_position_keyerror(self):
        """EDGE-02: step 缺少 position 字段 → KeyError"""
        step = {"output": [0.0] * 225}
        with self.assertRaises(KeyError):
            is_variant_move(step)

    def test_output_wrong_length(self):
        """EDGE-03: output 长度 ≠ 225 → False（兜底跳过）"""
        step = {"output": [0.0] * 100, "position": [7, 7]}
        self.assertFalse(is_variant_move(step))

    def test_output_contains_nan(self):
        """EDGE-04: output 含 NaN → max() 返回 NaN，output[idx]!=NaN → True

        注：NaN != NaN 恒成立，故 is_variant_move 返回 True。
        这是 Python max() 在 NaN 存在时的已知行为。
        """
        output = [0.0] * 225
        output[0] = float("nan")
        step = {"output": output, "position": [7, 7]}
        # NaN 下 max() 行为：如果有 NaN，max() 返回 NaN
        # NaN != NaN → True（变异步）
        result = is_variant_move(step)
        # 只需验证不崩溃，行为不严格要求
        self.assertIsInstance(result, bool)


if __name__ == '__main__':
    unittest.main()
