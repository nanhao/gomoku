"""MoveDecision 单元测试

测试 AIEngine.get_move() 返回值的封装类。
遵循 TDD：先写测试，再实现。
"""

import sys
import os
import unittest

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestMoveDecision(unittest.TestCase):
    """MoveDecision 类测试"""

    def test_construction(self):
        """构造时传入 x, y，get_position 返回对应的 tuple"""
        from ai.engine import MoveDecision
        d = MoveDecision(7, 8)
        self.assertEqual(d.get_position(), (7, 8))

    def test_get_position_consistent(self):
        """多次调用 get_position 返回一致结果"""
        from ai.engine import MoveDecision
        d = MoveDecision(3, 14)
        self.assertEqual(d.get_position(), (3, 14))
        self.assertEqual(d.get_position(), (3, 14))

    def test_two_instances_independent(self):
        """两个实例的 position 互不影响"""
        from ai.engine import MoveDecision
        d1 = MoveDecision(1, 2)
        d2 = MoveDecision(3, 4)
        self.assertEqual(d1.get_position(), (1, 2))
        self.assertEqual(d2.get_position(), (3, 4))


if __name__ == "__main__":
    unittest.main()
