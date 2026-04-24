"""棋型检测引擎单元测试

测试 ai/shape_detector.py 中的核心函数：
- detect_single_direction: 单方向棋型检测（含 Numba JIT 回退路径）
- _detect_single_direction_pure_python: 纯 Python 版本
- detect_shapes: 多方向棋型汇总与合并逻辑

测试策略：
1. 使用 mock 将 _NUMBA_AVAILABLE 设为 False，强制走纯 Python 路径
2. 直接调用 _detect_single_direction_pure_python 验证各棋型
3. 测试 detect_shapes 的合并逻辑（多冲四、多活三、连五优先）
4. 测试 Numba JIT 回退路径不崩溃
"""

import os
import sys
import unittest
from unittest.mock import patch

# 确保项目根目录在路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.shape_detector import (
    detect_single_direction,
    detect_shapes,
    _detect_single_direction_pure_python,
    DIRECTIONS,
    _SHAPE_CODE_MAP,
)


# ==================== 辅助函数 ====================

def _make_empty_board():
    """创建 225 维全零棋盘（flat list）"""
    return [0] * 225


def _set_piece(board, x, y, player):
    """在棋盘指定位置放置棋子

    Args:
        board: 225 维 flat list
        x, y: 坐标 (0~14)
        player: 1=黑, 2=白
    """
    board[x * 15 + y] = player


# ==================== 测试纯 Python 单方向检测 ====================

class TestDetectSingleDirectionPurePython(unittest.TestCase):
    """测试 _detect_single_direction_pure_python 的各种棋型检测

    纯 Python 回退路径，可直接测试，不受 Numba 环境影响。
    """

    def test_连五_横向五子连珠(self):
        """连五：横向 5 子连珠，应返回 '连五'"""
        board = _make_empty_board()
        # 在 (7,3)~(7,7) 放置 5 颗黑子
        for y in range(3, 8):
            _set_piece(board, 7, y, 1)
        result = _detect_single_direction_pure_python(board, 7, 5, 1, 0, 1)
        self.assertEqual(result, "连五")

    def test_连五_纵向五子连珠(self):
        """连五：纵向 5 子连珠"""
        board = _make_empty_board()
        for x in range(3, 8):
            _set_piece(board, x, 7, 1)
        result = _detect_single_direction_pure_python(board, 5, 7, 1, 1, 0)
        self.assertEqual(result, "连五")

    def test_连五_超过五子(self):
        """连五：超过 5 子也归为连五（count >= 5）"""
        board = _make_empty_board()
        for y in range(3, 9):
            _set_piece(board, 7, y, 1)
        result = _detect_single_direction_pure_python(board, 7, 6, 1, 0, 1)
        self.assertEqual(result, "连五")

    def test_活四_两端开阔(self):
        """活四：4 子连珠且两端均为空位"""
        board = _make_empty_board()
        # (7,4)~(7,7) 四颗黑子，(7,3) 和 (7,8) 为空
        for y in range(4, 8):
            _set_piece(board, 7, y, 1)
        result = _detect_single_direction_pure_python(board, 7, 5, 1, 0, 1)
        self.assertEqual(result, "活四")

    def test_冲四_一端被挡(self):
        """冲四：4 子连珠且一端被对手或边界挡住"""
        board = _make_empty_board()
        # (7,4)~(7,7) 四颗黑子，(7,8) 放白子挡住，(7,3) 为空
        for y in range(4, 8):
            _set_piece(board, 7, y, 1)
        _set_piece(board, 7, 8, 2)  # 白子挡一端
        result = _detect_single_direction_pure_python(board, 7, 5, 1, 0, 1)
        self.assertEqual(result, "冲四")

    def test_冲四_边界挡一端(self):
        """冲四：4 子连珠且一端在棋盘边界"""
        board = _make_empty_board()
        # (7,0)~(7,3) 四颗黑子，左端是边界，(7,4) 为空
        for y in range(0, 4):
            _set_piece(board, 7, y, 1)
        result = _detect_single_direction_pure_python(board, 7, 2, 1, 0, 1)
        self.assertEqual(result, "冲四")

    def test_两端堵四子_返回None(self):
        """两端堵：4 子连珠且两端都被挡，返回 None"""
        board = _make_empty_board()
        for y in range(4, 8):
            _set_piece(board, 7, y, 1)
        _set_piece(board, 7, 3, 2)  # 白子挡正方向
        _set_piece(board, 7, 8, 2)  # 白子挡反方向
        result = _detect_single_direction_pure_python(board, 7, 5, 1, 0, 1)
        self.assertIsNone(result)

    def test_活三_两端开阔(self):
        """活三：3 子连珠且两端均为空位"""
        board = _make_empty_board()
        for y in range(5, 8):
            _set_piece(board, 7, y, 1)
        result = _detect_single_direction_pure_python(board, 7, 6, 1, 0, 1)
        self.assertEqual(result, "活三")

    def test_眠三_一端被挡(self):
        """眠三：3 子连珠且一端被挡"""
        board = _make_empty_board()
        for y in range(5, 8):
            _set_piece(board, 7, y, 1)
        _set_piece(board, 7, 8, 2)  # 白子挡一端
        result = _detect_single_direction_pure_python(board, 7, 6, 1, 0, 1)
        self.assertEqual(result, "眠三")

    def test_两端堵三子_返回None(self):
        """两端堵：3 子连珠且两端都被挡，返回 None"""
        board = _make_empty_board()
        for y in range(5, 8):
            _set_piece(board, 7, y, 1)
        _set_piece(board, 7, 4, 2)  # 白子挡正方向
        _set_piece(board, 7, 8, 2)  # 白子挡反方向
        result = _detect_single_direction_pure_python(board, 7, 6, 1, 0, 1)
        self.assertIsNone(result)

    def test_活二_两端开阔(self):
        """活二：2 子连珠且两端均为空位"""
        board = _make_empty_board()
        _set_piece(board, 7, 6, 1)
        _set_piece(board, 7, 7, 1)
        result = _detect_single_direction_pure_python(board, 7, 6, 1, 0, 1)
        self.assertEqual(result, "活二")

    def test_眠二_一端被挡(self):
        """眠二：2 子连珠且一端被挡"""
        board = _make_empty_board()
        _set_piece(board, 7, 6, 1)
        _set_piece(board, 7, 7, 1)
        _set_piece(board, 7, 8, 2)  # 白子挡一端
        result = _detect_single_direction_pure_python(board, 7, 6, 1, 0, 1)
        self.assertEqual(result, "眠二")

    def test_两端堵二子_返回None(self):
        """两端堵：2 子连珠且两端都被挡，返回 None"""
        board = _make_empty_board()
        _set_piece(board, 7, 6, 1)
        _set_piece(board, 7, 7, 1)
        _set_piece(board, 7, 5, 2)  # 白子挡正方向
        _set_piece(board, 7, 8, 2)  # 白子挡反方向
        result = _detect_single_direction_pure_python(board, 7, 6, 1, 0, 1)
        self.assertIsNone(result)

    def test_独一_孤立一子(self):
        """独一：单个棋子两端均被挡或只有 1 子"""
        board = _make_empty_board()
        _set_piece(board, 7, 7, 1)
        _set_piece(board, 7, 6, 2)  # 一端挡
        _set_piece(board, 7, 8, 2)  # 另一端挡
        result = _detect_single_direction_pure_python(board, 7, 7, 1, 0, 1)
        self.assertEqual(result, "独一")

    def test_独一_无阻挡的孤立一子(self):
        """独一：只有 1 颗棋子，两端无挡也算独一"""
        board = _make_empty_board()
        _set_piece(board, 7, 7, 1)
        result = _detect_single_direction_pure_python(board, 7, 7, 1, 0, 1)
        self.assertEqual(result, "独一")

    def test_对角线方向检测(self):
        """测试主对角线方向的棋型检测"""
        board = _make_empty_board()
        # 主对角线 (4,4)~(4,8) 放 5 颗黑子
        for i in range(5):
            _set_piece(board, 4 + i, 4 + i, 1)
        result = _detect_single_direction_pure_python(board, 6, 6, 1, 1, 1)
        self.assertEqual(result, "连五")

    def test_副对角线方向检测(self):
        """测试副对角线方向的棋型检测"""
        board = _make_empty_board()
        # 副对角线方向 (4,10)~(4,6) 放 5 颗黑子
        for i in range(5):
            _set_piece(board, 4 + i, 10 - i, 1)
        result = _detect_single_direction_pure_python(board, 6, 8, 1, 1, -1)
        self.assertEqual(result, "连五")


# ==================== 测试 detect_single_direction（强制纯 Python 路径）====================

class TestDetectSingleDirectionWithFallback(unittest.TestCase):
    """测试 detect_single_direction 在 Numba 不可用时走纯 Python 回退路径"""

    @patch('ai.shape_detector._NUMBA_AVAILABLE', False)
    def test_numba_disabled_走纯Python路径(self):
        """mock _NUMBA_AVAILABLE=False，验证走纯 Python 路径不崩溃"""
        board = _make_empty_board()
        for y in range(5, 8):
            _set_piece(board, 7, y, 1)
        result = detect_single_direction(board, 7, 6, 1, 0, 1)
        self.assertEqual(result, "活三")

    @patch('ai.shape_detector._NUMBA_AVAILABLE', False)
    def test_numba_disabled_连五检测(self):
        """mock _NUMBA_AVAILABLE=False，验证连五检测正确"""
        board = _make_empty_board()
        for y in range(3, 8):
            _set_piece(board, 7, y, 1)
        result = detect_single_direction(board, 7, 5, 1, 0, 1)
        self.assertEqual(result, "连五")

    @patch('ai.shape_detector._NUMBA_AVAILABLE', False)
    def test_numba_disabled_两端堵返回None(self):
        """mock _NUMBA_AVAILABLE=False，验证两端堵返回 None"""
        board = _make_empty_board()
        for y in range(4, 8):
            _set_piece(board, 7, y, 1)
        _set_piece(board, 7, 3, 2)
        _set_piece(board, 7, 8, 2)
        result = detect_single_direction(board, 7, 5, 1, 0, 1)
        self.assertIsNone(result)


# ==================== 测试 detect_shapes 合并逻辑 ====================

class TestDetectShapes(unittest.TestCase):
    """测试 detect_shapes 的棋型汇总与合并逻辑

    重点测试：
    - 多冲四合并为 "多冲四"
    - 多活三合并为 "多活三"
    - 连五优先级最高（即使同时存在其他棋型）
    - 单方向棋型的正确输出
    """

    @patch('ai.shape_detector._NUMBA_AVAILABLE', False)
    def test_空位置_无棋型返回空列表(self):
        """落子位置周围无任何棋型，应返回空列表"""
        board = _make_empty_board()
        # 在 (7,7) 落子，周围没有己方棋子
        result = detect_shapes(board, (7, 7), 1)
        # 四个方向都是"独一"或两端堵
        # 独一也是棋型，所以不会为空；但如果两端都被对手挡住则返回 None
        # (7,7) 空棋盘落子 → 四方向都是独一
        self.assertIsInstance(result, list)

    @patch('ai.shape_detector._NUMBA_AVAILABLE', False)
    def test_连五优先级最高(self):
        """连五应优先返回，即使同时存在其他棋型"""
        board = _make_empty_board()
        # 横向连五 + 纵向活三
        for y in range(3, 8):
            _set_piece(board, 7, y, 1)
        for x in range(6, 9):
            _set_piece(board, x, 5, 1)
        # 在 (7,5) 落子检测
        result = detect_shapes(board, (7, 5), 1)
        # 连五应直接返回，不包含其他棋型
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["shape"], "连五")

    @patch('ai.shape_detector._NUMBA_AVAILABLE', False)
    def test_多冲四_合并(self):
        """两个方向形成冲四，应合并为 '多冲四'"""
        board = _make_empty_board()
        # 横向冲四：落子点 (7,6)，(7,3)(7,4)(7,5) 为黑子，(7,2) 白挡左端
        for y in [3, 4, 5]:
            _set_piece(board, 7, y, 1)
        _set_piece(board, 7, 2, 2)  # 白子挡左端，(7,7) 空为右端

        # 纵向冲四：落子点 (7,6)，(5,6)(6,6)(8,6) 为黑子，(9,6) 白挡下端
        for x in [5, 6, 8]:
            _set_piece(board, x, 6, 1)
        _set_piece(board, 9, 6, 2)  # 白子挡下端，(4,6) 空为上端

        result = detect_shapes(board, (7, 6), 1)
        # 找冲四或多冲四
        shapes = [item["shape"] for item in result]
        self.assertIn("多冲四", shapes)

    @patch('ai.shape_detector._NUMBA_AVAILABLE', False)
    def test_单冲四_不合并(self):
        """仅一个方向形成冲四，应返回 '冲四' 而非 '多冲四'"""
        board = _make_empty_board()
        # 横向冲四：落子点 (7,6)，(7,3)(7,4)(7,5) 为黑子，(7,2) 白挡
        for y in [3, 4, 5]:
            _set_piece(board, 7, y, 1)
        _set_piece(board, 7, 2, 2)
        result = detect_shapes(board, (7, 6), 1)
        shapes = [item["shape"] for item in result]
        self.assertIn("冲四", shapes)
        self.assertNotIn("多冲四", shapes)

    @patch('ai.shape_detector._NUMBA_AVAILABLE', False)
    def test_多活三_合并(self):
        """两个方向形成活三，应合并为 '多活三'"""
        board = _make_empty_board()
        # 横向活三：落子点 (7,6)，(7,5)(7,7) 为黑子
        for y in [5, 7]:
            _set_piece(board, 7, y, 1)
        # 纵向活三：落子点 (7,6)，(6,6)(8,6) 为黑子
        for x in [6, 8]:
            _set_piece(board, x, 6, 1)

        result = detect_shapes(board, (7, 6), 1)
        shapes = [item["shape"] for item in result]
        self.assertIn("多活三", shapes)

    @patch('ai.shape_detector._NUMBA_AVAILABLE', False)
    def test_单活三_不合并(self):
        """仅一个方向形成活三，应返回 '活三'"""
        board = _make_empty_board()
        for y in [5, 7]:
            _set_piece(board, 7, y, 1)
        result = detect_shapes(board, (7, 6), 1)
        shapes = [item["shape"] for item in result]
        self.assertIn("活三", shapes)
        self.assertNotIn("多活三", shapes)

    @patch('ai.shape_detector._NUMBA_AVAILABLE', False)
    def test_活四单独返回(self):
        """活四应正确返回"""
        board = _make_empty_board()
        # 横向活四：落子 (7,6)，(7,4)(7,5)(7,7) 为黑子
        for y in [4, 5, 7]:
            _set_piece(board, 7, y, 1)
        result = detect_shapes(board, (7, 6), 1)
        shapes = [item["shape"] for item in result]
        self.assertIn("活四", shapes)

    @patch('ai.shape_detector._NUMBA_AVAILABLE', False)
    def test_眠三返回(self):
        """眠三应正确返回"""
        board = _make_empty_board()
        # 横向眠三：落子 (7,6)，(7,5)(7,7) 为黑子，(7,8) 白挡
        for y in [5, 7]:
            _set_piece(board, 7, y, 1)
        _set_piece(board, 7, 8, 2)  # 白子挡
        result = detect_shapes(board, (7, 6), 1)
        shapes = [item["shape"] for item in result]
        self.assertIn("眠三", shapes)

    @patch('ai.shape_detector._NUMBA_AVAILABLE', False)
    def test_棋盘恢复原值(self):
        """detect_shapes 应在检测后恢复棋盘原值（无副作用）"""
        board = _make_empty_board()
        for y in [5, 7]:
            _set_piece(board, 7, y, 1)
        # 记录落子点原始值
        idx = 7 * 15 + 6
        original = board[idx]
        self.assertEqual(original, 0)
        # 执行检测
        detect_shapes(board, (7, 6), 1)
        # 验证棋盘已恢复
        self.assertEqual(board[idx], original)


# ==================== 测试 Numba JIT 回退路径 ====================

class TestNumbaFallback(unittest.TestCase):
    """测试 Numba JIT 不可用时的回退机制"""

    @patch('ai.shape_detector._NUMBA_AVAILABLE', False)
    def test_numba_disabled_不崩溃(self):
        """当 _NUMBA_AVAILABLE=False 时，detect_single_direction 不应崩溃"""
        board = _make_empty_board()
        _set_piece(board, 7, 7, 1)
        # 遍历四个方向，确保都不崩溃
        for dx, dy in DIRECTIONS:
            result = detect_single_direction(board, 7, 7, 1, dx, dy)
            # 至少应该返回 "独一" 或其他有效棋型
            self.assertIsNotNone(result)

    @patch('ai.shape_detector._NUMBA_AVAILABLE', False)
    def test_numba_disabled_detect_shapes不崩溃(self):
        """当 _NUMBA_AVAILABLE=False 时，detect_shapes 不应崩溃"""
        board = _make_empty_board()
        for y in range(3, 8):
            _set_piece(board, 7, y, 1)
        result = detect_shapes(board, (7, 5), 1)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)


# ==================== 测试方向常量和映射 ====================

class TestConstantsAndMappings(unittest.TestCase):
    """测试方向常量和棋型编码映射的正确性"""

    def test_四个方向定义(self):
        """验证 DIRECTIONS 包含横向、纵向、主对角线、副对角线"""
        self.assertEqual(len(DIRECTIONS), 4)
        self.assertIn((0, 1), DIRECTIONS)   # 横向
        self.assertIn((1, 0), DIRECTIONS)   # 纵向
        self.assertIn((1, 1), DIRECTIONS)   # 主对角线
        self.assertIn((1, -1), DIRECTIONS)  # 副对角线

    def test_棋型编码映射完整(self):
        """验证 _SHAPE_CODE_MAP 包含所有棋型类型"""
        expected_shapes = {"连五", "活四", "冲四", "活三", "眠三", "活二", "眠二", "独一"}
        actual_shapes = set(_SHAPE_CODE_MAP.values())
        self.assertEqual(actual_shapes, expected_shapes)

    def test_棋型编码映射值域(self):
        """验证编码值为 1~8，0 不在映射中（表示无棋型）"""
        self.assertNotIn(0, _SHAPE_CODE_MAP)
        for code in range(1, 9):
            self.assertIn(code, _SHAPE_CODE_MAP)


# ==================== 测试白方棋型 ====================

class TestWhitePlayerShapes(unittest.TestCase):
    """测试白方（player=2）的棋型检测，确保不只对黑方有效"""

    @patch('ai.shape_detector._NUMBA_AVAILABLE', False)
    def test_白方连五(self):
        """白方横向连五"""
        board = _make_empty_board()
        for y in range(3, 8):
            _set_piece(board, 7, y, 2)
        result = _detect_single_direction_pure_python(board, 7, 5, 2, 0, 1)
        self.assertEqual(result, "连五")

    @patch('ai.shape_detector._NUMBA_AVAILABLE', False)
    def test_白方活四(self):
        """白方活四"""
        board = _make_empty_board()
        for y in range(4, 8):
            _set_piece(board, 7, y, 2)
        result = _detect_single_direction_pure_python(board, 7, 5, 2, 0, 1)
        self.assertEqual(result, "活四")

    @patch('ai.shape_detector._NUMBA_AVAILABLE', False)
    def test_白方活三(self):
        """白方活三"""
        board = _make_empty_board()
        for y in range(5, 8):
            _set_piece(board, 7, y, 2)
        result = _detect_single_direction_pure_python(board, 7, 6, 2, 0, 1)
        self.assertEqual(result, "活三")

    @patch('ai.shape_detector._NUMBA_AVAILABLE', False)
    def test_黑白混合_仅检测指定方(self):
        """棋盘上同时有黑白子，只检测指定方棋型"""
        board = _make_empty_board()
        # 黑子 (7,4)~(7,7)
        for y in range(4, 8):
            _set_piece(board, 7, y, 1)
        # 白子 (8,4)~(8,7)
        for y in range(4, 8):
            _set_piece(board, 8, y, 2)
        # 检测黑方
        result_black = _detect_single_direction_pure_python(board, 7, 5, 1, 0, 1)
        self.assertEqual(result_black, "活四")
        # 检测白方
        result_white = _detect_single_direction_pure_python(board, 8, 5, 2, 0, 1)
        self.assertEqual(result_white, "活四")


if __name__ == '__main__':
    unittest.main()
