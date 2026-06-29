"""第一级性能优化测试

测试 performance_optimization_proposal.md 第一级三项优化的正确性：
1. 消除 SelfPlay 冗余 predict
2. 棋盘拷贝消除
3. JSON 去缩进 + prepare_tensors 向量化
"""

import sys
import os
import unittest
import json
import shutil
import tempfile
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.engine import SinglePolicyAiEngine
from ai.models.random.model import RandomModel
from ai.shape_detector import detect_shapes
from ai.sample_loader import SampleLoader
from ai.data_recorder import DataRecorder
import torch


# ==================== 优化1：消除冗余 predict ====================

class TestGetMoveWithProbs(unittest.TestCase):
    """测试 AIEngine.get_move 支持预计算 probs 参数

    优化后 get_move 新增可选 probs 参数，避免重复调用 model.predict。
    """

    def setUp(self):
        """测试前准备"""
        self.model = RandomModel()
        self.engine = SinglePolicyAiEngine(self.model)

    def test_get_move_returns_valid_position(self):
        """get_move 应返回合法落子位置，且 probs 通过 get_probs() 获取"""
        board = [[0] * 15 for _ in range(15)]

        decision = self.engine.get_move(board, 1)
        x, y = decision.get_position()
        returned_probs = decision.get_probs()

        self.assertTrue(0 <= x < 15)
        self.assertTrue(0 <= y < 15)
        self.assertEqual(board[x][y], 0)
        self.assertIsNotNone(returned_probs, "get_probs() 应返回概率分布")
        self.assertEqual(len(returned_probs), 225)

    def test_get_move_calls_predict(self):
        """get_move 应调用 model.predict"""
        board = [[0] * 15 for _ in range(15)]
        board_vector = SinglePolicyAiEngine.board_to_vector(board, 1)
        real_probs = self.model.predict(board_vector)

        with patch.object(self.model, 'predict', return_value=real_probs) as mock_predict:
            self.engine.get_move(board, 1)
            mock_predict.assert_called_once()

    def test_get_move_on_partial_board(self):
        """部分填充棋盘，get_move 应正确排除已有棋子"""
        board = [[0] * 15 for _ in range(15)]
        for i in range(5, 10):
            for j in range(5, 10):
                board[i][j] = 1

        decision = self.engine.get_move(board, 2)
        x, y = decision.get_position()

        self.assertTrue(0 <= x < 15)
        self.assertTrue(0 <= y < 15)
        self.assertEqual(board[x][y], 0)
        self.assertFalse(5 <= x < 10 and 5 <= y < 10)


# ==================== 优化2：棋盘拷贝消除 ====================

class TestDetectShapesInPlace(unittest.TestCase):
    """测试 detect_shapes 原地修改-恢复模式

    优化后 detect_shapes 不再复制棋盘，改为原地修改后恢复。
    """

    def test_detect_shapes_board_restored(self):
        """detect_shapes 调用后棋盘应恢复原状"""
        # 构建一个有棋子的棋盘
        board = [0] * 225
        board[7 * 15 + 7] = 1  # (7,7) 黑子
        board[7 * 15 + 8] = 2  # (7,8) 白子
        original_board = board.copy()

        # 在 (7,9) 位置检测黑方棋型
        result = detect_shapes(board, (7, 9), 1)

        # 棋盘应恢复原状
        self.assertEqual(board, original_board,
                         "detect_shapes 调用后棋盘应恢复原状")

    def test_detect_shapes_result_unchanged(self):
        """原地修改优化后，检测结果应与之前完全一致"""
        # 构建多种棋型场景
        board = [0] * 225
        board[7 * 15 + 7] = 1  # (7,7) 黑子
        board[7 * 15 + 8] = 1  # (7,8) 黑子
        board[7 * 15 + 9] = 1  # (7,9) 黑子

        # 先用拷贝模式计算结果
        board_copy = board.copy()
        result_copy = detect_shapes(board_copy, (7, 10), 1)

        # 再用原地模式计算（同一个 board 对象）
        result_inplace = detect_shapes(board, (7, 10), 1)

        # 结果应一致
        self.assertEqual(result_copy, result_inplace,
                         "原地修改模式的检测结果应与拷贝模式一致")

    def test_detect_shapes_empty_position_restored(self):
        """检测空位后应恢复为0"""
        board = [0] * 225
        board[7 * 15 + 7] = 1

        detect_shapes(board, (7, 8), 1)

        # (7,8) 位置应恢复为 0
        self.assertEqual(board[7 * 15 + 8], 0,
                         "检测后空位应恢复为0")

    def test_detect_shapes_multiple_calls_same_board(self):
        """同一棋盘多次调用 detect_shapes 应互不影响"""
        board = [0] * 225
        board[7 * 15 + 7] = 1
        board[7 * 15 + 8] = 1
        original = board.copy()

        # 连续多次调用
        for pos in [(7, 9), (7, 10), (8, 7), (6, 7)]:
            detect_shapes(board, pos, 1)

        # 棋盘应始终保持原状
        self.assertEqual(board, original,
                         "多次调用后棋盘应始终恢复原状")

    def test_detect_shapes_live_three(self):
        """活三检测：原地修改后应正确识别"""
        board = [0] * 225
        board[7 * 15 + 7] = 1  # 黑
        board[7 * 15 + 8] = 1  # 黑
        original = board.copy()

        result = detect_shapes(board, (7, 9), 1)

        # 应检测到活三（两端开放的三连）
        shape_types = [s["shape"] for s in result]
        self.assertIn("活三", shape_types,
                      "应正确识别活三")
        # 棋盘恢复
        self.assertEqual(board, original)

    def test_detect_shapes_four(self):
        """冲四检测：原地修改后应正确识别"""
        board = [0] * 225
        board[7 * 15 + 7] = 1  # 黑
        board[7 * 15 + 8] = 1  # 黑
        board[7 * 15 + 9] = 1  # 黑
        board[7 * 15 + 10] = 2  # 白（挡住一端）
        original = board.copy()

        result = detect_shapes(board, (7, 6), 1)

        shape_types = [s["shape"] for s in result]
        self.assertIn("冲四", shape_types,
                      "应正确识别冲四")
        self.assertEqual(board, original)


# ==================== 优化3a：JSON 去缩进 ====================

class TestJsonCompactFormat(unittest.TestCase):
    """测试 JSON 紧凑格式（去掉 indent=2）

    优化后 json.dump 不再使用 indent=2，减少文件大小。
    """

    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_data_recorder_saves_compact_json(self):
        """DataRecorder 保存的 JSON 应为紧凑格式（无缩进）"""
        recorder = DataRecorder("test_compact", self.test_dir)
        game_id = recorder.new_game()

        # 记录一步
        recorder.record_move(game_id, [0] * 226, [0.1] * 225, (7, 7))
        recorder.save_game_result(game_id, 1, [0] * 226)

        # 读取文件内容
        record_path = os.path.join(recorder.get_round_dir(), "1", "record.json")
        with open(record_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 紧凑格式不应包含缩进（无换行+多空格的模式）
        # indent=2 的格式每行开头有大量空格，紧凑格式则没有
        lines = content.split('\n')
        # 紧凑格式通常只有1行，或行数很少
        # 如果还是 indent=2 格式，会有很多行且每行有前导空格
        indented_lines = sum(1 for line in lines if line.startswith('  '))
        self.assertEqual(indented_lines, 0,
                         "紧凑格式不应有缩进行")

    def test_compact_json_is_valid(self):
        """紧凑格式 JSON 应可正确解析"""
        recorder = DataRecorder("test_valid", self.test_dir)
        game_id = recorder.new_game()
        recorder.record_move(game_id, [0] * 226, [0.1] * 225, (7, 7))
        recorder.save_game_result(game_id, 1, [0] * 226)

        record_path = os.path.join(recorder.get_round_dir(), "1", "record.json")
        with open(record_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 验证数据完整性
        self.assertIn("history", data)
        self.assertIn("winner", data)
        self.assertEqual(data["winner"], 1)
        self.assertEqual(len(data["history"]), 1)

    def test_compact_json_smaller_than_indented(self):
        """紧凑格式文件应比缩进格式小"""
        # 生成同样的数据，比较大小
        data = {
            "history": [{"input": list(range(226)), "output": [0.1] * 225, "position": [7, 7]}],
            "winner": 1,
            "end_board": [0] * 226
        }

        # 紧凑格式
        compact = json.dumps(data, ensure_ascii=False)
        # 缩进格式
        indented = json.dumps(data, ensure_ascii=False, indent=2)

        self.assertLess(len(compact), len(indented),
                        "紧凑格式应比缩进格式小")


# ==================== 优化3b：prepare_tensors 向量化 ====================

class TestPrepareTensorsVectorized(unittest.TestCase):
    """测试 prepare_tensors 向量化优化

    优化后使用列表推导替代逐条 append。
    """

    def setUp(self):
        """测试前准备"""
        self.loader = SampleLoader(device=torch.device("cpu"))

    def test_prepare_tensors_correct_shape(self):
        """向量化后 prepare_tensors 应返回正确形状的张量"""
        sample_data = [
            {"input": [1.0] * 226, "output": [0.1] * 225, "mask": [1] * 225},
            {"input": [2.0] * 226, "output": [0.2] * 225, "mask": [1] * 225},
        ]

        inputs, targets, masks = self.loader.prepare_tensors(sample_data)

        self.assertEqual(inputs.shape, (2, 226))
        self.assertEqual(targets.shape, (2, 225))
        self.assertEqual(masks.shape, (2, 225))

    def test_prepare_tensors_correct_values(self):
        """向量化后张量值应正确"""
        sample_data = [
            {"input": [float(i) for i in range(226)], "output": [0.01 * i for i in range(225)], "mask": [1] * 225},
        ]

        inputs, targets, masks = self.loader.prepare_tensors(sample_data)

        # 验证第一个样本的输入值
        for i in range(226):
            self.assertAlmostEqual(inputs[0, i].item(), float(i), places=5)

    def test_prepare_tensors_empty_data(self):
        """空数据应返回空张量"""
        sample_data = []

        inputs, targets, masks = self.loader.prepare_tensors(sample_data)

        self.assertEqual(inputs.shape[0], 0)
        self.assertEqual(targets.shape[0], 0)
        self.assertEqual(masks.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
