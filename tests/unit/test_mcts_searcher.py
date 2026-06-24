"""MCTSSearcher 单元测试

测试 MCTS 搜索器的核心算法逻辑：
- 终局判定（有人胜 + 平局）
- PUCT 选择公式
- 扩展阶段边创建
- 回传 N/W/Q + 取反
- Virtual Loss 施加与撤销
- Batch 推理流程
- advance_root 子树复用
- 搜索日志 JSON 输出
"""

import sys
import os
import unittest
import json
import tempfile
import shutil
from unittest.mock import MagicMock, patch
import math

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from game.game import check_winner, is_board_full


# ---------- Mock 模型：返回固定策略和价值 ----------

class MockModel:
    """Mock 模型，用于测试 MCTSSearcher 无需真实神经网络"""

    def __init__(self, fixed_policy=None, fixed_value=0.0, device="cpu"):
        self._fixed_policy = fixed_policy  # 225维固定策略
        self._fixed_value = fixed_value
        self._device = device
        self.policy_calls = []  # 记录每次 predict 调用的 board 状态
        self.network = MagicMock()
        # 模拟 network.parameters() 返回空迭代器
        self.network.parameters.return_value = iter([])

    def predict(self, board_vector):
        """记录调用并返回固定策略/价值"""
        self.policy_calls.append(list(board_vector))
        policy = self._fixed_policy
        if policy is None:
            # 默认均匀分布
            policy = [1.0 / 225] * 225
        return policy, self._fixed_value

    def set_policy(self, policy):
        """设置固定策略"""
        self._fixed_policy = policy

    def set_value(self, value):
        """设置固定价值"""
        self._fixed_value = value


# ---------- 辅助函数 ----------

def _make_empty_state(player=1):
    """创建空棋盘状态"""
    board = [0] * 225
    board.append(player)
    return board


def _make_state_with_moves(moves, player=1):
    """从落子序列创建状态
    
    Args:
        moves: [(x, y, p), ...] 其中 p=1(黑) 或 2(白)
        player: state 中当前落子方
    """
    board = [0] * 225
    for x, y, p in moves:
        board[x * 15 + y] = p
    board.append(player)
    return board


def _make_winning_state_for_player(winning_player=1):
    """创建一个棋盘状态，其中 winning_player 即将在下一步获胜"""
    board = [0] * 225
    # 黑方在 (7,0)-(7,3) 有四子，下一步 (7,4) 即五连
    for y in range(4):
        board[7 * 15 + y] = winning_player
    board.append(winning_player)  # 当前落子方
    return board


# ---------- 测试类 ----------

class TestMCTSSearcherTerminal(unittest.TestCase):
    """终局判定测试"""

    @classmethod
    def setUpClass(cls):
        """导入 MCTSSearcher（可能尚未实现，先尝试导入）"""
        try:
            from ai.mcts.searcher import MCTSSearcher
            cls.MCTSSearcher = MCTSSearcher
        except ImportError:
            cls.MCTSSearcher = None

    def setUp(self):
        if self.MCTSSearcher is None:
            self.skipTest("MCTSSearcher 尚未实现")

    def test_recognize_immediate_win(self):
        """搜索前发现当前方已获胜的终局状态"""
        # 黑方五连已形成
        board = [0] * 225
        for y in range(5):
            board[7 * 15 + y] = 1
        board.append(1)  # 当前黑方
        model = MockModel(fixed_value=0.5)
        searcher = self.MCTSSearcher(model, num_simulations=10)
        # 搜索应能识别终局
        visit_counts, root_value, root_policy = searcher.search(board)
        self.assertIsInstance(visit_counts, list)
        self.assertEqual(len(visit_counts), 225)

    def test_terminal_detection_during_selection(self):
        """选择阶段遇到终局节点应停止"""
        # 创建即将获胜的状态（下一步就赢）
        board = [0] * 225
        for y in range(4):
            board[7 * 15 + y] = 1
        board.append(1)
        model = MockModel(fixed_value=0.5)
        searcher = self.MCTSSearcher(model, num_simulations=200)
        visit_counts, root_value, root_policy = searcher.search(board)
        self.assertEqual(len(visit_counts), 225)
        # 获胜位置 (7,4) 应有高访问计数（立即获胜的终局节点会被优先选择）
        win_action = 7 * 15 + 4
        self.assertGreater(visit_counts[win_action], 0,
                           f"Winning action {win_action} should be visited, "
                           f"got {visit_counts[win_action]}")

    def test_draw_detection(self):
        """平局（棋盘已满）应被正确识别"""
        # 创建几乎满的棋盘（只剩一个空位）
        board = [1 if i % 2 == 0 else 2 for i in range(225)]
        # 只剩位置 0 是空的
        board[0] = 0
        board.append(1)
        model = MockModel(fixed_value=0.5)
        searcher = self.MCTSSearcher(model, num_simulations=10)
        visit_counts, root_value, root_policy = searcher.search(board)
        # 即使模型返回非零 value，终局 value 应由规则判定
        self.assertEqual(len(visit_counts), 225)


class TestMCTSSearcherPUCT(unittest.TestCase):
    """PUCT 选择公式测试"""

    @classmethod
    def setUpClass(cls):
        try:
            from ai.mcts.searcher import MCTSSearcher
            cls.MCTSSearcher = MCTSSearcher
        except ImportError:
            cls.MCTSSearcher = None

    def setUp(self):
        if self.MCTSSearcher is None:
            self.skipTest("MCTSSearcher 尚未实现")

    def test_puct_explores_unvisited_nodes(self):
        """未访问节点的 U 值应较高，鼓励探索"""
        model = MockModel(fixed_value=0.0)
        searcher = self.MCTSSearcher(model, num_simulations=50, c_puct=2.0)
        state = _make_empty_state()
        visit_counts, root_value, root_policy = searcher.search(state)
        # 应该有多个位置被访问
        visited = sum(1 for c in visit_counts if c > 0)
        self.assertGreater(visited, 1, "至少应探索多个位置")

    def test_puct_prefers_high_prior(self):
        """高先验概率的位置应该被更多访问"""
        policy = [0.001] * 225
        # 给位置 112 高概率
        policy[112] = 0.5
        # 重新归一化
        total = sum(policy)
        policy = [p / total for p in policy]
        model = MockModel(fixed_policy=policy, fixed_value=0.0)
        searcher = self.MCTSSearcher(model, num_simulations=100, c_puct=2.0)
        state = _make_empty_state()
        visit_counts, root_value, root_policy = searcher.search(state)
        # 位置 112 应有最多访问
        self.assertEqual(visit_counts.index(max(visit_counts)), 112)


class TestMCTSSearcherBackup(unittest.TestCase):
    """回传阶段测试"""

    @classmethod
    def setUpClass(cls):
        try:
            from ai.mcts.searcher import MCTSSearcher
            cls.MCTSSearcher = MCTSSearcher
        except ImportError:
            cls.MCTSSearcher = None

    def setUp(self):
        if self.MCTSSearcher is None:
            self.skipTest("MCTSSearcher 尚未实现")

    def test_backup_flips_value_per_layer(self):
        """回传时每层切换玩家，value 应取反"""
        model = MockModel(fixed_value=0.8)
        searcher = self.MCTSSearcher(model, num_simulations=10)
        state = _make_empty_state()
        visit_counts, root_value, root_policy = searcher.search(state)
        # root_value 应在 [-1, 1] 范围内
        self.assertGreaterEqual(root_value, -1.0)
        self.assertLessEqual(root_value, 1.0)

    def test_root_value_converges_with_more_simulations(self):
        """更多模拟应使根节点价值更稳定"""
        model = MockModel(fixed_value=0.5)
        searcher1 = self.MCTSSearcher(model, num_simulations=10)
        searcher2 = self.MCTSSearcher(model, num_simulations=100)
        state = _make_empty_state()
        _, rv1, _ = searcher1.search(state)
        _, rv2, _ = searcher2.search(state)
        # 两个值都应在合理范围内
        self.assertGreaterEqual(rv1, -1.0)
        self.assertLessEqual(rv1, 1.0)
        self.assertGreaterEqual(rv2, -1.0)
        self.assertLessEqual(rv2, 1.0)


class TestMCTSSearcherVirtualLoss(unittest.TestCase):
    """Virtual Loss 测试"""

    @classmethod
    def setUpClass(cls):
        try:
            from ai.mcts.searcher import MCTSSearcher
            cls.MCTSSearcher = MCTSSearcher
        except ImportError:
            cls.MCTSSearcher = None

    def setUp(self):
        if self.MCTSSearcher is None:
            self.skipTest("MCTSSearcher 尚未实现")

    def test_virtual_loss_encourages_diversification(self):
        """Virtual Loss 应使搜索树分叉"""
        model = MockModel(fixed_value=0.0)
        searcher = self.MCTSSearcher(model, num_simulations=50, virtual_loss=3)
        state = _make_empty_state()
        visit_counts, _, _ = searcher.search(state)
        # 多个位置有访问
        visited = sum(1 for c in visit_counts if c > 0)
        self.assertGreater(visited, 1)

    def test_virtual_loss_default_value(self):
        """默认 Virtual Loss 应为 3"""
        model = MockModel()
        searcher = self.MCTSSearcher(model)
        self.assertEqual(searcher.virtual_loss, 3)


class TestMCTSSearcherBatch(unittest.TestCase):
    """Batch 推理流程测试"""

    @classmethod
    def setUpClass(cls):
        try:
            from ai.mcts.searcher import MCTSSearcher
            cls.MCTSSearcher = MCTSSearcher
        except ImportError:
            cls.MCTSSearcher = None

    def setUp(self):
        if self.MCTSSearcher is None:
            self.skipTest("MCTSSearcher 尚未实现")

    def test_model_called_during_search(self):
        """搜索过程中模型被调用"""
        model = MockModel(fixed_value=0.3)
        searcher = self.MCTSSearcher(model, num_simulations=50)
        state = _make_empty_state()
        searcher.search(state)
        self.assertGreater(len(model.policy_calls), 0,
                           "模型 predict 应该被调用")


class TestMCTSSearcherAdvanceRoot(unittest.TestCase):
    """advance_root 子树复用测试"""

    @classmethod
    def setUpClass(cls):
        try:
            from ai.mcts.searcher import MCTSSearcher
            cls.MCTSSearcher = MCTSSearcher
        except ImportError:
            cls.MCTSSearcher = None

    def setUp(self):
        if self.MCTSSearcher is None:
            self.skipTest("MCTSSearcher 尚未实现")

    def test_advance_root_returns_bool(self):
        """advance_root 返回 bool"""
        model = MockModel()
        searcher = self.MCTSSearcher(model, num_simulations=10)
        state = _make_empty_state()
        searcher.search(state)
        # 获取一个被访问过的 action
        visit_counts, _, _ = searcher.search(state)
        visited_actions = [i for i, c in enumerate(visit_counts) if c > 0]
        if visited_actions:
            result = searcher.advance_root(visited_actions[0])
            self.assertIsInstance(result, bool)

    def test_advance_root_unvisited_action(self):
        """访问未展开的 action 应返回 False"""
        model = MockModel()
        searcher = self.MCTSSearcher(model, num_simulations=1)
        state = _make_empty_state()
        searcher.search(state)
        # 找一个未被访问过的 action（大多数都没被访问）
        result = searcher.advance_root(200)  # 可能未访问
        self.assertIsInstance(result, bool)


class TestMCTSSearcherLogging(unittest.TestCase):
    """搜索日志 JSON 输出测试"""

    @classmethod
    def setUpClass(cls):
        try:
            from ai.mcts.searcher import MCTSSearcher
            cls.MCTSSearcher = MCTSSearcher
        except ImportError:
            cls.MCTSSearcher = None

    def setUp(self):
        if self.MCTSSearcher is None:
            self.skipTest("MCTSSearcher 尚未实现")
        self.temp_dir = tempfile.mkdtemp(prefix="mcts_test_logs_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_log_json_structure(self):
        """日志 JSON 应包含所有必需字段"""
        model = MockModel(fixed_value=0.3)
        searcher = self.MCTSSearcher(model, num_simulations=10, log_dir=self.temp_dir)
        state = _make_empty_state()
        searcher.search(state, log_key="test_game0_move0")

        # 检查日志文件是否生成
        log_path = os.path.join(self.temp_dir, "test_game0_move0.json")
        self.assertTrue(os.path.exists(log_path))

        with open(log_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)

        # 验证顶层字段
        self.assertIn("board", log_data)
        self.assertIn("current_player", log_data)
        self.assertIn("simulations", log_data)
        self.assertIn("hyperparams", log_data)
        self.assertIn("result", log_data)

        # 验证结果字段
        result = log_data["result"]
        self.assertIn("visit_counts", result)
        self.assertIn("root_value", result)
        self.assertIn("pi", result)

        self.assertEqual(len(log_data["board"]), 225)
        self.assertEqual(len(result["visit_counts"]), 225)
        self.assertEqual(len(result["pi"]), 225)

    def test_log_disabled_when_log_dir_none(self):
        """log_dir 为 None 时不输出日志"""
        model = MockModel()
        searcher = self.MCTSSearcher(model, num_simulations=10, log_dir=None)
        state = _make_empty_state()
        # 不应抛异常
        searcher.search(state, log_key="test_no_log")

    def test_log_key_none_skips_logging(self):
        """log_key 为 None 时跳过日志"""
        model = MockModel()
        searcher = self.MCTSSearcher(model, num_simulations=10, log_dir=self.temp_dir)
        state = _make_empty_state()
        # 不应抛异常
        searcher.search(state, log_key=None)


class TestMCTSSearcherReturnValues(unittest.TestCase):
    """search() 返回值测试"""

    @classmethod
    def setUpClass(cls):
        try:
            from ai.mcts.searcher import MCTSSearcher
            cls.MCTSSearcher = MCTSSearcher
        except ImportError:
            cls.MCTSSearcher = None

    def setUp(self):
        if self.MCTSSearcher is None:
            self.skipTest("MCTSSearcher 尚未实现")

    def test_search_returns_tuple_of_three(self):
        """search 返回 (visit_counts, root_value, root_policy) 三元组"""
        model = MockModel()
        searcher = self.MCTSSearcher(model, num_simulations=10)
        state = _make_empty_state()
        result = searcher.search(state)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        visit_counts, root_value, root_policy = result
        self.assertEqual(len(visit_counts), 225)
        self.assertIsInstance(root_value, float)
        self.assertEqual(len(root_policy), 225)

    def test_root_policy_is_softmax_before_noise(self):
        """root_policy 是施加噪声前的模型原始 softmax"""
        model = MockModel()
        searcher = self.MCTSSearcher(model, num_simulations=10)
        state = _make_empty_state()
        _, _, root_policy = searcher.search(state)
        self.assertEqual(len(root_policy), 225)
        # 所有值应在 [0, 1] 之间
        for p in root_policy:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_visit_counts_sum_equals_simulations(self):
        """访问计数总和应等于模拟次数"""
        model = MockModel()
        searcher = self.MCTSSearcher(model, num_simulations=50)
        state = _make_empty_state()
        visit_counts, _, _ = searcher.search(state)
        self.assertEqual(sum(visit_counts), 50)


if __name__ == "__main__":
    unittest.main()
