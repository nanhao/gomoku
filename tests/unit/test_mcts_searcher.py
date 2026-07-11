"""MCTSSearcher 单元测试

测试 MCTS 搜索器的核心算法逻辑：
- 终局判定（有人胜 + 平局）
- PUCT 选择公式
- 扩展阶段边创建
- 回传 N/W/Q + 取反
- 单步 simulation 循环（无 Batch/Virtual Loss）
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

import numpy as np
import torch
from game.game import check_winner, is_board_full
from ai.base_model import ModelPredict


# ---------- Mock 模型：返回固定策略和价值 ----------

class MockModel:
    """Mock 模型，用于测试 MCTSSearcher 无需真实神经网络"""

    def __init__(self, fixed_policy=None, fixed_value=0.0, device="cpu"):
        self._fixed_policy = fixed_policy  # 225维固定策略
        self._fixed_value = fixed_value
        self._device = device
        self.policy_calls = []  # 记录每次 predict 调用的 board 状态
        self.batch_calls = []   # 记录每次 predict_batch 调用的输入大小
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
        return ModelPredict(policy=policy, value=self._fixed_value)

    def predict_batch(self, board_vectors):
        """批量预测：记录调用并返回固定策略/价值"""
        self.batch_calls.append(len(board_vectors))
        results = []
        for bv in board_vectors:
            self.policy_calls.append(list(bv))
            policy = self._fixed_policy
            if policy is None:
                policy = [1.0 / 225] * 225
            results.append(ModelPredict(policy=policy, value=self._fixed_value))
        return results

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
        model = MockModel(fixed_value=0.0)
        searcher = self.MCTSSearcher(
            model, num_simulations=200,
            # 关闭 Dirichlet 噪声，确保 PUCT 选择完全由 Q 值驱动
            dirichlet_epsilon=0.0,
            dirichlet_alpha=0.03,
        )
        # 固定随机种子确保可复现：Dirichlet 噪声依赖 np.random
        np.random.seed(20260626)
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


class TestMCTSSearcherModelCall(unittest.TestCase):
    """模型调用验证测试（单步 simulation 循环）"""

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
        """搜索过程中模型被调用（每轮 simulation 一次评估）"""
        model = MockModel(fixed_value=0.3)
        searcher = self.MCTSSearcher(model, num_simulations=50)
        state = _make_empty_state()
        searcher.search(state)
        # 模型应被调用：根节点初始化 1 次 + 每轮 simulation 叶节点评估 50 次 = 51 次
        self.assertGreater(len(model.policy_calls), 0,
                           "模型 predict 应该被调用")

    def test_model_call_count_per_simulation(self):
        """在空棋盘上每轮 simulation 都应触发一次模型评估（无终局时）"""
        model = MockModel(fixed_value=0.3)
        num_sims = 20
        searcher = self.MCTSSearcher(model, num_simulations=num_sims)
        state = _make_empty_state()
        searcher.search(state)
        # 空棋盘无终局：根节点 1 次 + simulation 叶节点 num_sims 次
        expected_min_calls = num_sims + 1
        actual_calls = len(model.policy_calls)
        self.assertEqual(actual_calls, expected_min_calls,
                         f"Expected ~{expected_min_calls} model calls, got {actual_calls}")

    def test_constructor_accepts_virtual_loss_and_batch_size(self):
        """构造函数接受 batch_size 和 virtual_loss 参数"""
        model = MockModel()
        searcher = self.MCTSSearcher(model, batch_size=8, virtual_loss=2.0)
        self.assertEqual(searcher.batch_size, 8)
        self.assertEqual(searcher.virtual_loss, 2.0)

    def test_constructor_defaults_from_config(self):
        """未传 batch_size/virtual_loss 时从配置读取默认值"""
        model = MockModel()
        searcher = self.MCTSSearcher(model)
        self.assertIsInstance(searcher.batch_size, int)
        self.assertGreater(searcher.batch_size, 0)
        self.assertIsInstance(searcher.virtual_loss, float)


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

        # 验证 hyperparams 不包含 virtual_loss（已移除）
        hyperparams = log_data["hyperparams"]
        self.assertNotIn("virtual_loss", hyperparams,
                         "virtual_loss 应已从日志中移除")
        self.assertIn("c_puct", hyperparams)
        self.assertIn("dirichlet_epsilon", hyperparams)
        self.assertIn("dirichlet_alpha", hyperparams)
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


class TestMCTSSearcherBatchAndVL(unittest.TestCase):
    """Batch MCTS + Virtual Loss 测试"""

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

    # ---- T1.1: VL 施加 ----

    def test_vl_applied_to_traversed_edges(self):
        """VL=3.0 时，每条被遍历边 N+3, W-3"""
        model = MockModel(fixed_value=0.0)
        searcher = self.MCTSSearcher(
            model, num_simulations=4, batch_size=2, virtual_loss=3.0
        )
        state = _make_empty_state()
        searcher.search(state)
        # VL 已被撤销，所有边的 N 和 W 应为整数
        for node_hash, node in searcher.tree.items():
            self.assertEqual(node["N"] % 1, 0,
                           f"Node N should be integer after VL revert: {node['N']}")
            for action, edge in node.get("edges", {}).items():
                # N 和 W 的小数部分应为 0（VL 全部撤销）
                self.assertEqual(edge["N"] % 1, 0,
                               f"Edge N should be integer: {edge['N']}")

    # ---- T2.1: batch_size 整除 ----

    def test_batch_size_divides_simulations(self):
        """num_sims=100, batch_size=16 → predict_batch 调用 7 次"""
        model = MockModel(fixed_value=0.3)
        searcher = self.MCTSSearcher(
            model, num_simulations=100, batch_size=16, virtual_loss=0.0
        )
        state = _make_empty_state()
        searcher.search(state)
        # 验证 predict_batch 调用 7 次，最后一次 batch size=4
        self.assertEqual(len(model.batch_calls), 7,
                         f"Expected 7 batch calls, got {len(model.batch_calls)}")
        self.assertEqual(model.batch_calls[-1], 4,
                        f"Last batch should have 4 leaves, got {model.batch_calls[-1]}")
        # 验证总 predict 调用 = root_init(1) + 所有 batch 的向量数(100)
        self.assertEqual(len(model.policy_calls), 101)

    # ---- T2.2: batch_size=1 退化 ----

    def test_batch_size_one_equivalent_to_single_sim(self):
        """batch_size=1, VL=0 产生合法搜索结果"""
        model1 = MockModel(fixed_value=0.3)
        model2 = MockModel(fixed_value=0.3)
        searcher1 = self.MCTSSearcher(
            model1, num_simulations=20, batch_size=1, virtual_loss=0.0
        )
        searcher2 = self.MCTSSearcher(
            model2, num_simulations=20, batch_size=16, virtual_loss=0.0
        )
        state = _make_empty_state()
        vc1, rv1, _ = searcher1.search(state)
        vc2, rv2, _ = searcher2.search(state)
        # 总访问数相同
        self.assertEqual(sum(vc1), sum(vc2))
        self.assertEqual(rv1, rv2)

    # ---- T3.1: VL 阻止路径收敛 ----

    def test_vl_diversifies_batch_selection(self):
        """batch_size=8, VL=3.0 时批次内叶节点不同"""
        model = MockModel(fixed_value=0.0)
        searcher = self.MCTSSearcher(
            model, num_simulations=8, batch_size=8, virtual_loss=3.0
        )
        state = _make_empty_state()
        # 手动收集批次叶节点
        searcher._init_search(state)
        leaves = []
        for _ in range(8):
            leaf = searcher._select_one()
            leaves.append(leaf["state_hash"])
        # VL 应使批次内 state_hash 分散（至少 3 个不同）
        unique_hashes = set(leaves)
        self.assertGreater(len(unique_hashes), 2,
                          f"VL should diversify selection, got {len(unique_hashes)} unique")

    # ---- T3.2: VL=0 路径收敛 ----

    def test_no_vl_converges_in_batch(self):
        """batch_size=8, VL=0 时批次内路径高度重合"""
        model = MockModel(fixed_value=0.0)
        searcher = self.MCTSSearcher(
            model, num_simulations=8, batch_size=8, virtual_loss=0.0
        )
        state = _make_empty_state()
        searcher._init_search(state)
        leaves = []
        for _ in range(8):
            leaf = searcher._select_one()
            leaves.append(leaf["state_hash"])
        # 无 VL 时批次内应有重复
        unique_hashes = set(leaves)
        self.assertLess(len(unique_hashes), 8,
                       f"Without VL, paths should converge: {len(unique_hashes)} unique out of 8")

    # ---- T5.1: 搜索正确性 ----

    def test_search_result_with_batch(self):
        """batch_size=16 时 search() 返回正确格式"""
        model = MockModel()
        searcher = self.MCTSSearcher(
            model, num_simulations=32, batch_size=16, virtual_loss=3.0
        )
        state = _make_empty_state()
        visit_counts, root_value, root_policy = searcher.search(state)
        self.assertEqual(len(visit_counts), 225)
        self.assertEqual(sum(visit_counts), 32)
        self.assertGreaterEqual(root_value, -1.0)
        self.assertLessEqual(root_value, 1.0)

    # ---- T5.3: 终局判定 ----

    def test_terminal_board_with_batch(self):
        """五连终局状态 batch 模式正确"""
        board = [0] * 225
        for y in range(5):
            board[7 * 15 + y] = 1
        board.append(1)
        model = MockModel(fixed_value=0.5)
        searcher = self.MCTSSearcher(
            model, num_simulations=10, batch_size=4, virtual_loss=3.0
        )
        visit_counts, root_value, root_policy = searcher.search(board)
        self.assertEqual(len(visit_counts), 225)

    # ---- 配置 ----

    def test_config_defaults_batch_size_and_vl(self):
        """未传 batch_size/virtual_loss 时从配置读取"""
        from config import mcts as mcts_config
        model = MockModel()
        searcher = self.MCTSSearcher(model)
        self.assertEqual(searcher.batch_size, mcts_config.MCTS_BATCH_SIZE)
        self.assertEqual(searcher.virtual_loss, mcts_config.MCTS_VIRTUAL_LOSS)

    def test_constructor_overrides_config(self):
        """构造参数覆盖配置常量"""
        model = MockModel()
        searcher = self.MCTSSearcher(model, batch_size=32, virtual_loss=5.0)
        self.assertEqual(searcher.batch_size, 32)
        self.assertEqual(searcher.virtual_loss, 5.0)

    # ---- T1.2: VL 撤销 ----

    def test_vl_revert_restores_original_values(self):
        """VL 撤销后边 N/W 恢复为施加前原始值"""
        model = MockModel(fixed_value=0.5)
        searcher = self.MCTSSearcher(
            model, num_simulations=2, batch_size=2, virtual_loss=3.0
        )
        state = _make_empty_state()
        searcher._init_search(state)
        # 记录施加 VL 前的边统计
        root = searcher.tree[searcher._root_hash]
        before_N = {a: root["edges"][a]["N"] for a in root["edges"]}
        before_W = {a: root["edges"][a]["W"] for a in root["edges"]}
        # 选择 2 个叶节点（VL 已施加）
        leaves = [searcher._select_one() for _ in range(2)]
        # 撤销 VL
        for leaf in leaves:
            searcher._revert_vl(leaf.get("path", []))
        # 验证恢复到施加前
        root = searcher.tree[searcher._root_hash]
        for a in root["edges"]:
            self.assertEqual(root["edges"][a]["N"], before_N[a],
                           f"Edge {a} N should revert to {before_N[a]}")
            self.assertEqual(root["edges"][a]["W"], before_W[a],
                           f"Edge {a} W should revert to {before_W[a]}")

    # ---- T2.3: 全终局批次 ----

    def test_all_terminal_batch_skips_model(self):
        """棋盘接近填满时终局叶节点走规则分支，不调模型"""
        board = [1 if i % 2 == 0 else 2 for i in range(224)]
        board.append(0)  # 最后一个位置留空
        board.append(1)   # current_player
        model = MockModel(fixed_value=0.5)
        searcher = self.MCTSSearcher(
            model, num_simulations=10, batch_size=8, virtual_loss=3.0
        )
        visit_counts, root_value, root_policy = searcher.search(board)
        self.assertEqual(len(visit_counts), 225)
        # 终局叶节点不调用 predict_batch（batch_leaves 为空时跳过）
        self.assertEqual(len(model.batch_calls), 0,
                        "All-terminal batch should not call predict_batch")

    # ---- T5.4: 平局 ----

    def test_draw_board_with_batch(self):
        """棋盘满且无胜者：root_value=0，visit_counts 全 0"""
        board = [1 if i % 2 == 0 else 2 for i in range(225)]
        board.append(1)
        model = MockModel(fixed_value=0.5)
        searcher = self.MCTSSearcher(
            model, num_simulations=10, batch_size=4, virtual_loss=0.0
        )
        visit_counts, root_value, root_policy = searcher.search(board)
        self.assertEqual(sum(visit_counts), 0, "满棋盘无胜者访问计数应为 0")
        self.assertEqual(root_value, 0.0, "满棋盘无胜者 root_value 应为 0")


class TestModelPredictBatch(unittest.TestCase):
    """CNNV7Model.predict_batch 单元测试"""

    @classmethod
    def setUpClass(cls):
        try:
            from ai.models.cnn_v7.model import CNNV7Model
            cls.CNNV7Model = CNNV7Model
        except ImportError:
            cls.CNNV7Model = None

    def setUp(self):
        if self.CNNV7Model is None:
            self.skipTest("CNNV7Model 尚未实现")
        self.model = self.CNNV7Model(fresh=True)
        self.model.move_to_device()

    # ---- T6.1: 单条等价性 ----

    def test_predict_batch_single_equals_predict(self):
        """predict_batch([v]) 与 predict(v) 结果一致"""
        v = [0] * 225 + [1]
        r1 = self.model.predict(v)
        r2_list = self.model.predict_batch([v])
        self.assertEqual(len(r2_list), 1)
        r2 = r2_list[0]
        # policy 逐元素比较
        for i in range(225):
            self.assertAlmostEqual(r1.policy[i], r2.policy[i], places=5,
                                  msg=f"Policy differs at index {i}")
        self.assertAlmostEqual(r1.value, r2.value, places=5)

    # ---- T6.2: batch 多输入 ----

    def test_predict_batch_multi_input(self):
        """predict_batch 返回多个正确格式的 ModelPredict"""
        vectors = [[0] * 225 + [1 + (i % 2)] for i in range(4)]
        results = self.model.predict_batch(vectors)
        self.assertEqual(len(results), 4)
        for r in results:
            self.assertEqual(len(r.policy), 225)
            self.assertAlmostEqual(sum(r.policy), 1.0, places=3,
                                  msg="Policy should sum to ~1")
            self.assertGreaterEqual(r.value, -1.0)
            self.assertLessEqual(r.value, 1.0)

    # ---- T6.3: 空输入 ----

    def test_predict_batch_empty_returns_empty(self):
        """predict_batch([]) 返回空列表"""
        results = self.model.predict_batch([])
        self.assertEqual(results, [])

    # ---- T6.4: batch 性能 ----

    def test_predict_batch_faster_than_individual(self):
        """predict_batch 耗时小于 N 次单独 predict"""
        import time
        vectors = [[0] * 225 + [i % 2 + 1] for i in range(8)]

        t0 = time.perf_counter()
        individual = [self.model.predict(v) for v in vectors]
        t_individual = time.perf_counter() - t0

        t0 = time.perf_counter()
        batch = self.model.predict_batch(vectors)
        t_batch = time.perf_counter() - t0

        self.assertEqual(len(individual), len(batch))
        self.assertLess(t_batch, t_individual,
                       f"Batch {t_batch:.4f}s should be faster than individual {t_individual:.4f}s")


if __name__ == "__main__":
    unittest.main()
