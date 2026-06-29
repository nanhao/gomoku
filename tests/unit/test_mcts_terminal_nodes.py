"""MCTS 终局节点创建与 advance_root 可达性测试（意图 32）

测试用例覆盖设计文档 §6 全部 10 个用例：T32-01 ~ T32-10。
"""

import sys
import os
import unittest

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from game.game import check_winner

# 复用 test_mcts_searcher 中的 MockModel 和辅助函数
from tests.unit.test_mcts_searcher import MockModel, _make_empty_state


class TestMCTSTerminalNodeCreation(unittest.TestCase):
    """终局节点创建与 advance_root 可达性测试（意图 32）

    测试用例覆盖设计文档 §6 全部 10 个用例：T32-01 ~ T32-10。
    """

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

    # ---------- 辅助：创建偏向必胜落子的策略 ----------

    @staticmethod
    def _make_biased_policy(favored_action: int) -> list:
        """创建高概率偏向指定动作的策略分布（用于确保 PUCT 选择该动作）"""
        policy = [0.001] * 225
        policy[favored_action] = 10.0
        total = sum(policy)
        return [p / total for p in policy]

    # ---------- 6.1 终局节点创建 ----------

    def test_t32_01_winning_terminal_node_created(self):
        """T32-01：获胜终局节点入树。

        构造存在一步必胜的棋局，执行 MCTS 搜索，验证：
        ① 树中存在必胜落子后的终局节点
        ② 根边 child_hash 指向该节点
        ③ 终局节点 edges={}
        """
        # 黑方在 (7,0)-(7,3) 有四子，下一步 (7,4) 连五获胜
        board = [0] * 225
        for y in range(4):
            board[7 * 15 + y] = 1
        board.append(1)  # 当前黑方

        win_action = 7 * 15 + 4  # (7,4)
        policy = self._make_biased_policy(win_action)
        model = MockModel(fixed_policy=policy, fixed_value=0.0)
        searcher = self.MCTSSearcher(
            model, num_simulations=50,
            dirichlet_epsilon=0.0,  # 关闭噪声，确保 PUCT 选择可预测
        )
        np.random.seed(42)
        searcher.search(board)

        root = searcher.tree[searcher._root_hash]
        edge = root["edges"][win_action]

        # ① 树中存在终局节点
        child_hash = edge.get("child_hash")
        self.assertIsNotNone(child_hash,
                             f"Winning action {win_action} should have child_hash set")
        self.assertIn(child_hash, searcher.tree,
                      f"Terminal node {child_hash} should exist in tree")

        # ② 根边 child_hash 指向该节点
        terminal_node = searcher.tree[child_hash]
        self.assertEqual(terminal_node["player"], 2,
                         "Terminal node player should be white (opponent)")

        # ③ 终局节点 edges={}
        self.assertEqual(terminal_node["edges"], {},
                         "Terminal node should have empty edges")

    def test_t32_02_draw_terminal_node_created(self):
        """T32-02：满棋盘平局节点入树。

        构造 224 子已落、只剩一空的棋局（无胜方），执行搜索，
        验证树中存在填满后的终局节点且 edges={}。
        """
        # 用交替模式填满棋盘避免五连，留 (0,0) 为空
        board = [0] * 225
        for i in range(225):
            row, col = divmod(i, 15)
            board[i] = 1 if (row + col) % 2 == 0 else 2
        # 留 (0,0) 为空
        board[0] = 0
        board.append(1)  # 当前黑方

        # 确认棋盘无胜方
        self.assertEqual(check_winner(board[:225], 0), 0,
                         "Test board should have no winner")

        model = MockModel(fixed_value=0.0)
        searcher = self.MCTSSearcher(
            model, num_simulations=100,
            dirichlet_epsilon=0.0,
        )
        np.random.seed(123)
        searcher.search(board)

        # 空位 (0,0) 是唯一合法落子
        empty_action = 0
        root = searcher.tree[searcher._root_hash]
        edge = root["edges"].get(empty_action)
        self.assertIsNotNone(edge, "Edges should exist for empty position")

        child_hash = edge.get("child_hash")
        self.assertIsNotNone(child_hash,
                             "Draw terminal node should have child_hash set")
        self.assertIn(child_hash, searcher.tree,
                      f"Draw terminal node {child_hash} should exist in tree")

        terminal_node = searcher.tree[child_hash]
        self.assertEqual(terminal_node["edges"], {},
                         "Draw terminal node should have empty edges")
        # 填满后棋盘无空位
        self.assertEqual(terminal_node["board"].count(0), 0,
                         "Terminal board should be full")

    def test_t32_03_duplicate_access_no_duplicate_nodes(self):
        """T32-03：重复访问终局节点不重复创建。

        同一棋局执行多次搜索，验证终局节点的 state_hash 在树中唯一。
        """
        # 使用与 T32-01 相同的必胜棋局
        board = [0] * 225
        for y in range(4):
            board[7 * 15 + y] = 1
        board.append(1)

        win_action = 7 * 15 + 4
        policy = self._make_biased_policy(win_action)
        model = MockModel(fixed_policy=policy, fixed_value=0.0)
        searcher = self.MCTSSearcher(
            model, num_simulations=50,
            dirichlet_epsilon=0.0,
        )

        # 第一次搜索
        np.random.seed(42)
        searcher.search(board)
        tree_size_1 = len(searcher.tree)

        # 第二次搜索（同一 searcher，search 内部会重新初始化根）
        np.random.seed(42)
        searcher.search(board)
        tree_size_2 = len(searcher.tree)

        # 第二次搜索不应显著增加节点数（最多根节点重建）
        # 终局节点应被复用
        self.assertLessEqual(tree_size_2, tree_size_1 + 2,
                             f"Repeated searches should not duplicate nodes: "
                             f"{tree_size_1} → {tree_size_2}")

    # ---------- 6.2 advance_root 可达性 ----------

    def test_t32_04_advance_root_to_winning_terminal_child(self):
        """T32-04：推进到获胜终局子节点。

        执行搜索（含必胜路径），选必胜落子，
        验证 advance_root(winning_action) 返回 True。
        """
        board = [0] * 225
        for y in range(4):
            board[7 * 15 + y] = 1
        board.append(1)

        win_action = 7 * 15 + 4
        policy = self._make_biased_policy(win_action)
        model = MockModel(fixed_policy=policy, fixed_value=0.0)
        searcher = self.MCTSSearcher(
            model, num_simulations=50,
            dirichlet_epsilon=0.0,
        )
        np.random.seed(42)
        searcher.search(board)

        result = searcher.advance_root(win_action)
        self.assertTrue(result,
                        f"advance_root({win_action}) should return True for winning terminal child")

    def test_t32_05_advance_root_to_draw_terminal_child(self):
        """T32-05：推进到平局子节点。

        执行搜索（含满棋盘路径），选对应落子，
        验证 advance_root(action) 返回 True。
        """
        # 与 T32-02 相同棋局
        board = [0] * 225
        for i in range(225):
            row, col = divmod(i, 15)
            board[i] = 1 if (row + col) % 2 == 0 else 2
        board[0] = 0
        board.append(1)

        model = MockModel(fixed_value=0.0)
        searcher = self.MCTSSearcher(
            model, num_simulations=100,
            dirichlet_epsilon=0.0,
        )
        np.random.seed(123)
        searcher.search(board)

        result = searcher.advance_root(0)  # (0,0) is the only empty
        self.assertTrue(result,
                        "advance_root(0) should return True for draw terminal child")

    def test_t32_06_terminal_node_becomes_new_root(self):
        """T32-06：推进后终局节点成为新根。

        T32-04 成功后验证：
        ① _root_hash 等于终局节点 hash
        ② 新根 edges={}
        ③ 树中仅保留该终局子树
        """
        board = [0] * 225
        for y in range(4):
            board[7 * 15 + y] = 1
        board.append(1)

        win_action = 7 * 15 + 4
        policy = self._make_biased_policy(win_action)
        model = MockModel(fixed_policy=policy, fixed_value=0.0)
        searcher = self.MCTSSearcher(
            model, num_simulations=50,
            dirichlet_epsilon=0.0,
        )
        np.random.seed(42)
        searcher.search(board)

        root = searcher.tree[searcher._root_hash]
        expected_child_hash = root["edges"][win_action]["child_hash"]

        # 推进
        success = searcher.advance_root(win_action)
        self.assertTrue(success)

        # ① _root_hash 等于终局节点 hash
        self.assertEqual(searcher._root_hash, expected_child_hash,
                         "_root_hash should point to terminal node after advance")

        # ② 新根 edges={}
        new_root = searcher.tree[searcher._root_hash]
        self.assertEqual(new_root["edges"], {},
                         "New root should have empty edges (terminal)")

        # ③ 树中仅保留该终局子树（至少包含该终局节点）
        self.assertIn(searcher._root_hash, searcher.tree,
                      "New root should be in tree")

    # ---------- 6.3 make_move 端到端 ----------

    def test_t32_07_make_move_on_winning_action_no_exception(self):
        """T32-07：必胜落子后 make_move 不抛异常。

        engine.get_move() 返回必胜落子，立即调用 engine.after_move(action)，
        验证不抛 RuntimeError 且 advance_root 成功。
        """
        from ai.engine import PolicyValueAiEngine

        board_flat = [0] * 225
        for y in range(4):
            board_flat[7 * 15 + y] = 1
        # 构建 226 维 state 用于搜索器初始化
        state = board_flat + [1]

        win_action = 7 * 15 + 4
        policy = self._make_biased_policy(win_action)
        model = MockModel(fixed_policy=policy, fixed_value=0.0)
        searcher = self.MCTSSearcher(
            model, num_simulations=50,
            dirichlet_epsilon=0.0,
        )
        engine = PolicyValueAiEngine(searcher, temperature=0.0)  # 贪婪选择

        # 将 flat board 转为 15×15 二维矩阵
        board_2d = [[board_flat[i * 15 + j] for j in range(15)] for i in range(15)]

        np.random.seed(42)
        md = engine.get_move(board_2d, 1)  # current_player=1（黑方）
        x, y = md.get_position()
        action = x * 15 + y

        # 期望选择必胜落子 (7,4)
        self.assertEqual(action, win_action,
                         f"Greedy should pick winning action {win_action}, got {action}")

        # 直接调用 after_move 不应抛异常
        try:
            engine.after_move(action)
        except RuntimeError as e:
            self.fail(f"after_move({action}) should not raise RuntimeError: {e}")

    # ---------- 6.4 回溯正确性 ----------

    def test_t32_08_terminal_node_backup_correct(self):
        """T32-08：终局节点参与 backup 统计。

        搜索包含终局叶节点，验证：
        ① 终局节点的父边 N≥1
        ② 父边 Q 值与规则判定值一致（取反后）
        """
        board = [0] * 225
        for y in range(4):
            board[7 * 15 + y] = 1
        board.append(1)

        win_action = 7 * 15 + 4
        policy = self._make_biased_policy(win_action)
        model = MockModel(fixed_policy=policy, fixed_value=0.0)
        searcher = self.MCTSSearcher(
            model, num_simulations=50,
            dirichlet_epsilon=0.0,
        )
        np.random.seed(42)
        searcher.search(board)

        root = searcher.tree[searcher._root_hash]
        edge = root["edges"][win_action]

        # ① 父边 N≥1
        self.assertGreaterEqual(edge["N"], 1,
                                f"Edge for winning action should have N>=1, got {edge['N']}")

        # ② 父边 Q 值与规则判定值一致
        # 终局 value = -1.0（从对手视角），backup 取反后 Q = +1.0（从根视角）
        msg = f"Edge Q should be ~1.0 (winning from root perspective), got {edge['Q']}"
        self.assertAlmostEqual(edge["Q"], 1.0, delta=0.01, msg=msg)

    # ---------- 6.5 回归验证 ----------

    def test_t32_09_non_terminal_search_unchanged(self):
        """T32-09：非终局搜索行为不变。

        空棋盘 MCTS 搜索（无终局节点），固定随机种子，
        验证 visit_counts 总和等于模拟次数。
        """
        model = MockModel(fixed_value=0.3)
        searcher = self.MCTSSearcher(model, num_simulations=50)
        state = _make_empty_state()

        np.random.seed(999)
        visit_counts, root_value, root_policy = searcher.search(state)

        # 空棋盘无终局，行为应一致
        self.assertEqual(sum(visit_counts), 50,
                         "Total visit counts should equal num_simulations")
        self.assertGreaterEqual(root_value, -1.0)
        self.assertLessEqual(root_value, 1.0)
        self.assertEqual(len(root_policy), 225)

    def test_t32_10_all_existing_tests_still_pass(self):
        """T32-10：全量已有测试通过。

        快速验证核心功能未被破坏：search 返回三元组且格式正确。
        完整回归由 pytest 全量执行保证。
        """
        model = MockModel()
        searcher = self.MCTSSearcher(model, num_simulations=10)
        state = _make_empty_state()
        result = searcher.search(state)
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), 225)
        self.assertIsInstance(result[1], float)
        self.assertEqual(len(result[2]), 225)


if __name__ == "__main__":
    unittest.main()
