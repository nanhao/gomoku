"""MCTS 自对弈集成测试

测试完整自对弈流程：模型加载 → MCTS 搜索 → 样本生成 → JSON 输出。
"""

import sys
import os
import unittest
import tempfile
import shutil
import json

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestMCTSSelfPlayIntegration(unittest.TestCase):
    """自对弈集成测试"""

    @classmethod
    def setUpClass(cls):
        """尝试导入 mcts_self_play 模块"""
        try:
            from ai.scripts import mcts_self_play
            cls.mcts_self_play = mcts_self_play
        except ImportError:
            cls.mcts_self_play = None
        try:
            from ai.mcts.searcher import MCTSSearcher
            from ai.mcts.decision_maker import MCTSMoveDecider
            cls.has_mcts = True
        except ImportError:
            cls.has_mcts = False

    def setUp(self):
        if self.mcts_self_play is None and not self.has_mcts:
            self.skipTest("mcts_self_play 模块或 MCTS 模块尚未实现")
        self.temp_dir = tempfile.mkdtemp(prefix="mcts_self_play_test_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_import_modules(self):
        """验证 MCTS 模块可导入"""
        from ai.mcts.searcher import MCTSSearcher
        from ai.mcts.decision_maker import MCTSMoveDecider
        self.assertIsNotNone(MCTSSearcher)
        self.assertIsNotNone(MCTSMoveDecider)

    def test_run_single_game_generates_samples(self):
        """单局自对弈生成 sample.json"""
        from ai.mcts.searcher import MCTSSearcher
        from ai.mcts.decision_maker import MCTSMoveDecider
        from ai.models.cnn_v7.model import CNNV7Model

        # 创建随机初始化模型
        model = CNNV7Model(fresh=True)
        model.move_to_device()

        searcher = MCTSSearcher(model, num_simulations=20, log_dir=None)
        decider = MCTSMoveDecider(searcher, temp_steps=5)

        # 手动模拟一局对弈
        board = [0] * 225
        current_player = 1
        samples = []
        move_count = 0

        from game.game import check_winner, is_board_full

        while move_count < 225:
            state = board + [current_player]
            action, visit_counts, log_path, root_policy = decider.decide(state)

            # 验证 action 合法
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, 225)
            self.assertEqual(board[action], 0, f"Action {action} is not empty")

            # 落子
            board[action] = current_player

            # 记录样本
            pi = [c / sum(visit_counts) if sum(visit_counts) > 0 else 0 for c in visit_counts]
            samples.append({"state": state, "pi": pi, "z": 0})  # z 待填入

            # 子树复用
            decider.make_move(action)

            # 检查终局
            winner = check_winner(board, action)
            if winner != 0:
                break
            if is_board_full(board):
                winner = 0  # 平局
                break

            # 切换玩家
            current_player = 2 if current_player == 1 else 1
            move_count += 1

        # 填入 z 值（按设计文档 z 值翻转算法）
        for sample in samples:
            P = sample["state"][225]
            if winner == 0:
                sample["z"] = 0
            elif winner == P:
                sample["z"] = 1
            else:
                sample["z"] = -1

        # 写入 sample.json
        output_path = os.path.join(self.temp_dir, "sample.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f)

        # 验证文件存在
        self.assertTrue(os.path.exists(output_path))

        # 验证样本结构
        for s in samples:
            self.assertEqual(len(s["state"]), 226)
            self.assertEqual(len(s["pi"]), 225)
            self.assertIn(s["z"], [-1, 0, 1])

    def test_z_value_calculation(self):
        """z 值翻转算法正确性"""
        # 模拟：黑方赢，样本落子方为黑
        self.assertEqual(self._calc_z(1, 1), 1)  # W=黑, P=黑 → +1
        # 模拟：黑方赢，样本落子方为白
        self.assertEqual(self._calc_z(1, 2), -1)  # W=黑, P=白 → -1
        # 模拟：白方赢，样本落子方为白
        self.assertEqual(self._calc_z(2, 2), 1)  # W=白, P=白 → +1
        # 模拟：白方赢，样本落子方为黑
        self.assertEqual(self._calc_z(2, 1), -1)  # W=白, P=黑 → -1
        # 模拟：平局
        self.assertEqual(self._calc_z(0, 1), 0)
        self.assertEqual(self._calc_z(0, 2), 0)

    @staticmethod
    def _calc_z(winner, player):
        """z 值翻转算法（副本，用于测试）"""
        if winner == 0:
            return 0
        elif winner == player:
            return 1
        else:
            return -1

    def test_sample_json_structure(self):
        """sample.json 结构验证"""
        samples = [
            {"state": [0]*225 + [1], "pi": [1.0/225]*225, "z": 1},
            {"state": [0]*225 + [2], "pi": [1.0/225]*225, "z": -1},
        ]
        path = os.path.join(self.temp_dir, "test_sample.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(samples, f)

        with open(path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        self.assertEqual(len(loaded), 2)
        self.assertEqual(len(loaded[0]["state"]), 226)
        self.assertEqual(len(loaded[0]["pi"]), 225)
        self.assertIn(loaded[0]["z"], [-1, 0, 1])


if __name__ == "__main__":
    unittest.main()
