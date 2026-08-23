"""cnn_v4_self_play 脚本单元测试（intention 54）

测试 cnn_v4.3 自我对弈核心逻辑：
- _play_one_game 的样本结构与 z 值回填（胜负/平局）
- 温度采样策略编排（前 temp_steps 步 sampling，之后 argmax）
- parse_args 默认值与 CLI 参数解析

使用 Fake 模型（SequentialFakeModel），不加载真实权重，不修改真实数据文件。
"""

import os
import sys
import unittest
from unittest import mock

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
_src_dir = os.path.join(project_root, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from ai.base_model import ModelPredict  # noqa: E402


class SequentialFakeModel:
    """顺序落子 Fake 模型（无状态，可跨局复用）

    为每个玩家维护一个固定位置序列，predict 时返回该玩家序列中
    "第一个尚未被占据"的位置概率最高（1.0），其余 0。
    用于驱动对局走向可预测的终局（连五/平局），避免真实模型随机性；
    因基于棋盘状态推导而非计数器，多局复用不会累积状态。
    """

    def __init__(self, black_seq, white_seq):
        """初始化

        Args:
            black_seq: 黑方（玩家 1）依次落子位置列表（0-224）
            white_seq: 白方（玩家 2）依次落子位置列表（0-224）
        """
        self._seq = {1: list(black_seq), 2: list(white_seq)}
        self.network = None
        self.predict_calls = []

    def predict(self, board_vector):
        """返回当前玩家序列中第一个空位概率最高的分布"""
        player = board_vector[225]
        board = board_vector[:225]
        target = next(p for p in self._seq[player] if board[p] == 0)

        probs = [0.0] * 225
        probs[target] = 1.0  # 确定性输出：目标位置概率 1，其余 0，保证采样/argmax 必然命中
        # 记录调用（board_vector, target）供测试断言
        self.predict_calls.append((list(board_vector), target))
        return ModelPredict(policy=probs)

    def move_to_device(self, device):
        """兼容模型接口：空实现"""
        return device


class DefenseFakeModel:
    """防守型 Fake 模型：保证对局必然平局

    策略：优先堵截对手的 4 连（含活四/冲四），无 4 连时下第一个空位。
    因对手任何 4 连都会立即被堵截，永远无法形成 5 连，
    最终棋盘下满平局结束（完全确定性，无随机）。
    """

    _DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]

    def __init__(self):
        self.network = None

    def predict(self, board_vector):
        """返回堵截点（或第一个空位）概率最高的分布"""
        player = board_vector[225]
        board = list(board_vector[:225])
        opponent = 2 if player == 1 else 1
        target = self._find_block_point(board, opponent)
        if target is None:
            target = next(i for i in range(225) if board[i] == 0)
        probs = [0.0] * 225
        probs[target] = 1.0
        return ModelPredict(policy=probs)

    def _find_block_point(self, board, opponent):
        """找到对手 4 连的可堵截空位；无则返回 None"""
        for i in range(225):
            if board[i] != opponent:
                continue
            x, y = i // 15, i % 15
            for dx, dy in self._DIRS:
                # 沿方向数连续 opponent 棋子（向两端延伸）
                bx, by = x - dx, y - dy
                count = 1
                while 0 <= bx < 15 and 0 <= by < 15 and board[bx * 15 + by] == opponent:
                    count += 1
                    bx -= dx
                    by -= dy
                fx, fy = x + dx, y + dy
                while 0 <= fx < 15 and 0 <= fy < 15 and board[fx * 15 + fy] == opponent:
                    count += 1
                    fx += dx
                    fy += dy
                if count >= 4:
                    # 两端空位任选其一堵截
                    for ex, ey in ((bx, by), (fx, fy)):
                        if 0 <= ex < 15 and 0 <= ey < 15 and board[ex * 15 + ey] == 0:
                            return ex * 15 + ey
        return None

    def move_to_device(self, device):
        """兼容模型接口：空实现"""
        return device


def _black_win_seq():
    """构造黑方第 1-5 步连五、白方远端落子的序列

    黑方依次落 (0,0)(0,1)(0,2)(0,3)(0,4) 第 5 步连五获胜；
    白方落在远端不阻挡黑方（黑 5 步之间需白 4 步间隔）。
    """
    black = [0, 1, 2, 3, 4]                          # (0,0)-(0,4)
    white = [14 * 15 + 0, 14 * 15 + 2, 14 * 15 + 4, 14 * 15 + 6]  # 远端行
    return black, white


class TestPlayOneGameSampleStructure(unittest.TestCase):
    """_play_one_game 样本结构与 z 值回填"""

    def test_black_win_backfills_z(self):
        """黑方连五获胜：样本结构与 z 回填正确"""
        from ai.scripts.cnn_v4_self_play import _play_one_game

        black, white = _black_win_seq()
        model = SequentialFakeModel(black, white)

        samples = _play_one_game(model, temperature=1.0, temp_steps=12)

        # 黑第 5 步连五：共 9 手（黑 5 + 白 4），9 个样本
        self.assertEqual(len(samples), 9)

        # 样本结构：state 226 维、pi 225 维、z 键存在
        for s in samples:
            self.assertEqual(len(s["state"]), 226)
            self.assertEqual(len(s["pi"]), 225)
            self.assertIn("z", s)

        # 黑方视角样本 z=1，白方视角 z=-1
        for s in samples:
            player = s["state"][225]
            if player == 1:
                self.assertEqual(s["z"], 1)
            else:
                self.assertEqual(s["z"], -1)

        # 最后一个样本是黑方（第 9 手前）
        self.assertEqual(samples[-1]["state"][225], 1)

    def test_draw_backfills_z_zero(self):
        """防守对局下满棋盘平局：所有样本 z=0"""
        from ai.scripts.cnn_v4_self_play import _play_one_game

        # 防守模型双方互相堵截 4 连，永不五连，下满 225 格平局
        model = DefenseFakeModel()

        samples = _play_one_game(model, temperature=1.0, temp_steps=12)

        # 225 手下满棋盘，共 225 个样本（最后一步前记录）
        self.assertEqual(len(samples), 225)
        for s in samples:
            self.assertEqual(s["z"], 0)

    def test_pi_is_model_output_distribution(self):
        """pi 直接采用模型输出概率分布（含非法位置概率）"""
        from ai.scripts.cnn_v4_self_play import _play_one_game

        black, white = _black_win_seq()
        model = SequentialFakeModel(black, white)

        samples = _play_one_game(model, temperature=1.0, temp_steps=12)

        # 每个样本的 pi 与模型 predict 输出完全一致（不掩码、不重归一化）
        for i, s in enumerate(samples):
            model_calls = model.predict_calls
            # 第 i 个样本对应第 i 次 predict（state 相同）
            self.assertEqual(s["state"], model_calls[i][0])
            # 模型输出中目标位置概率最高
            target = model_calls[i][1]
            self.assertGreater(s["pi"][target], 0.5)
            # pi 总和接近 1（模型 softmax 输出）
            self.assertAlmostEqual(sum(s["pi"]), 1.0, places=4)

    def test_state_is_before_move_snapshot(self):
        """样本 state 是落子前局面，落子后棋盘对应位置被占据"""
        from ai.scripts.cnn_v4_self_play import _play_one_game

        black, white = _black_win_seq()
        model = SequentialFakeModel(black, white)

        samples = _play_one_game(model, temperature=1.0, temp_steps=12)

        # 第 i 个样本记录后，黑/白按序列落子；下一个样本中对应位置应被占据
        expected_board = [0] * 225
        for i, s in enumerate(samples):
            self.assertEqual(s["state"][:225], expected_board)
            player = s["state"][225]
            # 黑方第 k 次落子在样本 index=2k-2，白方第 k 次在 index=2k-1
            if player == 1:
                target = model._seq[1][i // 2]
            else:
                target = model._seq[2][(i - 1) // 2]
            expected_board[target] = player


class TestPlayOneGameStrategy(unittest.TestCase):
    """_play_one_game 温度采样策略编排"""

    def _run_with_strategy_recorder(self, temp_steps):
        """运行一局并记录每次 get_move 收到的 strategy"""
        from ai.scripts import cnn_v4_self_play

        black, white = _black_win_seq()
        model = SequentialFakeModel(black, white)
        strategies = []

        class _RecordingEngine:
            """记录 strategy 的引擎替身"""

            def __init__(self, *a, **kw):
                pass

            def get_move(self, board_2d, current_player, strategy="argmax"):
                strategies.append(strategy)
                # 用真实模型语义：选择 fake 模型概率最高的位置
                board_vector = [0] * 226
                for x in range(15):
                    for y in range(15):
                        board_vector[x * 15 + y] = board_2d[x][y]
                board_vector[225] = current_player
                result = model.predict(board_vector)
                probs = result.policy
                best = max(range(225), key=lambda i: probs[i])
                from ai.engine import MoveDecision
                return MoveDecision(best // 15, best % 15, probs=probs)

        with mock.patch.object(cnn_v4_self_play, "SinglePolicyAiEngine", _RecordingEngine):
            samples = cnn_v4_self_play._play_one_game(
                model, temperature=1.0, temp_steps=temp_steps
            )

        return strategies, samples

    def test_sampling_before_temp_steps_then_argmax(self):
        """前 temp_steps 步 sampling，之后 argmax"""
        strategies, samples = self._run_with_strategy_recorder(temp_steps=4)

        # 共 9 手：前 4 手 sampling，后 5 手 argmax
        self.assertEqual(len(strategies), 9)
        self.assertEqual(strategies[:4], ["sampling"] * 4)
        self.assertEqual(strategies[4:], ["argmax"] * 5)

    def test_temp_steps_zero_disables_sampling(self):
        """temp_steps=0 时全程 argmax"""
        strategies, samples = self._run_with_strategy_recorder(temp_steps=0)

        self.assertEqual(strategies, ["argmax"] * 9)

    def test_temp_steps_larger_than_game_all_sampling(self):
        """temp_steps 大于对局步数时全程 sampling"""
        strategies, samples = self._run_with_strategy_recorder(temp_steps=100)

        self.assertEqual(strategies, ["sampling"] * 9)


class TestParseArgs(unittest.TestCase):
    """parse_args 默认值与参数解析"""

    def test_defaults(self):
        """默认值与 intention 54 一致"""
        from ai.scripts.cnn_v4_self_play import parse_args

        args = parse_args([])
        self.assertEqual(args.games, 100)
        self.assertEqual(args.temperature, 1.0)
        self.assertEqual(args.temp_steps, 12)
        self.assertEqual(args.output, "samples/boots/1")
        self.assertEqual(args.model, "cnn_v4.3")

    def test_custom_values(self):
        """自定义参数生效"""
        from ai.scripts.cnn_v4_self_play import parse_args

        args = parse_args([
            "--games", "50",
            "--temperature", "0.5",
            "--temp-steps", "8",
            "--output", "tmp_out",
        ])
        self.assertEqual(args.games, 50)
        self.assertEqual(args.temperature, 0.5)
        self.assertEqual(args.temp_steps, 8)
        self.assertEqual(args.output, "tmp_out")


if __name__ == "__main__":
    unittest.main()
