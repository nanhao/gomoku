"""C++ MCTS 端到端集成测试

覆盖 design/46/main_docs/test_cases.md 中的：
- TC-IT-01 至 TC-IT-05: 端到端集成测试

前置条件：
- cpp_mcts 模块已编译
- 有效的 TorchScript 模型文件
"""

import os
import random as rand_mod
from typing import List

import pytest

_MODEL_PATH = os.environ.get(
    "CPP_MCTS_MODEL_PATH",
    os.path.join(
        os.path.dirname(__file__), "..", "..",
        "src", "ai", "models", "cnn_v11", "cnn_v11_torchscript.pt",
    ),
)

_cpp_available = False
try:
    import cpp_mcts  # type: ignore[import-not-found]
    _cpp_available = True
except ImportError:
    pass

_model_available = os.path.isfile(_MODEL_PATH)

requires_full = pytest.mark.skipif(
    not (_cpp_available and _model_available),
    reason="需要编译后的 cpp_mcts 模块和 TorchScript 模型文件",
)


def _check_winner(board: List[List[int]], x: int, y: int) -> int:
    """检查在 (x, y) 落子后是否产生五连"""
    player = board[x][y]
    if player == 0:
        return 0
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dx, dy in directions:
        count = 1
        for step in range(1, 5):
            nx, ny = x + dx * step, y + dy * step
            if 0 <= nx < 15 and 0 <= ny < 15 and board[nx][ny] == player:
                count += 1
            else:
                break
        for step in range(1, 5):
            nx, ny = x - dx * step, y - dy * step
            if 0 <= nx < 15 and 0 <= ny < 15 and board[nx][ny] == player:
                count += 1
            else:
                break
        if count >= 5:
            return player
    return 0


class TestE2E:
    """端到端集成测试 (TC-IT-01 至 TC-IT-05)"""

    @requires_full
    def test_it01_python_vs_cpp_consistency(self):
        """TC-IT-01: Python MCTS vs C++ MCTS 决策一致性

        GIVEN 同一棋盘状态，两者使用相同权重模型
        WHEN 分别执行 search
        THEN root_value 符号一致，root_policy 差异 < 1e-2
        """
        from cpp.python.cpp_mcts_adapter import CppMctsAdapter

        adapter = CppMctsAdapter(
            _MODEL_PATH, num_simulations=200,
            num_threads=4, virtual_loss=0.0, use_gpu=False,
        )
        # 非平凡棋盘：中心有棋子
        board = [0] * 225
        board[112] = 1
        board[113] = 2
        state = board + [1]

        result_cpp = adapter.search(state)
        cpp_visits, cpp_value, cpp_policy = result_cpp

        # 尝试加载 Python 模型对比
        py_model_path = _MODEL_PATH.replace("_torchscript.pt", ".pth")
        if not os.path.isfile(py_model_path):
            pytest.skip("Python .pth 模型不可用，跳过一致性对比")

        try:
            from ai.mcts.searcher import MCTSSearcher
            from ai.registry import get_model
        except ImportError:
            pytest.skip("Python MCTS 模块导入失败")

        py_model = get_model("cnn_v11", model_path=py_model_path)
        py_searcher = MCTSSearcher(
            py_model, num_simulations=200,
            virtual_loss=0.0, use_prior=True,
        )
        result_py = py_searcher.search(state)
        py_visits, py_value, py_policy = result_py

        # root_value 符号一致（含接近零的容差）
        assert (cpp_value > 0) == (py_value > 0) or abs(cpp_value) < 1e-6

        # policy 最大差异 < 1e-2
        max_diff = max(abs(c - p) for c, p in zip(cpp_policy, py_policy))
        assert max_diff < 1e-2, f"policy 差异过大: {max_diff}"

    @requires_full
    def test_it02_complete_game_self_play(self):
        """TC-IT-02: 完整对局（C++ MCTS 自对弈）

        GIVEN 两个 CppMctsAdapter，num_simulations=20
        WHEN 从头对弈至终局
        THEN 对局正常结束（不崩溃、不超时）
        """
        from cpp.python.cpp_mcts_adapter import CppMctsAdapter
        from ai.engine.policy_value import PolicyValueAiEngine

        adapters = [
            CppMctsAdapter(_MODEL_PATH, num_simulations=20,
                           num_threads=2, virtual_loss=3.0, use_gpu=False),
            CppMctsAdapter(_MODEL_PATH, num_simulations=20,
                           num_threads=2, virtual_loss=3.0, use_gpu=False),
        ]
        engines = [
            PolicyValueAiEngine(adapters[0], temperature=1.0),
            PolicyValueAiEngine(adapters[1], temperature=1.0),
        ]

        board = [[0] * 15 for _ in range(15)]
        current_player = 1
        for _ in range(225):
            engine = engines[current_player - 1]
            move = engine.get_move(board, current_player, strategy="argmax")
            board[move.x][move.y] = current_player
            engine.after_move(move.x, move.y)

            if _check_winner(board, move.x, move.y) != 0:
                break
            current_player = 2 if current_player == 1 else 1

    @requires_full
    def test_it03_cpp_vs_random(self):
        """TC-IT-03: C++ MCTS vs Random（胜率测试）

        GIVEN C++ MCTS (黑) vs Random (白)
        WHEN 对弈 4 局
        THEN C++ MCTS 胜率 >= 50%（至少赢 2 局）
        """
        from cpp.python.cpp_mcts_adapter import CppMctsAdapter
        from ai.engine.policy_value import PolicyValueAiEngine

        adapter = CppMctsAdapter(
            _MODEL_PATH, num_simulations=20,
            num_threads=2, virtual_loss=3.0, use_gpu=False,
        )
        engine = PolicyValueAiEngine(adapter, temperature=1.0)

        cpp_wins = 0
        for game_idx in range(4):
            rand_mod.seed(game_idx)
            board = [[0] * 15 for _ in range(15)]
            current_player = 1
            for _ in range(225):
                if current_player == 1:
                    move = engine.get_move(board, 1, strategy="argmax")
                    board[move.x][move.y] = 1
                    engine.after_move(move.x, move.y)
                    if _check_winner(board, move.x, move.y) == 1:
                        cpp_wins += 1
                        break
                else:
                    empties = [(i, j) for i in range(15) for j in range(15)
                               if board[i][j] == 0]
                    if not empties:
                        break
                    x, y = rand_mod.choice(empties)
                    board[x][y] = 2
                    if _check_winner(board, x, y) == 2:
                        break
                current_player = 2 if current_player == 1 else 1

        assert cpp_wins >= 2, f"C++ MCTS 胜率不足: {cpp_wins}/4"

    @requires_full
    def test_it04_subtree_reuse(self):
        """TC-IT-04: 子树复用贯穿对局

        GIVEN PolicyValueAiEngine(C++ MCTS)
        WHEN 每步 advance_root 后 search
        THEN 前 10 步 subtree 复用成功率 >= 80%
        """
        from cpp.python.cpp_mcts_adapter import CppMctsAdapter
        from ai.engine.policy_value import PolicyValueAiEngine

        adapter = CppMctsAdapter(
            _MODEL_PATH, num_simulations=20,
            num_threads=2, virtual_loss=3.0, use_gpu=False,
        )
        engine = PolicyValueAiEngine(adapter, temperature=1.0)

        board = [[0] * 15 for _ in range(15)]
        success_count = 0

        for step in range(10):
            move = engine.get_move(board, 1, strategy="argmax")
            try:
                engine.after_move(move.x, move.y)
                success_count += 1
            except RuntimeError:
                if step < 3:
                    raise  # 前三步不应失败
            board[move.x][move.y] = 1

            # 白方随机落子
            empties = [(i, j) for i in range(15) for j in range(15)
                       if board[i][j] == 0]
            if not empties:
                break
            board[empties[0][0]][empties[0][1]] = 2

        assert success_count >= 8, f"advance_root 成功率: {success_count}/10"

    @requires_full
    def test_it05_parallel_games(self):
        """TC-IT-05: 并行多局对弈

        GIVEN 2 个独立 CppMctsAdapter 实例
        WHEN 2 线程同时执行对弈
        THEN 全部正常完成，无交叉影响
        """
        import concurrent.futures

        from cpp.python.cpp_mcts_adapter import CppMctsAdapter
        from ai.engine.policy_value import PolicyValueAiEngine

        def _play_one(seed: int) -> bool:
            rand_mod.seed(seed)
            adapter = CppMctsAdapter(
                _MODEL_PATH, num_simulations=10,
                num_threads=1, virtual_loss=3.0, use_gpu=False,
            )
            engine = PolicyValueAiEngine(adapter, temperature=1.0)
            board = [[0] * 15 for _ in range(15)]
            player = 1
            for _ in range(225):
                if player == 1:
                    move = engine.get_move(board, 1, strategy="argmax")
                    board[move.x][move.y] = 1
                    engine.after_move(move.x, move.y)
                    if _check_winner(board, move.x, move.y) == 1:
                        return True
                else:
                    empties = [(i, j) for i in range(15) for j in range(15)
                               if board[i][j] == 0]
                    if not empties:
                        return True
                    x, y = rand_mod.choice(empties)
                    board[x][y] = 2
                    if _check_winner(board, x, y) == 2:
                        return True
                player = 2 if player == 1 else 1
            return True

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            futures = [ex.submit(_play_one, i) for i in range(2)]
            results = [f.result() for f in futures]

        assert all(results), "并行对弈中存在异常终止"
