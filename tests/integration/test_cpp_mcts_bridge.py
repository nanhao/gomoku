"""C++ MCTS 桥接层与适配器集成测试

覆盖 design/46/main_docs/test_cases.md 中的：
- TC-PB-01 至 TC-PB-05: pybind11 桥接层测试
- TC-AD-01 至 TC-AD-03: Python 适配层测试

前置条件：
- cpp_mcts 模块已编译（cpp_mcts.pyd 在 Python 路径中）
- 有效的 TorchScript 模型文件

未满足前置条件时，相关测试会被跳过。
"""

import os
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

requires_cpp = pytest.mark.skipif(
    not _cpp_available,
    reason="需要编译后的 cpp_mcts 模块",
)


class TestPybind11Bridge:
    """pybind11 桥接层测试 (TC-PB-01 至 TC-PB-05)"""

    @requires_cpp
    def test_pb01_module_import(self):
        """TC-PB-01: 模块导入成功"""
        import cpp_mcts
        assert hasattr(cpp_mcts, "MctsSearcher")

    @requires_cpp
    def test_pb02_constructor_defaults(self):
        """TC-PB-02: 构造器默认参数"""
        if not _model_available:
            pytest.skip("模型文件不可用")
        searcher = cpp_mcts.MctsSearcher(_MODEL_PATH)
        assert searcher is not None

    @requires_full
    def test_pb03_search_return_type(self):
        """TC-PB-03: search 返回类型"""
        searcher = cpp_mcts.MctsSearcher(
            _MODEL_PATH, num_simulations=10,
            num_threads=1, virtual_loss=0.0, use_gpu=False,
        )
        state = [0] * 225 + [1]
        result = searcher.search(state)

        assert isinstance(result, tuple)
        visit_counts, root_value, root_policy = result
        assert isinstance(visit_counts, list) and len(visit_counts) == 225
        assert isinstance(root_value, float)
        assert isinstance(root_policy, list) and len(root_policy) == 225
        assert sum(visit_counts) == 10

    @requires_full
    def test_pb04_advance_root_return_type(self):
        """TC-PB-04: advance_root 返回类型"""
        searcher = cpp_mcts.MctsSearcher(
            _MODEL_PATH, num_simulations=10,
            num_threads=1, virtual_loss=0.0, use_gpu=False,
        )
        state = [0] * 225 + [1]
        result = searcher.search(state)
        max_action = max(range(225), key=lambda i: result[0][i])
        ret = searcher.advance_root(max_action)
        assert isinstance(ret, bool)

    @requires_cpp
    def test_pb05_invalid_state_input(self):
        """TC-PB-05: 非法 state 输入"""
        if not _model_available:
            pytest.skip("模型文件不可用")
        searcher = cpp_mcts.MctsSearcher(
            _MODEL_PATH, num_simulations=1,
            num_threads=1, virtual_loss=0.0, use_gpu=False,
        )
        with pytest.raises(RuntimeError):
            searcher.search([1] * 100)
        with pytest.raises(RuntimeError):
            searcher.search([1] * 300)


class TestCppMctsAdapter:
    """Python 适配层测试 (TC-AD-01 至 TC-AD-03)"""

    @requires_full
    def test_ad01_interface_compatibility(self):
        """TC-AD-01: CppMctsAdapter 接口兼容"""
        from cpp.python.cpp_mcts_adapter import CppMctsAdapter

        adapter = CppMctsAdapter(
            _MODEL_PATH, num_simulations=10,
            num_threads=1, virtual_loss=0.0, use_gpu=False,
        )
        state = [0] * 225 + [1]
        visit_counts, root_value, root_policy = adapter.search(state)

        assert isinstance(visit_counts, list) and len(visit_counts) == 225
        assert isinstance(root_value, float)
        assert isinstance(root_policy, list) and len(root_policy) == 225
        assert sum(visit_counts) == 10

    @requires_full
    def test_ad02_log_key_ignored(self):
        """TC-AD-02: log_key 参数忽略无异常"""
        from cpp.python.cpp_mcts_adapter import CppMctsAdapter

        adapter = CppMctsAdapter(
            _MODEL_PATH, num_simulations=10,
            num_threads=1, virtual_loss=0.0, use_gpu=False,
        )
        state = [0] * 225 + [1]
        result = adapter.search(state, log_key="test_search")
        assert result is not None and len(result[0]) == 225

    @requires_full
    def test_ad03_with_policy_value_engine(self):
        """TC-AD-03: 与 PolicyValueAiEngine 集成"""
        from cpp.python.cpp_mcts_adapter import CppMctsAdapter
        from ai.engine.policy_value import PolicyValueAiEngine
        from ai.engine.ai_engine import MoveDecision

        adapter = CppMctsAdapter(
            _MODEL_PATH, num_simulations=10,
            num_threads=1, virtual_loss=0.0, use_gpu=False,
        )
        engine = PolicyValueAiEngine(adapter, temperature=1.0, temp_steps=2)

        board = [[0] * 15 for _ in range(15)]
        move = engine.get_move(board, 1, strategy="argmax")
        assert isinstance(move, MoveDecision)
        assert 0 <= move.x < 15 and 0 <= move.y < 15

        engine.after_move(move.x, move.y)
        board[move.x][move.y] = 1
        move2 = engine.get_move(board, 2, strategy="sampling")
        assert isinstance(move2, MoveDecision)
