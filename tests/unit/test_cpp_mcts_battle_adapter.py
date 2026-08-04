"""CppMctsBattleAdapter 单元测试

测试 C++ MCTS 对战适配器的路径解析、结果转换和错误处理。
需要 cpp_mcts 模块的测试在模块不可用时自动跳过（LibTorch DLL 环境问题）。
"""

import sys
import os

_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import pytest

from cpp.python.cpp_battle_game_result import CppBattleGameResult


# 检测 cpp_mcts 是否可用
try:
    import cpp_mcts  # noqa: F401
    _CPP_MCTS_AVAILABLE = True
except ImportError:
    _CPP_MCTS_AVAILABLE = False


# 检查 TorchScript 文件是否存在
_TS_FILE_EXISTS = os.path.isfile(
    os.path.join(_proj_root, "src", "ai", "models", "cnn_v11",
                 "gen1", "cnn_v11_torchscript.pt")
)

# 标记需要 cpp_mcts + TorchScript 的测试
needs_cpp_mcts = pytest.mark.skipif(
    not (_CPP_MCTS_AVAILABLE and _TS_FILE_EXISTS),
    reason="需要 cpp_mcts 模块和 TorchScript 模型文件"
)


class TestResolveTorchscriptPath:
    """路径解析逻辑测试（不需要 cpp_mcts）"""

    def test_valid_spec_returns_absolute_path(self):
        """TC1.1 部分：有效 spec 返回 TorchScript 绝对路径"""
        from cpp.python.cpp_mcts_battle_adapter import CppMctsBattleAdapter

        path = CppMctsBattleAdapter._resolve_torchscript_path("cnn_v11.1")
        assert os.path.isabs(path)
        assert path.endswith("cnn_v11_torchscript.pt")
        assert os.path.isfile(path)

    def test_invalid_spec_raises_value_error(self):
        """TC1.3：无效 spec（不存在的模型名）抛出 ValueError"""
        from cpp.python.cpp_mcts_battle_adapter import CppMctsBattleAdapter

        with pytest.raises(ValueError):
            CppMctsBattleAdapter._resolve_torchscript_path("nonexistent_model.1")

    def test_nonexistent_gen_raises_file_not_found(self):
        """不存在的代数抛出 FileNotFoundError（模型目录不存在）"""
        from cpp.python.cpp_mcts_battle_adapter import CppMctsBattleAdapter

        with pytest.raises(FileNotFoundError):
            CppMctsBattleAdapter._resolve_torchscript_path("cnn_v11.999")



    def test_torchscript_file_not_found_raises(self):
        """TC1.2：TorchScript 文件不存在抛出 FileNotFoundError

        使用 random 模型（没有 TorchScript 文件）来触发此错误。
        """
        from cpp.python.cpp_mcts_battle_adapter import CppMctsBattleAdapter

        with pytest.raises(FileNotFoundError, match="TorchScript"):
            CppMctsBattleAdapter._resolve_torchscript_path("random")


class TestResultConversion:
    """结果转换逻辑测试（不需要 cpp_mcts）"""

    def test_convert_cpp_result_to_dataclass_model1_black_wins(self):
        """C++ GameResult → CppBattleGameResult：model1 执黑获胜"""
        from cpp.python.cpp_mcts_battle_adapter import CppMctsBattleAdapter

        # 模拟 C++ GameResult
        class MockCppResult:
            game_index = 0
            winner = 1
            model1_color = 1
            model2_color = 2
            move_count = 45

        result = CppMctsBattleAdapter._convert_result(MockCppResult())
        assert isinstance(result, CppBattleGameResult)
        assert result.game_index == 0
        assert result.winner == 1
        assert result.model1_color == 1
        assert result.model2_color == 2
        assert result.model1_won is True
        assert result.model2_won is False
        assert result.move_count == 45

    def test_convert_cpp_result_to_dataclass_draw(self):
        """C++ GameResult → CppBattleGameResult：平局"""
        from cpp.python.cpp_mcts_battle_adapter import CppMctsBattleAdapter

        class MockCppResult:
            game_index = 3
            winner = 0
            model1_color = 1
            model2_color = 2
            move_count = 225

        result = CppMctsBattleAdapter._convert_result(MockCppResult())
        assert result.winner == 0
        assert result.model1_won is False
        assert result.model2_won is False

    def test_convert_cpp_result_to_dataclass_model2_white_wins(self):
        """C++ GameResult → CppBattleGameResult：model2 执白获胜"""
        from cpp.python.cpp_mcts_battle_adapter import CppMctsBattleAdapter

        class MockCppResult:
            game_index = 5
            winner = 2
            model1_color = 1
            model2_color = 2
            move_count = 52

        result = CppMctsBattleAdapter._convert_result(MockCppResult())
        assert result.winner == 2
        assert result.model1_won is False
        assert result.model2_won is True

    def test_convert_cpp_result_to_dataclass_model1_white_wins(self):
        """C++ GameResult → CppBattleGameResult：model1 执白获胜"""
        from cpp.python.cpp_mcts_battle_adapter import CppMctsBattleAdapter

        class MockCppResult:
            game_index = 7
            winner = 2
            model1_color = 2
            model2_color = 1
            move_count = 60

        result = CppMctsBattleAdapter._convert_result(MockCppResult())
        # model1 执白，白胜 → model1_won
        assert result.model1_won is True
        assert result.model2_won is False


class TestCppMctsBattleAdapter:
    """适配器集成测试（需要 cpp_mcts）"""

    @needs_cpp_mcts
    def test_construct_with_valid_specs(self):
        """TC1.1：有效 spec 创建适配器成功"""
        from cpp.python.cpp_mcts_battle_adapter import CppMctsBattleAdapter
        from ai.scripts.mcts_config import MctsSelfPlayConfig

        config = MctsSelfPlayConfig(simulations=10, batch_size=4)
        adapter = CppMctsBattleAdapter(
            "cnn_v11.1", "cnn_v11.1", num_games=1, config=config
        )
        assert adapter is not None

    @needs_cpp_mcts
    def test_start_non_blocking(self):
        """TC1.4：start() 非阻塞，立即返回"""
        import time
        from cpp.python.cpp_mcts_battle_adapter import CppMctsBattleAdapter
        from ai.scripts.mcts_config import MctsSelfPlayConfig

        config = MctsSelfPlayConfig(simulations=10, batch_size=4)
        adapter = CppMctsBattleAdapter(
            "cnn_v11.1", "cnn_v11.1", num_games=1, config=config
        )
        t0 = time.time()
        adapter.start()
        elapsed = time.time() - t0
        assert elapsed < 0.5, f"start() 耗时 {elapsed:.2f}s，应 < 0.5s"

    @needs_cpp_mcts
    def test_get_progress_and_wait_completion(self):
        """TC1.5 + TC1.6：进度查询和等待完成"""
        from cpp.python.cpp_mcts_battle_adapter import CppMctsBattleAdapter
        from ai.scripts.mcts_config import MctsSelfPlayConfig

        config = MctsSelfPlayConfig(simulations=10, batch_size=4)
        adapter = CppMctsBattleAdapter(
            "cnn_v11.1", "cnn_v11.1", num_games=2, config=config
        )
        adapter.start()

        completed, total = adapter.get_progress()
        assert total == 2

        adapter.wait_completion()
        assert adapter.is_finished()

        completed, total = adapter.get_progress()
        assert completed == 2

    @needs_cpp_mcts
    def test_get_results_count(self):
        """TC1.7：结果数量正确"""
        from cpp.python.cpp_mcts_battle_adapter import CppMctsBattleAdapter
        from ai.scripts.mcts_config import MctsSelfPlayConfig

        config = MctsSelfPlayConfig(simulations=10, batch_size=4)
        adapter = CppMctsBattleAdapter(
            "cnn_v11.1", "cnn_v11.1", num_games=3, config=config
        )
        adapter.start()
        adapter.wait_completion()

        results = adapter.get_results()
        assert len(results) == 3
        indices = sorted(r.game_index for r in results)
        assert indices == [0, 1, 2]

    @needs_cpp_mcts
    def test_get_results_field_validity(self):
        """TC1.8：结果字段合法性"""
        from cpp.python.cpp_mcts_battle_adapter import CppMctsBattleAdapter
        from ai.scripts.mcts_config import MctsSelfPlayConfig

        config = MctsSelfPlayConfig(simulations=10, batch_size=4)
        adapter = CppMctsBattleAdapter(
            "cnn_v11.1", "cnn_v11.1", num_games=4, config=config
        )
        adapter.start()
        adapter.wait_completion()

        for r in adapter.get_results():
            assert r.winner in (0, 1, 2)
            assert r.model1_color in (1, 2)
            assert r.model2_color == 3 - r.model1_color
            assert not (r.model1_won and r.model2_won)
            assert r.move_count >= 5

    def test_invalid_spec_raises_value_error(self):
        """TC1.3：无效 spec（不存在的模型）在构造时抛出 ValueError"""
        from cpp.python.cpp_mcts_battle_adapter import CppMctsBattleAdapter
        from ai.scripts.mcts_config import MctsSelfPlayConfig

        config = MctsSelfPlayConfig()
        with pytest.raises(ValueError):
            CppMctsBattleAdapter(
                "nonexistent.1", "cnn_v11.1", num_games=1, config=config
            )

    def test_nonexistent_gen_raises_file_not_found(self):
        """不存在的代数在构造时抛出 FileNotFoundError"""
        from cpp.python.cpp_mcts_battle_adapter import CppMctsBattleAdapter
        from ai.scripts.mcts_config import MctsSelfPlayConfig

        config = MctsSelfPlayConfig()
        with pytest.raises(FileNotFoundError):
            CppMctsBattleAdapter(
                "cnn_v11.999", "cnn_v11.1", num_games=1, config=config
            )

    def test_torchscript_not_found_raises(self):
        """TC1.2：TorchScript 不存在时抛出 FileNotFoundError"""
        from cpp.python.cpp_mcts_battle_adapter import CppMctsBattleAdapter
        from ai.scripts.mcts_config import MctsSelfPlayConfig

        config = MctsSelfPlayConfig()
        with pytest.raises(FileNotFoundError, match="TorchScript"):
            CppMctsBattleAdapter(
                "random", "cnn_v11.1", num_games=1, config=config
            )
