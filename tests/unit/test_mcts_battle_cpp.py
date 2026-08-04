"""mcts_battle_cpp.py CLI 脚本单元测试

测试 CLI 参数解析、配置构建和输出格式化函数。
完整的对局流程需要 cpp_mcts 模块，在不可用时跳过。
"""

import sys
import os

_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import pytest

# 检测 cpp_mcts 是否可用
try:
    import cpp_mcts  # noqa: F401
    _CPP_MCTS_AVAILABLE = True
except ImportError:
    _CPP_MCTS_AVAILABLE = False

_TS_FILE_EXISTS = os.path.isfile(
    os.path.join(_proj_root, "src", "ai", "models", "cnn_v11",
                 "gen1", "cnn_v11_torchscript.pt")
)

needs_cpp_mcts = pytest.mark.skipif(
    not (_CPP_MCTS_AVAILABLE and _TS_FILE_EXISTS),
    reason="需要 cpp_mcts 模块和 TorchScript 模型文件"
)


class TestArgumentParsing:
    """CLI 参数解析测试"""

    def test_required_args(self):
        """必需参数 --ai1 --ai2 --games"""
        from src.ai.scripts.mcts_battle_cpp import parse_args

        argv = ["--ai1", "cnn_v11.1", "--ai2", "cnn_v11.2", "--games", "10"]
        args = parse_args(argv)
        assert args.ai1 == "cnn_v11.1"
        assert args.ai2 == "cnn_v11.2"
        assert args.games == 10

    def test_default_values(self):
        """默认参数值"""
        from src.ai.scripts.mcts_battle_cpp import parse_args

        argv = ["--ai1", "cnn_v11.1", "--ai2", "cnn_v11.2", "--games", "5"]
        args = parse_args(argv)
        assert args.simulations == 800
        assert args.c_puct == 2.0
        assert args.temp_steps == 8
        assert args.temperature == 1.0
        assert args.dirichlet_epsilon == 0.25
        assert args.dirichlet_alpha == 0.03

    def test_custom_values(self):
        """自定义超参数"""
        from src.ai.scripts.mcts_battle_cpp import parse_args

        argv = [
            "--ai1", "cnn_v11.1", "--ai2", "cnn_v11.2", "--games", "20",
            "--simulations", "400", "--c-puct", "1.5",
            "--temperature", "0.5", "--temp-steps", "12",
            "--dirichlet-epsilon", "0.3", "--dirichlet-alpha", "0.05",
            "--batch-size", "64", "--virtual-loss", "2.0",
        ]
        args = parse_args(argv)
        assert args.simulations == 400
        assert args.c_puct == 1.5
        assert args.temperature == 0.5
        assert args.temp_steps == 12
        assert args.dirichlet_epsilon == 0.3
        assert args.dirichlet_alpha == 0.05
        assert args.batch_size == 64
        assert args.virtual_loss == 2.0

    def test_games_zero_rejected_by_run_battle(self):
        """--games 0 被 run_battle 拒绝（SystemExit）"""
        from src.ai.scripts.mcts_battle_cpp import parse_args, run_battle

        argv = ["--ai1", "a", "--ai2", "b", "--games", "0"]
        args = parse_args(argv)
        with pytest.raises(SystemExit):
            run_battle(args)

    def test_games_negative_rejected_by_run_battle(self):
        """--games 负数被 run_battle 拒绝（SystemExit）"""
        from src.ai.scripts.mcts_battle_cpp import parse_args, run_battle

        argv = ["--ai1", "a", "--ai2", "b", "--games", "-1"]
        args = parse_args(argv)
        with pytest.raises(SystemExit):
            run_battle(args)


class TestConfigBuilding:
    """配置构建测试"""

    def test_build_config_from_args(self):
        """从 CLI args 构建 MctsSelfPlayConfig"""
        from src.ai.scripts.mcts_battle_cpp import parse_args, _build_config

        argv = [
            "--ai1", "cnn_v11.1", "--ai2", "cnn_v11.2", "--games", "10",
            "--simulations", "200", "--c-puct", "1.5",
            "--temperature", "0.8", "--temp-steps", "10",
            "--batch-size", "32", "--virtual-loss", "4.0",
        ]
        args = parse_args(argv)
        config = _build_config(args)
        assert config.simulations == 200
        assert config.c_puct == 1.5
        assert config.temperature == 0.8
        assert config.temp_steps == 10
        assert config.batch_size == 32
        assert config.virtual_loss == 4.0

    def test_build_config_default_batch_size(self):
        """未指定 batch_size 时使用默认值"""
        from src.ai.scripts.mcts_battle_cpp import parse_args, _build_config

        argv = ["--ai1", "a", "--ai2", "b", "--games", "1"]
        args = parse_args(argv)
        config = _build_config(args)
        assert config.batch_size > 0
        assert config.virtual_loss > 0.0


class TestOutputFormatting:
    """输出格式化测试"""

    def test_print_per_game_results_format(self):
        """每局结果输出格式正确"""
        from cpp.python.cpp_battle_game_result import CppBattleGameResult
        from src.ai.scripts.mcts_battle_cpp import _print_per_game_results

        results = [
            CppBattleGameResult(
                game_index=0, winner=1,
                model1_color=1, model2_color=2,
                model1_won=True, model2_won=False, move_count=45,
            ),
        ]
        # 不应抛出异常
        _print_per_game_results(results, "cnn_v11.1", "cnn_v11.2")

    def test_print_statistics_format(self):
        """统计输出格式正确"""
        from cpp.python.cpp_battle_game_result import CppBattleGameResult
        from src.ai.scripts.mcts_battle_cpp import _print_statistics

        results = [
            CppBattleGameResult(
                game_index=0, winner=1,
                model1_color=1, model2_color=2,
                model1_won=True, model2_won=False, move_count=45,
            ),
            CppBattleGameResult(
                game_index=1, winner=2,
                model1_color=2, model2_color=1,
                model1_won=False, model2_won=True, move_count=52,
            ),
            CppBattleGameResult(
                game_index=2, winner=0,
                model1_color=1, model2_color=2,
                model1_won=False, model2_won=False, move_count=225,
            ),
        ]
        # 不应抛出异常
        _print_statistics("cnn_v11.1", "cnn_v11.2", results)


class TestMainFlow:
    """主流程集成测试（需要 cpp_mcts）"""

    @needs_cpp_mcts
    def test_single_game_completes(self):
        """TC2.1：单局对战正常完成"""
        from src.ai.scripts.mcts_battle_cpp import parse_args, run_battle

        argv = [
            "--ai1", "cnn_v11.1", "--ai2", "cnn_v11.1",
            "--games", "1", "--simulations", "10", "--batch-size", "4",
        ]
        args = parse_args(argv)
        # 不应抛出异常
        run_battle(args)

    @needs_cpp_mcts
    def test_multi_game_completes(self):
        """TC2.2：多局对战正常完成"""
        from src.ai.scripts.mcts_battle_cpp import parse_args, run_battle

        argv = [
            "--ai1", "cnn_v11.1", "--ai2", "cnn_v11.1",
            "--games", "3", "--simulations", "10", "--batch-size", "4",
        ]
        args = parse_args(argv)
        run_battle(args)
