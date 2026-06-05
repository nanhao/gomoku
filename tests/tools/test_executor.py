"""executor 模块测试

测试 build_cli_command、execute（子进程执行）、
discover_round_dir（差集发现）。

覆盖 backend_api_design.md 中 T3-T7 + T5 相关测试点。
"""

import os
import tempfile
import subprocess
from unittest import mock

import pytest

from tools.create_samples_tool.executor import (
    build_cli_command,
    execute,
    discover_round_dir,
)


# ------------------------------------------------------------------
# T5: CLI 命令构造
# ------------------------------------------------------------------

class TestBuildCliCommand:
    """测试 CLI 命令构造"""

    def test_basic_command(self):
        """必填参数始终追加"""
        params = {"model": "cnn_v1", "output": "ai/train_data/samples.json"}
        cmd = build_cli_command(params)
        assert "--model" in cmd
        assert "cnn_v1" in cmd
        assert "--output" in cmd
        assert "ai/train_data/samples.json" in cmd

    def test_default_values(self):
        """默认值参数追加"""
        params = {
            "model": "cnn_v1",
            "output": "test.json",
            "mode": "winning_move",
            "games": 100,
            "temperature": 1.0,
            "temp_moves": 12,
            "temp_side": "both",
        }
        cmd = build_cli_command(params)
        assert "--mode" in cmd and "winning_move" in cmd
        assert "--games" in cmd and "100" in cmd
        assert "--temperature" in cmd and "1.0" in cmd
        assert "--temp-moves" in cmd and "12" in cmd
        assert "--temp-side" in cmd and "both" in cmd

    def test_boolean_flags_true_only(self):
        """布尔开关仅 true 时追加 flag"""
        params = {
            "model": "cnn_v1",
            "output": "test.json",
            "no_augment": True,
            "no_temp_sampling": True,
        }
        cmd = build_cli_command(params)
        assert "--no-augment" in cmd
        assert "--no-temp-sampling" in cmd

    def test_boolean_flags_false_not_added(self):
        """布尔开关 false 时不追加"""
        params = {
            "model": "cnn_v1",
            "output": "test.json",
            "no_augment": False,
            "no_temp_sampling": False,
        }
        cmd = build_cli_command(params)
        assert "--no-augment" not in cmd
        assert "--no-temp-sampling" not in cmd

    def test_temp_seed_null_not_added(self):
        """temp_seed 为 null 时不追加"""
        params = {
            "model": "cnn_v1",
            "output": "test.json",
            "temp_seed": None,
        }
        cmd = build_cli_command(params)
        assert "--temp-seed" not in cmd

    def test_temp_seed_value_added(self):
        """temp_seed 非 null 时追加"""
        params = {
            "model": "cnn_v1",
            "output": "test.json",
            "temp_seed": 42,
        }
        cmd = build_cli_command(params)
        assert "--temp-seed" in cmd
        assert "42" in cmd

    def test_command_starts_with_python(self):
        """命令以 'python ai/create_samples.py' 开头"""
        params = {"model": "cnn_v1", "output": "test.json"}
        cmd = build_cli_command(params)
        assert cmd[0] == "python"
        assert cmd[1] == "ai/create_samples.py"

    def test_returns_list_not_string(self):
        """返回值为 list，用于 shell=False 安全调用"""
        params = {"model": "cnn_v1", "output": "test.json"}
        cmd = build_cli_command(params)
        assert isinstance(cmd, list)


# ------------------------------------------------------------------
# 子进程执行测试
# ------------------------------------------------------------------

class TestExecute:
    """测试 execute 函数"""

    def test_success_path(self, tmp_path):
        """T3: 子进程成功 → status=completed + returncode=0"""
        # 创建一个简单的 Python 脚本作为 mock 子进程
        script = os.path.join(tmp_path, "mock_script.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write("import sys; print('OK'); sys.exit(0)\n")

        ai_data_dir = os.path.join(tmp_path, "ai_data")
        os.makedirs(ai_data_dir)

        params = {"model": "test", "output": "test.json"}
        mock_cmd = ["python", script]

        with mock.patch(
            "tools.create_samples_tool.executor.build_cli_command",
            return_value=mock_cmd,
        ):
            result = execute(
                params,
                project_root=str(tmp_path),
                ai_data_dir=ai_data_dir,
            )

        assert result["status"] == "completed"
        assert result["returncode"] == 0
        assert "OK" in result["output"]

    def test_error_path(self, tmp_path):
        """T4: 子进程失败 → status=error + returncode!=0"""
        script = os.path.join(tmp_path, "fail_script.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write("import sys; print('Error occurred'); sys.exit(1)\n")

        ai_data_dir = os.path.join(tmp_path, "ai_data")
        os.makedirs(ai_data_dir)

        params = {"model": "bad_model", "output": "test.json"}
        mock_cmd = ["python", script]

        with mock.patch(
            "tools.create_samples_tool.executor.build_cli_command",
            return_value=mock_cmd,
        ):
            result = execute(
                params,
                project_root=str(tmp_path),
                ai_data_dir=ai_data_dir,
            )

        assert result["status"] == "error"
        assert result["returncode"] == 1
        assert "Error occurred" in result["output"]
        assert result["round_dir"] is None
        assert result["output_path"] is None

    def test_command_recorded(self, tmp_path):
        """execute 结果包含实际执行的命令（展示用字符串）"""
        script = os.path.join(tmp_path, "mock_script.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write("import sys; print('OK'); sys.exit(0)\n")

        ai_data_dir = os.path.join(tmp_path, "ai_data")
        os.makedirs(ai_data_dir)

        params = {"model": "test", "output": "test.json"}
        mock_cmd = ["python", script]

        with mock.patch(
            "tools.create_samples_tool.executor.build_cli_command",
            return_value=mock_cmd,
        ):
            result = execute(
                params,
                project_root=str(tmp_path),
                ai_data_dir=ai_data_dir,
            )

        assert result["command"] == f"python {script}"


# ------------------------------------------------------------------
# 差集发现 round_dir
# ------------------------------------------------------------------

class TestDiscoverRoundDir:
    """测试 discover_round_dir 函数"""

    def test_single_new_dir(self, tmp_path):
        """差集为单个目录 → 返回该目录路径"""
        ai_data_dir = os.path.join(tmp_path, "ai_data")
        os.makedirs(ai_data_dir)
        s_before = set()
        os.makedirs(os.path.join(ai_data_dir, "cnn_v1_battle_1"))

        result = discover_round_dir(ai_data_dir, s_before, project_root=str(tmp_path))
        assert result is not None
        assert "cnn_v1_battle_1" in result

    def test_multiple_new_dirs_returns_one(self, tmp_path):
        """差集为多个目录时 → 返回排序后最后一个"""
        ai_data_dir = os.path.join(tmp_path, "ai_data")
        os.makedirs(ai_data_dir)
        s_before = set()

        os.makedirs(os.path.join(ai_data_dir, "model_1"))
        os.makedirs(os.path.join(ai_data_dir, "model_2"))

        result = discover_round_dir(ai_data_dir, s_before, project_root=str(tmp_path))
        assert result is not None
        assert "model_2" in result

    def test_empty_diff_returns_none(self, tmp_path):
        """差集为空 → 返回 None"""
        ai_data_dir = os.path.join(tmp_path, "ai_data")
        os.makedirs(ai_data_dir)
        os.makedirs(os.path.join(ai_data_dir, "existing_dir"))
        s_before = {"existing_dir"}

        result = discover_round_dir(ai_data_dir, s_before)
        assert result is None

    def test_nonexistent_ai_data_dir(self):
        """ai_data_dir 不存在 → 返回 None"""
        result = discover_round_dir("/nonexistent/path", set())
        assert result is None


# ------------------------------------------------------------------
# H: build_cli_command sample_games 互斥逻辑
# ------------------------------------------------------------------

class TestBuildCliCommandSampleGames:
    """测试 build_cli_command 的 --sample_games / --games 互斥逻辑

    覆盖设计文档 test_cases.md 中 H-01~H-05。
    """

    def test_h01_sample_games_has_value_games_skipped(self):
        """H-01: sample_games 有值时，命令含 --sample_games，不含 --games

        即使 params 中同时包含 games 和 sample_games（前端绕过场景），
        后端也必须保证互斥：sample_games 优先。
        """
        params = {
            "model": "cnn_v1",
            "output": "test.json",
            "sample_games": 10,
            "games": 100,
        }
        cmd = build_cli_command(params)
        assert "--sample_games" in cmd
        assert "10" in cmd
        # games 应被跳过（互斥兜底）
        assert "--games" not in cmd

    def test_h02_only_games_has_value(self):
        """H-02: 仅 games 有值时，命令含 --games，不含 --sample_games"""
        params = {
            "model": "cnn_v1",
            "output": "test.json",
            "games": 100,
        }
        cmd = build_cli_command(params)
        assert "--games" in cmd
        assert "100" in cmd
        assert "--sample_games" not in cmd

    def test_h03_neither_has_value(self):
        """H-03: 两者均无值时，命令不含 --games 和 --sample_games

        CLI 侧默认 --games 100，此处仅验证 executor 不追加。
        """
        params = {
            "model": "cnn_v1",
            "output": "test.json",
        }
        cmd = build_cli_command(params)
        assert "--games" not in cmd
        assert "--sample_games" not in cmd

    def test_h04_sample_games_null_falls_back_to_games(self):
        """H-04: sample_games 为 None 时，回退到 games

        collectParams() 传 sample_games: null（前端 sampleGamesInput 为空），
        后端应正确跳过 sample_games，使用 games。
        """
        params = {
            "model": "cnn_v1",
            "output": "test.json",
            "sample_games": None,
            "games": 50,
        }
        cmd = build_cli_command(params)
        assert "--games" in cmd
        assert "50" in cmd
        assert "--sample_games" not in cmd

    def test_h05_sample_games_zero_value_added(self):
        """H-05: sample_games 为 0 时，追加 --sample_games 0

        0 是合法值（is not None），由 CLI 侧校验值必须大于 0。
        """
        params = {
            "model": "cnn_v1",
            "output": "test.json",
            "sample_games": 0,
        }
        cmd = build_cli_command(params)
        assert "--sample_games" in cmd
        assert "0" in cmd
        assert "--games" not in cmd

    def test_both_null_neither_added(self):
        """补充: games 和 sample_games 均为 None 时，两者都不追加"""
        params = {
            "model": "cnn_v1",
            "output": "test.json",
            "games": None,
            "sample_games": None,
        }
        cmd = build_cli_command(params)
        assert "--games" not in cmd
        assert "--sample_games" not in cmd
