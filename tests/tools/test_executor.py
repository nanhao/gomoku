"""executor 模块测试

测试 build_cli_command、execute（子进程执行）、
discover_round_dir（差集发现）。

覆盖 backend_api_design.md 中 T3-T7 + T5 相关测试点。
"""

import os
import tempfile
import subprocess

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
        assert "--model cnn_v1" in cmd
        assert "--output ai/train_data/samples.json" in cmd

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
        assert "--mode winning_move" in cmd
        assert "--games 100" in cmd
        assert "--temperature 1.0" in cmd
        assert "--temp-moves 12" in cmd
        assert "--temp-side both" in cmd

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
        assert "--temp-seed 42" in cmd

    def test_command_starts_with_python(self):
        """命令以 'python ai/create_samples.py' 开头"""
        params = {"model": "cnn_v1", "output": "test.json"}
        cmd = build_cli_command(params)
        assert cmd.startswith("python ai/create_samples.py")


# ------------------------------------------------------------------
# 子进程执行测试
# ------------------------------------------------------------------

class TestExecute:
    """测试 execute 函数"""

    def test_success_path(self, tmp_path):
        """T3: 子进程成功 → status=completed + returncode=0"""
        # 创建一个简单的 Python 脚本作为"子进程"
        script = os.path.join(tmp_path, "mock_script.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write("import sys; print('OK'); sys.exit(0)\n")

        # 创建临时 ai_data_dir
        ai_data_dir = os.path.join(tmp_path, "ai_data")
        os.makedirs(ai_data_dir)

        params = {"model": "test", "output": "test.json"}
        # 使用 mock 脚本替代真实 create_samples.py
        # 通过修改命令前缀来测试
        result = execute(
            params,
            project_root=str(tmp_path),
            ai_data_dir=ai_data_dir,
            command_override=f"python {script}",
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
        result = execute(
            params,
            project_root=str(tmp_path),
            ai_data_dir=ai_data_dir,
            command_override=f"python {script}",
        )

        assert result["status"] == "error"
        assert result["returncode"] == 1
        assert "Error occurred" in result["output"]
        # error 时 round_dir 和 output_path 为 null
        assert result["round_dir"] is None
        assert result["output_path"] is None

    def test_command_recorded(self, tmp_path):
        """execute 结果包含实际执行的命令"""
        script = os.path.join(tmp_path, "mock_script.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write("import sys; print('OK'); sys.exit(0)\n")

        ai_data_dir = os.path.join(tmp_path, "ai_data")
        os.makedirs(ai_data_dir)

        params = {"model": "test", "output": "test.json"}
        override_cmd = f"python {script}"
        result = execute(
            params,
            project_root=str(tmp_path),
            ai_data_dir=ai_data_dir,
            command_override=override_cmd,
        )

        assert result["command"] == override_cmd


# ------------------------------------------------------------------
# 差集发现 round_dir
# ------------------------------------------------------------------

class TestDiscoverRoundDir:
    """测试 discover_round_dir 函数"""

    def test_single_new_dir(self, tmp_path):
        """差集为单个目录 → 返回该目录路径"""
        ai_data_dir = os.path.join(tmp_path, "ai_data")
        os.makedirs(ai_data_dir)
        # 模拟执行前：空目录
        s_before = set()
        # 执行后：新增一个目录
        os.makedirs(os.path.join(ai_data_dir, "cnn_v1_battle_1"))

        result = discover_round_dir(ai_data_dir, s_before)
        assert result is not None
        assert "cnn_v1_battle_1" in result

    def test_multiple_new_dirs_returns_first(self, tmp_path):
        """差集为多个目录时 → 返回排序后最后一个（最新创建）"""
        ai_data_dir = os.path.join(tmp_path, "ai_data")
        os.makedirs(ai_data_dir)
        s_before = set()

        os.makedirs(os.path.join(ai_data_dir, "model_1"))
        os.makedirs(os.path.join(ai_data_dir, "model_2"))

        result = discover_round_dir(ai_data_dir, s_before)
        assert result is not None
        # 差集不唯一时仍返回一个（选最后排序的）
        assert "model_2" in result

    def test_empty_diff_returns_none(self, tmp_path):
        """差集为空 → 返回 None"""
        ai_data_dir = os.path.join(tmp_path, "ai_data")
        os.makedirs(ai_data_dir)
        # 执行前已有目录，执行后无新增
        os.makedirs(os.path.join(ai_data_dir, "existing_dir"))
        s_before = {"existing_dir"}

        result = discover_round_dir(ai_data_dir, s_before)
        assert result is None

    def test_nonexistent_ai_data_dir(self):
        """ai_data_dir 不存在 → 返回 None"""
        result = discover_round_dir("/nonexistent/path", set())
        assert result is None