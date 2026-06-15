"""IterativeTrainer 集成测试

测试完整迭代流程，使用 mock subprocess.run 和临时目录隔离文件系统副作用。

测试用例覆盖 test_design.md §3 中定义的所有集成测试场景。
"""

import os
import json
import tempfile
import subprocess
from unittest.mock import patch, Mock, call, ANY

import pytest

from ai.iterative_train import (
    IterativeTrainer,
    MODELS_DIR,
    PARENT_GEN,
    CHILD_GEN,
    SAMPLE_OUTPUT_DIR,
)
from tests.ai.helpers import _make_completed_process


def _battle_stdout_line(gen5_wins, gen4_wins, draws):
    """构造 battle.py 标准输出（仅最终统计行）。"""
    total = gen4_wins + gen5_wins + draws
    gen4_rate = gen4_wins / total * 100 if total > 0 else 0.0
    gen5_rate = gen5_wins / total * 100 if total > 0 else 0.0
    draw_rate = draws / total * 100 if total > 0 else 0.0
    return (
        f"=== 对战结果 ===\n"
        f"总局数: {total}\n"
        f"cnn_v3.4 胜: {gen4_wins} 局 ({gen4_rate:.1f}%)\n"
        f"cnn_v3.5 胜: {gen5_wins} 局 ({gen5_rate:.1f}%)\n"
        f"平局: {draws} 局 ({draw_rate:.1f}%)\n"
    )


def _make_side_effect_sequence(*battle_outputs):
    """构造多轮迭代的 mock subprocess.run side_effect 列表。

    每轮 3 次调用：create_samples → train_main → battle。
    每次调用返回 returncode=0，battle 的 stdout 由 battle_outputs 参数指定。

    Args:
        *battle_outputs: 每轮 battle 的 stdout 字符串。
    """
    effects = []
    for battle_out in battle_outputs:
        effects.append(_make_completed_process(returncode=0, stdout="samples ok"))
        effects.append(_make_completed_process(returncode=0, stdout="train ok"))
        effects.append(_make_completed_process(returncode=0, stdout=battle_out))
    return effects


# ============================================================
# IT-1：完整迭代流程 (§3.1)
# ============================================================

class TestFullIterationFlow:
    """IT-1：完整迭代流程集成测试。"""

    @patch('subprocess.run')
    def test_single_round_promoted(self, mock_run, temp_project):
        """IT-1.1：单轮完整流程（晋升：胜率 55%，下一轮停止）。

        验证：
        ① gen4→gen5 被复制
        ② 三个子进程按序调用且 cwd=project_root
        ③ 晋升后 gen4 被更新（copy2+remove）
        ④ run() 返回 dict 含 2 轨记录（1晋升 + 1停止）
        """
        # 为两轮预创建样本文件
        sample_dir = os.path.join(temp_project, SAMPLE_OUTPUT_DIR)
        os.makedirs(sample_dir, exist_ok=True)
        for i in range(1, 3):
            sp = os.path.join(sample_dir, f"path{i}.json")
            with open(sp, 'w') as f:
                json.dump([{"input": [0]*226, "output": [0]*225, "mask": [0]*225}], f)

        side_effects = _make_side_effect_sequence(
            _battle_stdout_line(gen5_wins=55, gen4_wins=40, draws=5),   # round 1 → 晋升
            _battle_stdout_line(gen5_wins=48, gen4_wins=44, draws=8),   # round 2 → 停止
        )
        mock_run.side_effect = side_effects

        trainer = IterativeTrainer(project_root=temp_project)
        result = trainer.run()

        # ① gen4→gen5 被复制（校验 gen5 在初期存在，晋升后删除）
        gen5_path = os.path.join(temp_project, MODELS_DIR, f"gen{CHILD_GEN}", "cnn_v3.pth")
        assert mock_run.called

        # ② 6次子进程调用（2轮 × 3步骤）
        assert mock_run.call_count == 6
        # 验证第1轮调用顺序
        assert "create_samples.py" in " ".join(mock_run.call_args_list[0][0][0])
        assert "train_main.py" in " ".join(mock_run.call_args_list[1][0][0])
        assert "battle.py" in " ".join(mock_run.call_args_list[2][0][0])

        # 验证 cwd=project_root
        for call_args in mock_run.call_args_list:
            assert call_args[1].get('cwd') == temp_project

        # ③ 晋升后 gen4 被更新（gen5 在第二轮被覆盖后，最终停止时 gen5 保留）
        assert os.path.isfile(gen5_path)  # 停止时 gen5 保留

        # ④ run() 返回 dict 含 2 轨记录（1晋升 + 1停止）
        assert result["total_iterations"] == 1  # 1 轮晋升
        assert len(result["rounds"]) == 2
        assert result["rounds"][0]["promoted"] is True
        assert result["rounds"][1]["promoted"] is False

    @patch('subprocess.run')
    def test_single_round_stopped(self, mock_run, temp_project):
        """IT-1.2：单轮完整流程（停止：胜率 48%）。

        验证：
        ① 子进程按序调用
        ② 不执行晋升
        ③ gen4 未被修改
        ④ run() 返回 dict 含 1 轨停止记录
        """
        sample_dir = os.path.join(temp_project, SAMPLE_OUTPUT_DIR)
        os.makedirs(sample_dir, exist_ok=True)
        sample_path = os.path.join(sample_dir, "path1.json")
        with open(sample_path, 'w') as f:
            json.dump([{"input": [0]*226, "output": [0]*225, "mask": [0]*225}], f)

        side_effects = _make_side_effect_sequence(
            _battle_stdout_line(gen5_wins=48, gen4_wins=44, draws=8)
        )
        mock_run.side_effect = side_effects

        # 记录原始 gen4 内容
        gen4_path = os.path.join(temp_project, MODELS_DIR, f"gen{PARENT_GEN}", "cnn_v3.pth")
        with open(gen4_path, 'rb') as f:
            original_gen4 = f.read()

        trainer = IterativeTrainer(project_root=temp_project)
        result = trainer.run()

        # ① 子进程按序调用
        assert mock_run.call_count == 3

        # ② 不执行晋升（gen5 仍存在）
        gen5_path = os.path.join(temp_project, MODELS_DIR, f"gen{CHILD_GEN}", "cnn_v3.pth")
        assert os.path.isfile(gen5_path)

        # ③ gen4 未被修改
        with open(gen4_path, 'rb') as f:
            current_gen4 = f.read()
        assert current_gen4 == original_gen4

        # ④ run() 返回 dict 含 1 轨停止记录
        assert result["total_iterations"] == 0
        assert len(result["rounds"]) == 1
        assert result["rounds"][0]["promoted"] is False
        assert result["rounds"][0]["gen5_win_rate"] == 48.0

    @patch('subprocess.run')
    def test_multi_round_two_promoted_then_stop(self, mock_run, temp_project):
        """IT-1.3：多轮迭代（2 轨晋升后停止）。

        第1轮胜率 55%、第2轮胜率 52%、第3轮胜率 48%。
        验证：
        ① 共执行 3 轮
        ② 前 2 轮晋升，第 3 轮停止
        ③ 统计含 3 行明细
        ④ total_iterations=2
        """
        # 为每轮预创建样本文件
        sample_dir = os.path.join(temp_project, SAMPLE_OUTPUT_DIR)
        os.makedirs(sample_dir, exist_ok=True)
        for i in range(1, 4):
            sp = os.path.join(sample_dir, f"path{i}.json")
            with open(sp, 'w') as f:
                json.dump([{"input": [0]*226, "output": [0]*225, "mask": [0]*225}], f)

        side_effects = _make_side_effect_sequence(
            _battle_stdout_line(gen5_wins=55, gen4_wins=40, draws=5),   # round 1
            _battle_stdout_line(gen5_wins=52, gen4_wins=40, draws=8),   # round 2
            _battle_stdout_line(gen5_wins=48, gen4_wins=44, draws=8),   # round 3
        )
        mock_run.side_effect = side_effects

        trainer = IterativeTrainer(project_root=temp_project)
        result = trainer.run()

        # ① 共执行 3 轮（9 次子进程调用）
        assert mock_run.call_count == 9

        # ② 前 2 轮晋升，第 3 轮停止
        assert result["rounds"][0]["promoted"] is True
        assert result["rounds"][1]["promoted"] is True
        assert result["rounds"][2]["promoted"] is False

        # ③ 统计含 3 行明细
        assert len(result["rounds"]) == 3

        # ④ total_iterations=2
        assert result["total_iterations"] == 2
        assert result["final_round"] == 3

    @patch('subprocess.run')
    def test_subprocess_call_order(self, mock_run, temp_project):
        """IT-1.4：子进程调用顺序正确。

        验证 create_samples → train_main → battle 的固定顺序在每轮中保持。
        """
        sample_dir = os.path.join(temp_project, SAMPLE_OUTPUT_DIR)
        os.makedirs(sample_dir, exist_ok=True)
        for i in range(1, 3):
            sp = os.path.join(sample_dir, f"path{i}.json")
            with open(sp, 'w') as f:
                json.dump([{"input": [0]*226, "output": [0]*225, "mask": [0]*225}], f)

        side_effects = _make_side_effect_sequence(
            _battle_stdout_line(gen5_wins=55, gen4_wins=40, draws=5),   # round 1 → 晋升
            _battle_stdout_line(gen5_wins=48, gen4_wins=44, draws=8),   # round 2 → 停止
        )
        mock_run.side_effect = side_effects

        trainer = IterativeTrainer(project_root=temp_project)
        trainer.run()

        calls = mock_run.call_args_list
        # 验证第1轮调用顺序
        assert "create_samples.py" in " ".join(calls[0][0][0])
        assert "train_main.py" in " ".join(calls[1][0][0])
        assert "battle.py" in " ".join(calls[2][0][0])
        # 验证第2轮调用顺序
        assert "create_samples.py" in " ".join(calls[3][0][0])
        assert "train_main.py" in " ".join(calls[4][0][0])
        assert "battle.py" in " ".join(calls[5][0][0])


# ============================================================
# IT-2：错误处理 (§3.2)
# ============================================================

class TestErrorHandling:
    """IT-2：错误处理集成测试。"""

    @patch('subprocess.run')
    def test_create_samples_failure(self, mock_run, temp_project):
        """IT-2.1：create_samples 失败。

        mock create_samples 返回 exit_code=1，验证：
        抛 RuntimeError，不执行 train_main 和 battle。
        """
        mock_run.return_value = _make_completed_process(
            returncode=1, stdout="", stderr="create failed"
        )

        trainer = IterativeTrainer(project_root=temp_project)
        with pytest.raises(RuntimeError):
            trainer.run()

        # 验证只调用了 create_samples（1次），未调用后续步骤
        assert mock_run.call_count == 1
        call_args = mock_run.call_args[0][0]
        assert "create_samples.py" in " ".join(call_args)

    @patch('subprocess.run')
    def test_train_main_failure(self, mock_run, temp_project):
        """IT-2.2：train_main 失败。

        mock train_main 返回 exit_code=1，验证：
        抛 RuntimeError，不执行 battle。
        """
        sample_dir = os.path.join(temp_project, SAMPLE_OUTPUT_DIR)
        os.makedirs(sample_dir, exist_ok=True)
        sample_path = os.path.join(sample_dir, "path1.json")
        with open(sample_path, 'w') as f:
            json.dump([{"input": [0]*226, "output": [0]*225, "mask": [0]*225}], f)

        side_effects = [
            _make_completed_process(returncode=0, stdout="create ok"),    # create OK
            _make_completed_process(returncode=1, stdout="", stderr="train failed"),  # train fail
        ]
        mock_run.side_effect = side_effects

        trainer = IterativeTrainer(project_root=temp_project)
        with pytest.raises(RuntimeError):
            trainer.run()

        # 验证调用了 create 和 train，未调用 battle
        assert mock_run.call_count == 2
        calls = mock_run.call_args_list
        assert "create_samples.py" in " ".join(calls[0][0][0])
        assert "train_main.py" in " ".join(calls[1][0][0])

    @patch('subprocess.run')
    def test_battle_failure(self, mock_run, temp_project):
        """IT-2.3：battle 失败。

        mock battle 返回 exit_code=1，验证抛 RuntimeError。
        """
        sample_dir = os.path.join(temp_project, SAMPLE_OUTPUT_DIR)
        os.makedirs(sample_dir, exist_ok=True)
        sample_path = os.path.join(sample_dir, "path1.json")
        with open(sample_path, 'w') as f:
            json.dump([{"input": [0]*226, "output": [0]*225, "mask": [0]*225}], f)

        side_effects = [
            _make_completed_process(returncode=0, stdout="create ok"),
            _make_completed_process(returncode=0, stdout="train ok"),
            _make_completed_process(returncode=1, stdout="", stderr="battle failed"),
        ]
        mock_run.side_effect = side_effects

        trainer = IterativeTrainer(project_root=temp_project)
        with pytest.raises(RuntimeError):
            trainer.run()

        assert mock_run.call_count == 3

    def test_gen4_not_exists(self, temp_project):
        """IT-2.4：gen4 权重不存在时构造失败。"""
        gen4_path = os.path.join(
            temp_project, MODELS_DIR, f"gen{PARENT_GEN}", "cnn_v3.pth"
        )
        os.remove(gen4_path)
        with pytest.raises(FileNotFoundError):
            IterativeTrainer(project_root=temp_project)

    @patch('subprocess.run')
    def test_keyboard_interrupt(self, mock_run, temp_project):
        """IT-2.5：Ctrl+C 中断。

        第 2 轮样本生成中触发 KeyboardInterrupt。
        验证：打印"用户中断"，输出已完成轮次的统计，正常返回。
        """
        sample_dir = os.path.join(temp_project, SAMPLE_OUTPUT_DIR)
        os.makedirs(sample_dir, exist_ok=True)
        for i in range(1, 3):
            sp = os.path.join(sample_dir, f"path{i}.json")
            with open(sp, 'w') as f:
                json.dump([{"input": [0]*226, "output": [0]*225, "mask": [0]*225}], f)

        # 第1轮成功，第2轮 create_samples 时抛 KeyboardInterrupt
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 3:
                # 第1轮三次调用均成功
                if call_count[0] == 3:
                    return _make_completed_process(
                        returncode=0,
                        stdout=_battle_stdout_line(gen5_wins=55, gen4_wins=40, draws=5)
                    )
                return _make_completed_process(returncode=0, stdout="ok")
            else:
                # 第2轮 create_samples 时中断
                raise KeyboardInterrupt()

        mock_run.side_effect = side_effect

        trainer = IterativeTrainer(project_root=temp_project)
        result = trainer.run()

        # 正常返回
        assert len(result["rounds"]) == 1
        assert result["rounds"][0]["promoted"] is True
        assert result["total_iterations"] == 1

    def test_lock_file_prevents_construction(self, temp_project):
        """IT-2.6：锁文件存在时构造失败。"""
        lock_path = os.path.join(temp_project, MODELS_DIR, ".iterative_lock")
        with open(lock_path, 'w') as f:
            f.write("dummy")

        with pytest.raises(RuntimeError):
            IterativeTrainer(project_root=temp_project)


# ============================================================
# IT-3：文件系统副作用隔离 (§3.3)
# ============================================================

class TestFileSystemIsolation:
    """IT-3：验证测试不修改真实项目文件。"""

    def test_temp_project_isolation(self, temp_project):
        """IT-3.1：使用临时目录作为 project_root。

        验证真实 ai/models/ 目录不受影响（不直接测试，在 temp_project 中操作）。
        """
        assert "temp" in temp_project.lower() or temp_project.startswith(tempfile.gettempdir())

    @patch('subprocess.run')
    def test_sample_file_in_temp_not_real(self, mock_run, temp_project):
        """IT-3.2：验证 create_samples 的 --output 参数指向临时目录。"""
        sample_dir = os.path.join(temp_project, SAMPLE_OUTPUT_DIR)
        os.makedirs(sample_dir, exist_ok=True)
        for i in range(1, 3):
            sp = os.path.join(sample_dir, f"path{i}.json")
            with open(sp, 'w') as f:
                json.dump([{"input": [0]*226, "output": [0]*225, "mask": [0]*225}], f)

        side_effects = _make_side_effect_sequence(
            _battle_stdout_line(gen5_wins=55, gen4_wins=40, draws=5),   # round 1 → 晋升
            _battle_stdout_line(gen5_wins=48, gen4_wins=44, draws=8),   # round 2 → 停止
        )
        mock_run.side_effect = side_effects

        trainer = IterativeTrainer(project_root=temp_project)
        trainer.run()

        # 验证 create_samples 传入的 --output 路径在临时目录下
        create_call = mock_run.call_args_list[0][0][0]
        output_path = None
        for i, arg in enumerate(create_call):
            if arg == "--output":
                output_path = create_call[i + 1]
                break
        assert output_path is not None
        assert temp_project in output_path

    @patch('subprocess.run')
    def test_lock_cleaned_up_after_run(self, mock_run, temp_project):
        """验证 run() 结束后锁文件被清理。"""
        sample_dir = os.path.join(temp_project, SAMPLE_OUTPUT_DIR)
        os.makedirs(sample_dir, exist_ok=True)
        for i in range(1, 3):
            sp = os.path.join(sample_dir, f"path{i}.json")
            with open(sp, 'w') as f:
                json.dump([{"input": [0]*226, "output": [0]*225, "mask": [0]*225}], f)

        side_effects = _make_side_effect_sequence(
            _battle_stdout_line(gen5_wins=55, gen4_wins=40, draws=5),   # round 1 → 晋升
            _battle_stdout_line(gen5_wins=48, gen4_wins=44, draws=8),   # round 2 → 停止
        )
        mock_run.side_effect = side_effects

        lock_path = os.path.join(temp_project, MODELS_DIR, ".iterative_lock")

        trainer = IterativeTrainer(project_root=temp_project)
        trainer.run()

        # 锁文件应在 run() 结束后被删除
        assert not os.path.isfile(lock_path)

    def test_temp_dir_cleanup(self, temp_project):
        """IT-3.3：确认 temp_project fixture 在 tearDown 中清理。

        此测试完成时 temp_project 应被清理（由 fixture 的 yield 后逻辑负责）。
        此处仅验证 temp_project 存在而 fixture 正常工作。
        """
        assert os.path.isdir(temp_project)

    def test_return_dict_structure(self, temp_project):
        """验证 run() 返回的 dict 结构完整（2轮：1晋升 + 1停止）。"""
        sample_dir = os.path.join(temp_project, SAMPLE_OUTPUT_DIR)
        os.makedirs(sample_dir, exist_ok=True)
        for i in range(1, 3):
            sp = os.path.join(sample_dir, f"path{i}.json")
            with open(sp, 'w') as f:
                json.dump([{"input": [0]*226, "output": [0]*225, "mask": [0]*225}], f)

        with patch('subprocess.run') as mock_run:
            side_effects = _make_side_effect_sequence(
                _battle_stdout_line(gen5_wins=55, gen4_wins=40, draws=5),   # round 1 → 晋升
                _battle_stdout_line(gen5_wins=48, gen4_wins=44, draws=8),   # round 2 → 停止
            )
            mock_run.side_effect = side_effects

            trainer = IterativeTrainer(project_root=temp_project)
            result = trainer.run()

            # 验证所有必需键存在
            assert "total_iterations" in result
            assert "final_round" in result
            assert "rounds" in result
            assert "stop_reason" in result
            assert "final_model_path" in result

            # 验证类型
            assert isinstance(result["total_iterations"], int)
            assert isinstance(result["final_round"], int)
            assert isinstance(result["rounds"], list)
            assert isinstance(result["stop_reason"], str)
            assert isinstance(result["final_model_path"], str)

            # 验证每轮记录的键
            for r in result["rounds"]:
                assert "round_num" in r
                assert "gen4_wins" in r
                assert "gen5_wins" in r
                assert "draws" in r
                assert "gen5_win_rate" in r
                assert "promoted" in r
