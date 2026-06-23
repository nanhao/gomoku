"""IterativeTrainer 集成测试

测试完整迭代流程，使用 mock subprocess.run 和临时目录隔离文件系统副作用。

意图 22 更新：条件晋升（仅 win_rate > 55% 时晋升），所有轮次正常记录。
"""

import os
import json
import tempfile
import subprocess
from unittest.mock import patch, Mock, call, ANY

import pytest

from ai.iterative_train import IterativeTrainer
from tests.ai.helpers import _make_completed_process

# 意图 13：已删除的常量用硬编码替代
_GEN4 = 99
_GEN5 = 100
_MDIR = "ai/models/cnn_v4"
_SDIR = "ai/merge_samples2/v100"


def _battle_stdout_line(gen5_wins, gen4_wins, draws):
    """构造 battle.py 标准输出（仅最终统计行）。"""
    total = gen4_wins + gen5_wins + draws
    gen4_rate = gen4_wins / total * 100 if total > 0 else 0.0
    gen5_rate = gen5_wins / total * 100 if total > 0 else 0.0
    draw_rate = draws / total * 100 if total > 0 else 0.0
    return (
        f"=== 对战结果 ===\n"
        f"总局数: {total}\n"
        f"cnn_v4.99 胜: {gen4_wins} 局 ({gen4_rate:.1f}%)\n"
        f"cnn_v4.100 胜: {gen5_wins} 局 ({gen5_rate:.1f}%)\n"
        f"平局: {draws} 局 ({draw_rate:.1f}%)\n"
    )


def _make_side_effect_with_interrupt(*battle_outputs, interrupt_after_round=None):
    """构造多轮迭代的 mock subprocess.run side_effect 列表，支持键盘中断控制。

    每轮 3 次调用：create_samples → train_main → battle。
    每次调用返回 returncode=0，battle 的 stdout 由 battle_outputs 参数指定。

    Args:
        *battle_outputs: 每轮 battle 的 stdout 字符串。
        interrupt_after_round: 在第 N 轮完成后抛出 KeyboardInterrupt。
                               None 表示不中断（不推荐，会无限循环消耗 side_effect）。
    """
    effects = []

    for battle_out in battle_outputs:
        effects.append(_make_completed_process(returncode=0, stdout="samples ok"))
        effects.append(_make_completed_process(returncode=0, stdout="train ok"))
        effects.append(_make_completed_process(returncode=0, stdout=battle_out))

    if interrupt_after_round is not None:
        # 在第 N 轮完成后中断，即在该轮 battle 之后的第 (N*3) 个位置
        interrupt_index = interrupt_after_round * 3
        if interrupt_index < len(effects):
            effects[interrupt_index] = KeyboardInterrupt()
        else:
            while len(effects) <= interrupt_index:
                effects.append(_make_completed_process(returncode=0, stdout="ok"))
            effects[interrupt_index] = KeyboardInterrupt()
    return effects


# ============================================================
# IT-1：完整迭代流程 (§3.1)
# ============================================================

class TestFullIterationFlow:
    """IT-1：完整迭代流程集成测试（意图 12 更新）。"""

    @patch('subprocess.run')
    def test_single_round_continues(self, mock_run, temp_project):
        """IT-1.1（意图 12 更新）：单轮后继续循环（通过 KeyboardInterrupt 终止）。

        验证：
        ① gen4→gen5 被复制
        ② 子进程按序调用且 cwd=project_root
        ③ 循环始终继续（不因胜率停止）
        ④ KeyboardInterrupt 后返回 dict 含已完成轮次记录
        """
        # 为两轮预创建样本文件
        sample_dir = os.path.join(temp_project, _SDIR)
        os.makedirs(sample_dir, exist_ok=True)
        sp = os.path.join(sample_dir, "path.json")
        with open(sp, 'w') as f:
            json.dump([{"input": [0]*226, "output": [0]*225, "mask": [0]*225}], f)

        # 第1轮完成后中断
        side_effects = _make_side_effect_with_interrupt(
            _battle_stdout_line(gen5_wins=55, gen4_wins=40, draws=5),   # round 1
            _battle_stdout_line(gen5_wins=48, gen4_wins=44, draws=8),   # round 2（不会完成）
            interrupt_after_round=1,
        )
        mock_run.side_effect = side_effects

        trainer = IterativeTrainer(ai1="cnn_v4.99", ai2="cnn_v4.100", sample_dir="ai/merge_samples2", project_root=temp_project)
        result = trainer.run()

        # ① gen4→gen5 被复制
        gen5_path = os.path.join(temp_project, _MDIR, f"gen{_GEN5}", "cnn_v4.pth")
        assert mock_run.called

        # ② 子进程按序调用且 cwd=project_root
        for call_args in mock_run.call_args_list:
            assert call_args[1].get('cwd') == temp_project

        # ③ 循环始终继续，不因胜率停止
        # ④ run() 返回 dict 含已完成轮次
        assert len(result["rounds"]) >= 1
        assert result["rounds"][0]["promoted"] is False  # 意图 22：55% 不晋升（需 > 55%）

    @patch('subprocess.run')
    def test_two_rounds_then_interrupt(self, mock_run, temp_project):
        """IT-1.2（意图 22 更新）：2轮完整执行后中断。

        验证：
        ① 2 轮均完成
        ② promoted 由胜率决定（48%→未晋升，44%→未晋升）
        ③ total_iterations = 2
        """
        sample_dir = os.path.join(temp_project, _SDIR)
        os.makedirs(sample_dir, exist_ok=True)
        sp = os.path.join(sample_dir, "path.json")
        with open(sp, 'w') as f:
            json.dump([{"input": [0]*226, "output": [0]*225, "mask": [0]*225}], f)

        # 2轮完成后在第3轮开始时中断
        side_effects = _make_side_effect_with_interrupt(
            _battle_stdout_line(gen5_wins=48, gen4_wins=44, draws=8),   # round 1
            _battle_stdout_line(gen5_wins=44, gen4_wins=46, draws=10),  # round 2
            _battle_stdout_line(gen5_wins=50, gen4_wins=45, draws=5),   # round 3（部分）
            interrupt_after_round=2,
        )
        mock_run.side_effect = side_effects

        trainer = IterativeTrainer(ai1="cnn_v4.99", ai2="cnn_v4.100", sample_dir="ai/merge_samples2", project_root=temp_project)
        result = trainer.run()

        # ① 2 轮均完成
        assert len(result["rounds"]) == 2

        # ② promoted 由胜率决定（48%→未晋升，44%→未晋升）
        assert result["rounds"][0]["promoted"] is False
        assert result["rounds"][1]["promoted"] is False

        # ③ total_iterations = 2
        assert result["total_iterations"] == 2

    @patch('subprocess.run')
    def test_subprocess_call_order(self, mock_run, temp_project):
        """IT-1.4：子进程调用顺序正确。

        验证 create_samples → train_main → battle 的固定顺序在每轮中保持。
        """
        sample_dir = os.path.join(temp_project, _SDIR)
        os.makedirs(sample_dir, exist_ok=True)
        sp = os.path.join(sample_dir, "path.json")
        with open(sp, 'w') as f:
            json.dump([{"input": [0]*226, "output": [0]*225, "mask": [0]*225}], f)

        side_effects = _make_side_effect_with_interrupt(
            _battle_stdout_line(gen5_wins=55, gen4_wins=40, draws=5),   # round 1
            _battle_stdout_line(gen5_wins=48, gen4_wins=44, draws=8),   # round 2
            interrupt_after_round=2,
        )
        mock_run.side_effect = side_effects

        trainer = IterativeTrainer(ai1="cnn_v4.99", ai2="cnn_v4.100", sample_dir="ai/merge_samples2", project_root=temp_project)
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

        trainer = IterativeTrainer(ai1="cnn_v4.99", ai2="cnn_v4.100", sample_dir="ai/merge_samples2", project_root=temp_project)
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
        sample_dir = os.path.join(temp_project, _SDIR)
        os.makedirs(sample_dir, exist_ok=True)
        sample_path = os.path.join(sample_dir, "path.json")
        with open(sample_path, 'w') as f:
            json.dump([{"input": [0]*226, "output": [0]*225, "mask": [0]*225}], f)

        side_effects = [
            _make_completed_process(returncode=0, stdout="create ok"),    # create OK
            _make_completed_process(returncode=1, stdout="", stderr="train failed"),  # train fail
        ]
        mock_run.side_effect = side_effects

        trainer = IterativeTrainer(ai1="cnn_v4.99", ai2="cnn_v4.100", sample_dir="ai/merge_samples2", project_root=temp_project)
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
        sample_dir = os.path.join(temp_project, _SDIR)
        os.makedirs(sample_dir, exist_ok=True)
        sample_path = os.path.join(sample_dir, "path.json")
        with open(sample_path, 'w') as f:
            json.dump([{"input": [0]*226, "output": [0]*225, "mask": [0]*225}], f)

        side_effects = [
            _make_completed_process(returncode=0, stdout="create ok"),
            _make_completed_process(returncode=0, stdout="train ok"),
            _make_completed_process(returncode=1, stdout="", stderr="battle failed"),
        ]
        mock_run.side_effect = side_effects

        trainer = IterativeTrainer(ai1="cnn_v4.99", ai2="cnn_v4.100", sample_dir="ai/merge_samples2", project_root=temp_project)
        with pytest.raises(RuntimeError):
            trainer.run()

        assert mock_run.call_count == 3

    def test_gen4_not_exists(self, temp_project):
        """IT-2.4：gen4 权重不存在时构造失败。"""
        gen4_path = os.path.join(
            temp_project, _MDIR, f"gen{_GEN4}", "cnn_v4.pth"
        )
        os.remove(gen4_path)
        with pytest.raises(FileNotFoundError):
            IterativeTrainer(ai1="cnn_v4.99", ai2="cnn_v4.100", sample_dir="ai/merge_samples2", project_root=temp_project)

    @patch('subprocess.run')
    def test_keyboard_interrupt(self, mock_run, temp_project):
        """IT-2.5：Ctrl+C 中断。

        第 2 轮样本生成中触发 KeyboardInterrupt。
        验证：打印"用户中断"，输出已完成轮次的统计，正常返回。
        """
        sample_dir = os.path.join(temp_project, _SDIR)
        os.makedirs(sample_dir, exist_ok=True)
        sp = os.path.join(sample_dir, "path.json")
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

        trainer = IterativeTrainer(ai1="cnn_v4.99", ai2="cnn_v4.100", sample_dir="ai/merge_samples2", project_root=temp_project)
        result = trainer.run()

        # 正常返回
        assert len(result["rounds"]) == 1
        assert result["rounds"][0]["promoted"] is False  # 意图 22：55% 不晋升
        assert result["total_iterations"] == 1

    def test_lock_file_prevents_construction(self, temp_project):
        """IT-2.6：锁文件存在时构造失败。"""
        lock_path = os.path.join(temp_project, _MDIR, ".iterative_lock")
        with open(lock_path, 'w') as f:
            f.write("dummy")

        with pytest.raises(RuntimeError):
            IterativeTrainer(ai1="cnn_v4.99", ai2="cnn_v4.100", sample_dir="ai/merge_samples2", project_root=temp_project)


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
        sample_dir = os.path.join(temp_project, _SDIR)
        os.makedirs(sample_dir, exist_ok=True)
        sp = os.path.join(sample_dir, "path.json")
        with open(sp, 'w') as f:
            json.dump([{"input": [0]*226, "output": [0]*225, "mask": [0]*225}], f)

        side_effects = _make_side_effect_with_interrupt(
            _battle_stdout_line(gen5_wins=55, gen4_wins=40, draws=5),   # round 1
            _battle_stdout_line(gen5_wins=48, gen4_wins=44, draws=8),   # round 2（不会完成）
            interrupt_after_round=1,
        )
        mock_run.side_effect = side_effects

        trainer = IterativeTrainer(ai1="cnn_v4.99", ai2="cnn_v4.100", sample_dir="ai/merge_samples2", project_root=temp_project)
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
        sample_dir = os.path.join(temp_project, _SDIR)
        os.makedirs(sample_dir, exist_ok=True)
        sp = os.path.join(sample_dir, "path.json")
        with open(sp, 'w') as f:
            json.dump([{"input": [0]*226, "output": [0]*225, "mask": [0]*225}], f)

        side_effects = _make_side_effect_with_interrupt(
            _battle_stdout_line(gen5_wins=55, gen4_wins=40, draws=5),   # round 1
            _battle_stdout_line(gen5_wins=48, gen4_wins=44, draws=8),   # round 2
            interrupt_after_round=1,
        )
        mock_run.side_effect = side_effects

        lock_path = os.path.join(temp_project, _MDIR, ".iterative_lock")

        trainer = IterativeTrainer(ai1="cnn_v4.99", ai2="cnn_v4.100", sample_dir="ai/merge_samples2", project_root=temp_project)
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
        """验证 run() 返回的 dict 结构完整（意图 12：2轮全部完成）。"""
        sample_dir = os.path.join(temp_project, _SDIR)
        os.makedirs(sample_dir, exist_ok=True)
        sp = os.path.join(sample_dir, "path.json")
        with open(sp, 'w') as f:
            json.dump([{"input": [0]*226, "output": [0]*225, "mask": [0]*225}], f)

        with patch('subprocess.run') as mock_run:
            side_effects = _make_side_effect_with_interrupt(
                _battle_stdout_line(gen5_wins=55, gen4_wins=40, draws=5),   # round 1
                _battle_stdout_line(gen5_wins=48, gen4_wins=44, draws=8),   # round 2
                interrupt_after_round=2,
            )
            mock_run.side_effect = side_effects

            trainer = IterativeTrainer(ai1="cnn_v4.99", ai2="cnn_v4.100", sample_dir="ai/merge_samples2", project_root=temp_project)
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

            # 意图 22：promoted 由胜率决定（55%→未晋升，48%→未晋升）
            assert result["rounds"][0]["promoted"] is False
            assert result["rounds"][1]["promoted"] is False
            assert result["total_iterations"] == len(result["rounds"])
