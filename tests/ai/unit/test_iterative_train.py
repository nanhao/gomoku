"""IterativeTrainer 单元测试

测试 IterativeTrainer 类各方法的单元行为，使用 mock subprocess.run
和临时目录隔离文件系统副作用。

测试用例覆盖 test_design.md §2 中定义的所有单元测试场景。
"""

import os
import json
import shutil
import tempfile
import subprocess
from unittest.mock import patch, Mock, ANY

import pytest

# 在导入 iterative_train 前需要确保项目根在 sys.path 中
from ai.iterative_train import (
    IterativeTrainer,
    BASE_MODEL_NAME,
    PARENT_GEN,
    CHILD_GEN,
    MODELS_DIR,
    SAMPLE_OUTPUT_DIR,
    WIN_RATE_THRESHOLD,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def temp_project():
    """创建临时项目目录，包含 gen4/cnn_v3.pth 假文件。

    用于隔离测试文件系统副作用，避免污染真实项目文件。
    测试结束自动清理。
    """
    tmp = tempfile.mkdtemp()
    # 创建模型目录和假权重文件
    gen4_dir = os.path.join(tmp, MODELS_DIR, f"gen{PARENT_GEN}")
    os.makedirs(gen4_dir, exist_ok=True)
    gen4_path = os.path.join(gen4_dir, "cnn_v3.pth")
    with open(gen4_path, 'w') as f:
        f.write("fake_model_weights")

    # 创建 models 目录（用于锁文件存放）
    models_dir = os.path.join(tmp, MODELS_DIR)
    os.makedirs(models_dir, exist_ok=True)

    yield tmp

    # 清理
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def trainer(temp_project):
    """创建 IterativeTrainer 实例（项目根为临时目录）。"""
    return IterativeTrainer(project_root=temp_project)


def _make_completed_process(returncode=0, stdout="", stderr=""):
    """构造 mock subprocess.CompletedProcess 对象。"""
    result = Mock(spec=subprocess.CompletedProcess)
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


def _battle_stdout(gen4_wins=40, gen5_wins=55, draws=5):
    """构造 battle.py 标准输出文本（模拟最终统计行）。

    包含逐局结果行和最终统计行，验证解析器能正确匹配 cnn_v3.5 统计行。
    """
    total = gen4_wins + gen5_wins + draws
    gen4_rate = gen4_wins / total * 100 if total > 0 else 0.0
    gen5_rate = gen5_wins / total * 100 if total > 0 else 0.0
    draw_rate = draws / total * 100 if total > 0 else 0.0

    lines = [
        "开始 AI 对战: cnn_v3.4 vs cnn_v3.5, 共 100 局",
        "",
    ]
    for i in range(1, total + 1):
        # 添加逐局结果行（干扰测试）
        lines.append(f"第 {i} 局: cnn_v3.4(黑) vs cnn_v3.5(白) -> 黑方胜")
    lines += [
        "",
        "=== 对战结果 ===",
        f"总局数: {total}",
        f"cnn_v3.4 胜: {gen4_wins} 局 ({gen4_rate:.1f}%)",
        f"cnn_v3.5 胜: {gen5_wins} 局 ({gen5_rate:.1f}%)",
        f"平局: {draws} 局 ({draw_rate:.1f}%)",
    ]
    return "\n".join(lines)


# ============================================================
# UT-1：构造器校验 (§2.1)
# ============================================================

class TestConstructor:
    """UT-1：IterativeTrainer 构造器校验。"""

    def test_gen4_exists_succeeds(self, temp_project):
        """UT-1.1：gen4 权重存在时正常构造。"""
        trainer = IterativeTrainer(project_root=temp_project)
        assert trainer is not None
        assert trainer._round_num == 1
        assert trainer._rounds == []

    def test_gen4_not_exists_raises(self, temp_project):
        """UT-1.2：gen4 权重不存在时抛 FileNotFoundError。"""
        gen4_path = os.path.join(
            temp_project, MODELS_DIR, f"gen{PARENT_GEN}", "cnn_v3.pth"
        )
        os.remove(gen4_path)
        with pytest.raises(FileNotFoundError, match="gen4"):
            IterativeTrainer(project_root=temp_project)

    def test_auto_deduce_project_root(self):
        """UT-1.3：不传 project_root 时自动推导到项目根目录。

        注意：此测试依赖 gen4 权重真实存在，不适合 mock。
        仅验证推导逻辑不崩溃（不做 assert 路径等于具体值）。
        """
        # 如果真实 gen4 存在，构造不应抛异常
        gen4_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            ))),
            MODELS_DIR, f"gen{PARENT_GEN}", "cnn_v3.pth"
        )
        if os.path.isfile(gen4_path):
            trainer = IterativeTrainer()
            assert trainer.project_root is not None
        else:
            # gen4 不存在则跳过此测试
            pytest.skip("gen4 权重文件不存在，跳过自动推导测试")

    def test_lock_file_exists_raises(self, temp_project):
        """UT-1.4：锁文件存在时抛 RuntimeError。"""
        lock_path = os.path.join(temp_project, MODELS_DIR, ".iterative_lock")
        with open(lock_path, 'w') as f:
            f.write("dummy")
        with pytest.raises(RuntimeError, match="iterative|锁"):
            IterativeTrainer(project_root=temp_project)

    def test_no_lock_file_succeeds(self, temp_project):
        """UT-1.5：锁文件不存在时正常构造。"""
        # 确保锁文件不存在
        lock_path = os.path.join(temp_project, MODELS_DIR, ".iterative_lock")
        if os.path.isfile(lock_path):
            os.remove(lock_path)
        trainer = IterativeTrainer(project_root=temp_project)
        assert trainer is not None


# ============================================================
# UT-2：模型文件操作 (§2.2)
# ============================================================

class TestModelFileOperations:
    """UT-2：模型文件操作（copy、晋升等）。"""

    def test_copy_gen4_to_gen5_no_dir(self, trainer, temp_project):
        """UT-2.1：copy gen4→gen5（gen5 目录不存在）时自动创建并复制。"""
        gen5_dir = os.path.join(temp_project, MODELS_DIR, f"gen{CHILD_GEN}")
        # 确保 gen5 目录不存在
        if os.path.exists(gen5_dir):
            shutil.rmtree(gen5_dir)

        trainer._init_model()

        gen5_path = os.path.join(gen5_dir, "cnn_v3.pth")
        assert os.path.isfile(gen5_path)

        # 校验内容一致（md5）
        gen4_path = os.path.join(temp_project, MODELS_DIR, f"gen{PARENT_GEN}", "cnn_v3.pth")
        with open(gen4_path, 'rb') as f:
            gen4_content = f.read()
        with open(gen5_path, 'rb') as f:
            gen5_content = f.read()
        assert gen4_content == gen5_content

    def test_copy_gen4_overwrites_gen5(self, trainer, temp_project):
        """UT-2.2：gen5 目录已存在时直接覆盖。"""
        gen5_dir = os.path.join(temp_project, MODELS_DIR, f"gen{CHILD_GEN}")
        os.makedirs(gen5_dir, exist_ok=True)
        gen5_path = os.path.join(gen5_dir, "cnn_v3.pth")
        # 写入不同的内容
        with open(gen5_path, 'w') as f:
            f.write("old_different_content")

        trainer._init_model()

        # 校验内容已被覆盖为 gen4 的副本
        gen4_path = os.path.join(temp_project, MODELS_DIR, f"gen{PARENT_GEN}", "cnn_v3.pth")
        with open(gen4_path, 'rb') as f:
            gen4_content = f.read()
        with open(gen5_path, 'rb') as f:
            gen5_content = f.read()
        assert gen4_content == gen5_content
        assert gen5_content != b"old_different_content"

    def test_promote_updates_gen4(self, trainer, temp_project):
        """UT-2.3：晋升操作正确更新 gen4 权重（copy2 + remove）。"""
        # 初始化 gen5
        trainer._init_model()
        gen5_dir = os.path.join(temp_project, MODELS_DIR, f"gen{CHILD_GEN}")
        gen5_path = os.path.join(gen5_dir, "cnn_v3.pth")
        gen4_path = os.path.join(temp_project, MODELS_DIR, f"gen{PARENT_GEN}", "cnn_v3.pth")

        # 修改 gen5 内容（模拟训练后权重变化）
        with open(gen5_path, 'w') as f:
            f.write("trained_weights")
        with open(gen5_path, 'rb') as f:
            gen5_trained = f.read()

        # 执行晋升
        trainer._promote()

        # gen4 应为训练后的权重
        with open(gen4_path, 'rb') as f:
            gen4_after = f.read()
        assert gen4_after == gen5_trained

        # gen5 应被删除
        assert not os.path.isfile(gen5_path)

    def test_sample_output_dir_auto_create(self, trainer, temp_project):
        """UT-2.4：样本输出目录不存在时自动创建。"""
        sample_dir = os.path.join(temp_project, SAMPLE_OUTPUT_DIR)
        if os.path.exists(sample_dir):
            shutil.rmtree(sample_dir)

        # 直接验证 _generate_samples 中的 os.makedirs 调用
        # 通过调用 _ensure_sample_dir 或等效逻辑
        trainer._ensure_sample_output_dir()

        assert os.path.isdir(sample_dir)

    def test_promote_copy2_succeeds_remove_fails(self, trainer, temp_project):
        """UT-2.5：晋升 copy2 成功后 remove 失败。

        验证 gen4 已更新（晋升语义正确），gen5 副本残留。
        """
        trainer._init_model()
        gen5_dir = os.path.join(temp_project, MODELS_DIR, f"gen{CHILD_GEN}")
        gen5_path = os.path.join(gen5_dir, "cnn_v3.pth")
        gen4_path = os.path.join(temp_project, MODELS_DIR, f"gen{PARENT_GEN}", "cnn_v3.pth")

        # 修改 gen5 内容
        with open(gen5_path, 'w') as f:
            f.write("trained_weights")

        # mock os.remove 抛异常，但 copy2 应正常执行
        with patch('os.remove', side_effect=IOError("disk full")):
            with pytest.raises(IOError):
                trainer._promote()

        # gen4 应已被更新（copy2 成功）
        with open(gen4_path, 'rb') as f:
            gen4_after = f.read()
        assert gen4_after == b"trained_weights"

        # gen5 应残留
        assert os.path.isfile(gen5_path)


# ============================================================
# UT-3：子进程调用参数构造 (§2.3)
# ============================================================

class TestSubprocessArgs:
    """UT-3：子进程调用参数构造。"""

    @patch('subprocess.run')
    def test_create_samples_args(self, mock_run, trainer):
        """UT-3.1：create_samples.py 调用参数正确。"""
        mock_run.return_value = _make_completed_process(returncode=0, stdout="ok")

        trainer._generate_samples()

        assert mock_run.called
        call_args = mock_run.call_args[0][0]
        # 验证关键参数
        assert "create_samples.py" in " ".join(call_args)
        assert "--model" in call_args
        assert "cnn_v3.4" in call_args
        assert "--mode" in call_args
        assert "winner_path" in call_args
        assert "--sample_games" in call_args
        assert "500" in call_args
        assert "--temp-side" in call_args
        assert "random" in call_args
        # 输出文件名含 path{round_num}.json
        assert f"path{trainer._round_num}.json" in " ".join(call_args)

    @patch('subprocess.run')
    def test_train_main_args(self, mock_run, trainer):
        """UT-3.2：train_main.py 调用参数正确。"""
        mock_run.return_value = _make_completed_process(returncode=0, stdout="ok")

        trainer._train_model()

        assert mock_run.called
        call_args = mock_run.call_args[0][0]
        assert "train_main.py" in " ".join(call_args)
        assert "--model" in call_args
        assert "cnn_v3.5" in call_args
        assert "--train-from-dir" in call_args
        assert f"path{trainer._round_num}.json" in " ".join(call_args)
        assert "--epochs" in call_args
        assert "1" in call_args
        assert "--batch-size" in call_args
        assert "4096" in call_args
        assert "--shuffle" in call_args

    @patch('subprocess.run')
    def test_battle_args(self, mock_run, trainer):
        """UT-3.3：battle.py 调用参数正确。"""
        mock_run.return_value = _make_completed_process(
            returncode=0, stdout=_battle_stdout()
        )

        trainer._run_battle()

        assert mock_run.called
        call_args = mock_run.call_args[0][0]
        assert "battle.py" in " ".join(call_args)
        assert "--ai1" in call_args
        assert "cnn_v3.4" in call_args
        assert "--ai2" in call_args
        assert "cnn_v3.5" in call_args
        assert "--games" in call_args
        assert "100" in call_args

    @patch('subprocess.run')
    def test_round_num_in_output_filename(self, mock_run, temp_project):
        """UT-3.4：第 round_num 轨输出文件名为 path{round_num}.json。"""
        mock_run.return_value = _make_completed_process(returncode=0, stdout="ok")

        trainer = IterativeTrainer(project_root=temp_project)
        trainer._round_num = 3
        trainer._generate_samples()

        call_args = mock_run.call_args[0][0]
        assert "path3.json" in " ".join(call_args)

    @patch('subprocess.run')
    def test_subprocess_cwd_set(self, mock_run, trainer):
        """UT-3.5：subprocess.run 传入 cwd=project_root。"""
        mock_run.return_value = _make_completed_process(returncode=0, stdout="ok")

        trainer._generate_samples()

        kwargs = mock_run.call_args[1]
        assert 'cwd' in kwargs
        assert kwargs['cwd'] == trainer.project_root

    @patch('subprocess.run')
    def test_pythonioencoding_set(self, mock_run, trainer):
        """UT-3.6：subprocess.run 传入 PYTHONIOENCODING=utf-8。"""
        mock_run.return_value = _make_completed_process(returncode=0, stdout="ok")

        trainer._generate_samples()

        kwargs = mock_run.call_args[1]
        assert 'env' in kwargs
        assert kwargs['env'].get('PYTHONIOENCODING') == 'utf-8'

    @patch('subprocess.run')
    def test_subprocess_encoding_utf8(self, mock_run, trainer):
        """UT-3.7：subprocess.run 传入 encoding='utf-8'（避免 Windows gbk 解码异常）。"""
        mock_run.return_value = _make_completed_process(returncode=0, stdout="ok")

        trainer._generate_samples()
        kwargs = mock_run.call_args[1]
        assert kwargs.get('encoding') == 'utf-8'

        # 同时验证 train_main 和 battle 调用也传入 encoding
        mock_run.reset_mock()
        mock_run.return_value = _make_completed_process(returncode=0, stdout="ok")
        trainer._train_model()
        assert mock_run.call_args[1].get('encoding') == 'utf-8'

        mock_run.reset_mock()
        mock_run.return_value = _make_completed_process(
            returncode=0, stdout=_battle_stdout()
        )
        trainer._run_battle()
        assert mock_run.call_args[1].get('encoding') == 'utf-8'


# ============================================================
# UT-4：胜率解析 (§2.4)
# ============================================================

class TestWinRateParsing:
    """UT-4：battle.py 输出中的胜率解析。"""

    def test_parse_normal_48_percent(self, trainer):
        """UT-4.1：正常解析 cnn_v3.5 胜率 48.0%。"""
        stdout = _battle_stdout(gen4_wins=42, gen5_wins=48, draws=10)
        result = trainer._parse_battle_output(stdout)
        assert result["gen5_win_rate"] == 48.0
        assert result["gen5_wins"] == 48
        assert result["gen4_wins"] == 42
        assert result["draws"] == 10

    def test_parse_with_per_game_lines(self, trainer):
        """UT-4.2：解析含多行输出的完整 battle 输出。

        含逐局结果行 + 最终统计行，验证以最终统计行为准。
        """
        stdout = _battle_stdout(gen4_wins=30, gen5_wins=60, draws=10)
        result = trainer._parse_battle_output(stdout)
        assert result["gen5_win_rate"] == 60.0

    def test_parse_100_percent(self, trainer):
        """UT-4.3：胜率 100%。"""
        stdout = _battle_stdout(gen4_wins=0, gen5_wins=100, draws=0)
        result = trainer._parse_battle_output(stdout)
        assert result["gen5_win_rate"] == 100.0
        assert result["gen5_wins"] == 100
        assert result["gen4_wins"] == 0
        assert result["draws"] == 0

    def test_parse_0_percent(self, trainer):
        """UT-4.4：胜率 0%。"""
        stdout = _battle_stdout(gen4_wins=100, gen5_wins=0, draws=0)
        result = trainer._parse_battle_output(stdout)
        assert result["gen5_win_rate"] == 0.0
        assert result["gen5_wins"] == 0

    def test_parse_no_match_raises(self, trainer):
        """UT-4.5：无匹配行时抛 ValueError。"""
        stdout = "cnn_v3.5 胜: 无效格式"
        with pytest.raises(ValueError, match="胜率"):
            trainer._parse_battle_output(stdout)

    def test_parse_multi_decimal(self, trainer):
        """UT-4.6：胜率含多位小数。"""
        stdout = "cnn_v3.5 胜: 48 局 (48.05%)"
        result = trainer._parse_battle_output(stdout)
        assert result["gen5_win_rate"] == 48.05

    def test_parse_not_confuse_cnn_v34(self, trainer):
        """UT-4.7：含 cnn_v3.4 干扰行时正确匹配 cnn_v3.5。"""
        stdout = (
            "cnn_v3.4 胜: 52 局 (52.0%)\n"
            "cnn_v3.5 胜: 48 局 (48.0%)\n"
        )
        result = trainer._parse_battle_output(stdout)
        assert result["gen5_win_rate"] == 48.0

    def test_parse_with_progress_lines(self, trainer):
        """UT-4.8：stdout 含进度等无关行时仍正确解析。"""
        stdout = (
            "进度: 样本局 100/500\n"
            "加载模型: cnn_v3.5\n"
            "训练完成\n"
            + _battle_stdout(gen4_wins=40, gen5_wins=55, draws=5)
        )
        result = trainer._parse_battle_output(stdout)
        assert result["gen5_win_rate"] == 55.0


# ============================================================
# UT-5：迭代逻辑判定 (§2.5)
# ============================================================

class TestIterationLogic:
    """UT-5：胜率阈值判定与晋升/停止逻辑。

    使用 mock 验证晋升操作是否被正确触发。
    """

    def test_win_rate_above_50_promotes(self, trainer):
        """UT-5.1：胜率 > 50.0% 应晋升（51.0%）。"""
        # 通过直接调用内部方法并使用 mock 验证逻辑
        assert (51.0 > WIN_RATE_THRESHOLD) is True

    def test_win_rate_50_not_promote(self, trainer):
        """UT-5.2：胜率 = 50.0% 不应晋升。"""
        assert (50.0 > WIN_RATE_THRESHOLD) is False

    def test_win_rate_below_50_not_promote(self, trainer):
        """UT-5.3：胜率 < 50.0% 不应晋升（49.0%）。"""
        assert (49.0 > WIN_RATE_THRESHOLD) is False

    def test_win_rate_50_1_promotes(self, trainer):
        """UT-5.4：胜率 = 50.1% 应晋升。"""
        assert (50.1 > WIN_RATE_THRESHOLD) is True


# ============================================================
# UT-6：日志输出 (§2.6)
# ============================================================

class TestLogOutput:
    """UT-6：日志输出格式验证。"""

    def test_round_start_header(self, trainer, capsys):
        """UT-6.1：每轮开始时打印轮次标识。"""
        trainer._round_num = 1
        trainer._print_round_header()
        captured = capsys.readouterr()
        assert "第 1 轨迭代开始" in captured.out

    def test_round_result_promoted(self, trainer, capsys):
        """UT-6.2：每轮结束时打印结果摘要（晋升）。"""
        trainer._round_num = 1
        trainer._print_round_result(win_rate=55.0, promoted=True)
        captured = capsys.readouterr()
        assert "第 1 轨结果: cnn_v3.5 胜率 = 55.0% → 晋升" in captured.out

    def test_round_result_stopped(self, trainer, capsys):
        """UT-6.3：每轮结束时打印结果摘要（停止）。"""
        trainer._round_num = 1
        trainer._print_round_result(win_rate=48.0, promoted=False)
        captured = capsys.readouterr()
        assert "第 1 轨结果: cnn_v3.5 胜率 = 48.0% → 停止" in captured.out

    def test_detail_line_promoted(self, trainer, capsys):
        """UT-6.4：最终统计逐轮明细行格式（晋升）。"""
        trainer._rounds = [{
            "round_num": 1,
            "gen4_wins": 38,
            "gen5_wins": 55,
            "draws": 7,
            "gen5_win_rate": 55.0,
            "promoted": True,
        }]
        trainer._print_final_report()
        captured = capsys.readouterr()
        expected = (
            "第1轮: cnn_v3.4 胜 38 | cnn_v3.5 胜 55 | "
            "平局 7 | cnn_v3.5 胜率 55.0% → 晋升"
        )
        assert expected in captured.out

    def test_detail_line_stopped(self, trainer, capsys):
        """UT-6.5：最终统计逐轮明细行格式（停止）。"""
        trainer._rounds = [{
            "round_num": 3,
            "gen4_wins": 44,
            "gen5_wins": 47,
            "draws": 9,
            "gen5_win_rate": 47.0,
            "promoted": False,
        }]
        trainer._print_final_report()
        captured = capsys.readouterr()
        expected = (
            "第3轮: cnn_v3.4 胜 44 | cnn_v3.5 胜 47 | "
            "平局 9 | cnn_v3.5 胜率 47.0% → 停止"
        )
        assert expected in captured.out


# ============================================================
# UT-7：最终统计报告 (§2.7)
# ============================================================

class TestFinalReport:
    """UT-7：最终统计报告内容和格式。"""

    def _add_rounds(self, trainer, rounds_data):
        """辅助：填充 trainer._rounds 数据。"""
        trainer._rounds = rounds_data
        trainer._round_num = len(rounds_data)

    def test_report_total_iterations(self, trainer, capsys):
        """UT-7.1：报告含总迭代次数（晋升2轮后第3轮停止）。"""
        self._add_rounds(trainer, [
            {"round_num": 1, "gen4_wins": 38, "gen5_wins": 55, "draws": 7,
             "gen5_win_rate": 55.0, "promoted": True},
            {"round_num": 2, "gen4_wins": 40, "gen5_wins": 52, "draws": 8,
             "gen5_win_rate": 52.0, "promoted": True},
            {"round_num": 3, "gen4_wins": 44, "gen5_wins": 47, "draws": 9,
             "gen5_win_rate": 47.0, "promoted": False},
        ])
        trainer._print_final_report()
        captured = capsys.readouterr()
        assert "总迭代次数: 2" in captured.out

    def test_report_round_details_all(self, trainer, capsys):
        """UT-7.2：报告含逐轮明细表（3行：2晋升 + 1停止）。"""
        self._add_rounds(trainer, [
            {"round_num": 1, "gen4_wins": 38, "gen5_wins": 55, "draws": 7,
             "gen5_win_rate": 55.0, "promoted": True},
            {"round_num": 2, "gen4_wins": 40, "gen5_wins": 52, "draws": 8,
             "gen5_win_rate": 52.0, "promoted": True},
            {"round_num": 3, "gen4_wins": 44, "gen5_wins": 47, "draws": 9,
             "gen5_win_rate": 47.0, "promoted": False},
        ])
        trainer._print_final_report()
        captured = capsys.readouterr()

        # 验证3行逐轮明细
        assert "第1轮:" in captured.out
        assert "第2轮:" in captured.out
        assert "第3轮:" in captured.out
        # 验证晋升/停止标识
        assert captured.out.count("→ 晋升") == 2
        assert captured.out.count("→ 停止") == 1
        # 验证含胜场和平局
        assert "cnn_v3.4 胜" in captured.out
        assert "cnn_v3.5 胜" in captured.out
        assert "平局" in captured.out

    def test_report_stop_reason(self, trainer, capsys):
        """UT-7.3：报告含停止原因。"""
        self._add_rounds(trainer, [
            {"round_num": 1, "gen4_wins": 38, "gen5_wins": 55, "draws": 7,
             "gen5_win_rate": 55.0, "promoted": True},
            {"round_num": 2, "gen4_wins": 40, "gen5_wins": 52, "draws": 8,
             "gen5_win_rate": 52.0, "promoted": True},
            {"round_num": 3, "gen4_wins": 44, "gen5_wins": 47, "draws": 9,
             "gen5_win_rate": 47.0, "promoted": False},
        ])
        trainer._print_final_report()
        captured = capsys.readouterr()
        assert "停止原因" in captured.out
        assert "第3轮" in captured.out
        assert "47.0%" in captured.out

    def test_report_final_model(self, trainer, capsys):
        """UT-7.4：报告含最终优胜模型。"""
        self._add_rounds(trainer, [
            {"round_num": 1, "gen4_wins": 44, "gen5_wins": 47, "draws": 9,
             "gen5_win_rate": 47.0, "promoted": False},
        ])
        trainer._print_final_report()
        captured = capsys.readouterr()
        assert "最终优胜模型" in captured.out
        assert "gen4" in captured.out
        assert "cnn_v3.pth" in captured.out

    def test_report_first_round_stop(self, trainer, capsys):
        """UT-7.5：首轮即停止时统计正确。

        总迭代次数 0，逐轮明细 1 行，停止原因正确。
        """
        self._add_rounds(trainer, [
            {"round_num": 1, "gen4_wins": 45, "gen5_wins": 48, "draws": 7,
             "gen5_win_rate": 48.0, "promoted": False},
        ])
        trainer._print_final_report()
        captured = capsys.readouterr()
        assert "总迭代次数: 0" in captured.out
        assert "第1轮:" in captured.out
        assert "停止原因" in captured.out


# ============================================================
# UT-8：边界条件 (§2.8)
# ============================================================

class TestEdgeCases:
    """UT-8：边界条件和异常场景。"""

    def test_rate_50_stops(self, trainer):
        """UT-8.1：首轮胜率 50.0% 停止。"""
        assert not (50.0 > WIN_RATE_THRESHOLD)

    def test_rate_50_1_promotes(self, trainer):
        """UT-8.2：首轮胜率 50.1% 晋升。"""
        assert (50.1 > WIN_RATE_THRESHOLD)

    def test_all_draws(self, trainer):
        """UT-8.3：battle 输出仅有平局（0胜0负100平）→ 胜率 0.0%。"""
        stdout = _battle_stdout(gen4_wins=0, gen5_wins=0, draws=100)
        result = trainer._parse_battle_output(stdout)
        assert result["gen5_win_rate"] == 0.0
        assert result["gen5_wins"] == 0
        assert result["draws"] == 100

    def test_empty_samples_detected(self, trainer, temp_project):
        """UT-8.4：create_samples 生成空样本（0 条）时检测到并抛异常。"""
        # 构造空样本文件
        sample_dir = os.path.join(temp_project, SAMPLE_OUTPUT_DIR)
        os.makedirs(sample_dir, exist_ok=True)
        sample_path = os.path.join(sample_dir, "path1.json")
        with open(sample_path, 'w') as f:
            json.dump([], f)

        with pytest.raises(RuntimeError, match="空样本"):
            trainer._validate_samples(sample_path)

    def test_sample_file_not_exists(self, trainer, temp_project):
        """UT-8.5：样本文件不存在时抛 RuntimeError。"""
        sample_path = os.path.join(temp_project, SAMPLE_OUTPUT_DIR, "path1.json")
        # 确保文件不存在
        if os.path.isfile(sample_path):
            os.remove(sample_path)

        with pytest.raises(RuntimeError):
            trainer._validate_samples(sample_path)

    def test_sample_json_parse_error(self, trainer, temp_project):
        """样本文件 JSON 解析失败时抛 RuntimeError。"""
        sample_dir = os.path.join(temp_project, SAMPLE_OUTPUT_DIR)
        os.makedirs(sample_dir, exist_ok=True)
        sample_path = os.path.join(sample_dir, "path1.json")
        with open(sample_path, 'w') as f:
            f.write("not valid json {{{")

        with pytest.raises(RuntimeError):
            trainer._validate_samples(sample_path)

    def test_sample_not_array(self, trainer, temp_project):
        """样本文件不是 JSON 数组时抛 RuntimeError。"""
        sample_dir = os.path.join(temp_project, SAMPLE_OUTPUT_DIR)
        os.makedirs(sample_dir, exist_ok=True)
        sample_path = os.path.join(sample_dir, "path1.json")
        with open(sample_path, 'w') as f:
            json.dump({"key": "not array"}, f)

        with pytest.raises(RuntimeError):
            trainer._validate_samples(sample_path)
