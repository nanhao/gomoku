"""IterativeTrainer 单元测试

测试 IterativeTrainer 类各方法的单元行为，使用 mock subprocess.run
和临时目录隔离文件系统副作用。

测试用例覆盖：
- UT-1：构造器校验
- UT-2：模型文件操作
- UT-3：子进程调用参数构造
- UT-4：胜率解析
- UT-8：边界条件

UT-6/UT-7 已拆分到 test_iterative_train_report.py。
UT-5 已删除（无效测试，只验证 Python 内置比较逻辑）。
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
)
from tests.ai.helpers import _make_completed_process


@pytest.fixture
def trainer(temp_project):
    """创建 IterativeTrainer 实例（项目根为临时目录）。"""
    return IterativeTrainer(project_root=temp_project)


# _make_completed_process 已移入 tests.ai.helpers，从该处导入


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

    def test_promote_method_removed(self, trainer):
        """UT-2.3（意图 12 更新）：_promote() 方法已移除，不再有晋升操作。"""
        assert not hasattr(trainer, '_promote'), (
            "意图 12 不再需要晋升逻辑，_promote() 应已移除"
        )

    def test_sample_output_dir_auto_create(self, trainer, temp_project):
        """UT-2.4：样本输出目录不存在时自动创建。"""
        sample_dir = os.path.join(temp_project, SAMPLE_OUTPUT_DIR)
        if os.path.exists(sample_dir):
            shutil.rmtree(sample_dir)

        # 直接验证 _generate_samples 中的 os.makedirs 调用
        # 通过调用 _ensure_sample_dir 或等效逻辑
        trainer._ensure_sample_output_dir()

        assert os.path.isdir(sample_dir)

    def test_promote_method_removed_confirm_no_copy2(self, trainer, temp_project):
        """UT-2.5（意图 12 更新）：_promote() 已移除，gen4 不会被 gen5 覆盖。

        验证：对 trainer 实例调用 hasattr 确认 _promote 不存在。
        """
        assert not hasattr(trainer, '_promote'), (
            "_promote() 已在意图 12 中移除"
        )


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
        assert "--sample_games" in call_args
        assert "500" in call_args
        # 意图 12：不应包含 --mode 和 --temp-side
        assert "--mode" not in call_args
        assert "--temp-side" not in call_args
        # 输出文件名含 path{round_num}.json
        assert f"path{trainer._round_num}.json" in " ".join(call_args)

    @patch('subprocess.run')
    def test_train_main_args(self, mock_run, trainer):
        """UT-3.2：train_main.py 调用参数正确（意图 12：epochs 改为 100）。"""
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
        assert "100" in call_args
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
# UT-8：边界条件 (§2.8)
# ============================================================

class TestEdgeCases:
    """UT-8：边界条件和异常场景。"""

    def test_loop_always_continues(self, trainer):
        """UT-8.1（意图 12 更新）：循环始终继续，不因胜率低而停止。

        验证 WIN_RATE_THRESHOLD 常量已移除，不再有胜率判定逻辑。
        """
        # 导入模块确认 WIN_RATE_THRESHOLD 已移除
        import ai.iterative_train as it_mod
        assert not hasattr(it_mod, 'WIN_RATE_THRESHOLD'), (
            "意图 12 不再需要 WIN_RATE_THRESHOLD 常量"
        )

    def test_loop_always_continues_2(self, trainer):
        """UT-8.2（意图 12 更新）：循环始终继续，不因胜率低而停止。

        确认 _promote() 也已移除（晋升不再发生）。
        """
        assert not hasattr(trainer, '_promote')

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
