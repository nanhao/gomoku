"""iterative_train.py 测试 — 意图 12 变更验证

测试迭代进化训练脚本的核心逻辑：
- TRAIN_EPOCHS 常量为 100
- create_samples 子进程调用不包含 --mode 和 --temp-side 参数
- train_main 子进程调用使用 TRAIN_EPOCHS（100）个 epoch
- 循环始终继续，不因胜率低而停止
- KeyboardInterrupt 时返回有效结果字典
"""

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import patch, Mock

# 添加项目根目录
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.iterative_train import (
    IterativeTrainer, TRAIN_EPOCHS, PYTHON_EXECUTABLE,
    SAMPLE_GAMES, BATTLE_GAMES, TRAIN_BATCH_SIZE,
    MODELS_DIR, PARENT_GEN, CHILD_GEN, SAMPLE_OUTPUT_DIR,
)
from tests.ai.helpers import _make_completed_process


class TestConstants(unittest.TestCase):
    """测试意图 12 变更后的常量值"""

    def test_train_epochs_is_100(self):
        """意图 12 变更：TRAIN_EPOCHS 应为 100（原来为 1）"""
        self.assertEqual(
            TRAIN_EPOCHS, 100,
            f"意图 12 要求 TRAIN_EPOCHS=100，实际为 {TRAIN_EPOCHS}"
        )

    def test_win_rate_threshold_removed(self):
        """意图 12 变更：WIN_RATE_THRESHOLD 常量应已移除（不再需要胜率判定）"""
        with self.assertRaises(ImportError):
            from ai.iterative_train import WIN_RATE_THRESHOLD


class TestGenerateSamplesCmd(unittest.TestCase):
    """测试 _generate_samples() 构造的子进程命令"""

    def setUp(self):
        # 确保 gen4 权重目录存在
        self.tmp_dir = tempfile.mkdtemp()
        gen4_dir = os.path.join(self.tmp_dir, MODELS_DIR, f"gen{PARENT_GEN}")
        os.makedirs(gen4_dir, exist_ok=True)
        gen4_path = os.path.join(gen4_dir, "cnn_v3.pth")
        with open(gen4_path, 'w') as f:
            f.write("fake_weights")

        # 创建 trainer 实例
        self.trainer = IterativeTrainer(project_root=self.tmp_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_cmd_has_no_mode_flag(self):
        """意图 12 变更：create_samples 命令不应包含 --mode winner_path"""
        cmd_args = self._capture_generate_samples_cmd()

        # 验证不包含 --mode 参数
        self.assertNotIn('--mode', cmd_args,
                         f"意图 12 要求移除 --mode 参数，实际 cmd 包含: {cmd_args}")

    def test_cmd_has_no_temp_side_flag(self):
        """意图 12 变更：create_samples 命令不应包含 --temp-side random"""
        cmd_args = self._capture_generate_samples_cmd()

        # 验证不包含 --temp-side 参数
        self.assertNotIn('--temp-side', cmd_args,
                         f"意图 12 要求移除 --temp-side 参数，实际 cmd 包含: {cmd_args}")

    def test_cmd_has_correct_basic_structure(self):
        """create_samples 命令基础结构检查：python、脚本名、--model、--sample_games、--output"""
        cmd_args = self._capture_generate_samples_cmd()
        expected_sample_path = os.path.join(
            self.tmp_dir, SAMPLE_OUTPUT_DIR, f"path{self.trainer._round_num}.json"
        )

        self.assertIn(PYTHON_EXECUTABLE, cmd_args)
        self.assertIn('ai/create_samples.py', cmd_args)
        self.assertIn('--model', cmd_args)
        self.assertIn('cnn_v3.4', cmd_args[cmd_args.index('--model') + 1])
        self.assertIn('--sample_games', cmd_args)
        self.assertIn(str(SAMPLE_GAMES), cmd_args[cmd_args.index('--sample_games') + 1])
        self.assertIn('--output', cmd_args)
        self.assertIn(expected_sample_path, cmd_args[cmd_args.index('--output') + 1])

    def _capture_generate_samples_cmd(self):
        """拦截 _run_subprocess 调用，捕获 generate_samples 构造的 cmd 列表。"""
        captured_cmd = []

        def fake_run_subprocess(cmd):
            captured_cmd.extend(cmd)
            return _make_completed_process(stdout="样本局数: 500")

        with patch.object(self.trainer, '_run_subprocess', side_effect=fake_run_subprocess):
            with patch.object(self.trainer, '_ensure_sample_output_dir'):
                self.trainer._generate_samples()

        return captured_cmd


class TestTrainModelCmd(unittest.TestCase):
    """测试 _train_model() 构造的子进程命令"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        gen4_dir = os.path.join(self.tmp_dir, MODELS_DIR, f"gen{PARENT_GEN}")
        os.makedirs(gen4_dir, exist_ok=True)
        gen4_path = os.path.join(gen4_dir, "cnn_v3.pth")
        with open(gen4_path, 'w') as f:
            f.write("fake_weights")
        self.trainer = IterativeTrainer(project_root=self.tmp_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_train_model_uses_training_epochs(self):
        """意图 12 变更：train_main 命令应使用 TRAIN_EPOCHS（100），而非硬编码的 1"""
        captured_cmd = []

        def fake_run_subprocess(cmd):
            captured_cmd.extend(cmd)
            return _make_completed_process()

        with patch.object(self.trainer, '_run_subprocess', side_effect=fake_run_subprocess):
            self.trainer._train_model()

        # 验证 --epochs 参数值为 TRAIN_EPOCHS（100）
        if '--epochs' in captured_cmd:
            epochs_idx = captured_cmd.index('--epochs')
            actual_epochs = captured_cmd[epochs_idx + 1]
            self.assertEqual(
                actual_epochs, str(TRAIN_EPOCHS),
                f"train_main 命令的 --epochs 应为 {TRAIN_EPOCHS}，实际为 {actual_epochs}"
            )

    def test_train_model_cmd_structure(self):
        """train_main 命令包含正确的脚本名、--model、--train-from-dir、--batch-size、--shuffle"""
        captured_cmd = []

        def fake_run_subprocess(cmd):
            captured_cmd.extend(cmd)
            return _make_completed_process()

        with patch.object(self.trainer, '_run_subprocess', side_effect=fake_run_subprocess):
            self.trainer._train_model()

        self.assertIn('ai/train_main.py', captured_cmd)
        self.assertIn('--model', captured_cmd)
        self.assertIn('cnn_v3.5', captured_cmd)
        self.assertIn('--train-from-dir', captured_cmd)
        self.assertIn('--batch-size', captured_cmd)
        self.assertIn(str(TRAIN_BATCH_SIZE), captured_cmd)
        self.assertIn('--shuffle', captured_cmd)


class TestRunAlwaysContinues(unittest.TestCase):
    """测试 run() 循环始终继续，不因胜率低而停止"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        gen4_dir = os.path.join(self.tmp_dir, MODELS_DIR, f"gen{PARENT_GEN}")
        os.makedirs(gen4_dir, exist_ok=True)
        gen4_path = os.path.join(gen4_dir, "cnn_v3.pth")
        with open(gen4_path, 'w') as f:
            f.write("fake_weights")
        self.trainer = IterativeTrainer(project_root=self.tmp_dir)

    def tearDown(self):
        # 注意：run() 内部 try-finally 会删除 lock，但这里可能未创建
        # 手动确保清理
        lock_path = os.path.join(self.tmp_dir, MODELS_DIR, ".iterative_lock")
        if os.path.isfile(lock_path):
            os.remove(lock_path)
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_run_continues_when_win_rate_below_threshold(self):
        """意图 12 变更：即使 cnn_v3.5 胜率很低（如 10%），循环也应继续，而非停止"""
        # 使用键盘中断来控制循环次数
        call_count = [0]
        max_calls = 4  # 在 4 次 _run_subprocess 调用后触发中断

        # 模拟 battle 输出：胜率 10%（远低于原阈值 50%）
        battle_output = (
            "第 1 局: cnn_v3.4 胜\n"
            "第 2 局: cnn_v3.4 胜\n"
            "...\n"
            "cnn_v3.4 胜: 90 局\n"
            "cnn_v3.5 胜: 10 局 (10.0%)\n"
            "平局: 0 局\n"
        )

        def fake_run_subprocess(cmd):
            call_count[0] += 1
            if call_count[0] > max_calls:
                raise KeyboardInterrupt()
            return _make_completed_process(stdout=battle_output)

        with patch.object(self.trainer, '_run_subprocess', side_effect=fake_run_subprocess):
            with patch.object(self.trainer, '_ensure_sample_output_dir'):
                with patch.object(self.trainer, '_validate_samples'):
                    result = self.trainer.run()

        # 验证：至少完成了 1 轮（低胜率没有导致提前停止）
        self.assertGreaterEqual(
            len(result['rounds']), 1,
            f"意图 12 要求始终继续，但只完成了 {len(result['rounds'])} 轮"
        )

        # 验证：所有轮次的 promoted 均为 False（不再有晋升操作）
        for r in result['rounds']:
            self.assertFalse(r['promoted'],
                             f"意图 12 不执行晋升，但发现 promoted=True: {r}")

    def test_run_continues_when_win_rate_high(self):
        """意图 12 变更：即使 cnn_v3.5 胜率很高（如 90%），也不执行晋升，仅继续循环"""
        call_count = [0]
        max_calls = 2

        # 模拟 battle 输出：胜率 90%
        battle_output = (
            "cnn_v3.4 胜: 10 局\n"
            "cnn_v3.5 胜: 90 局 (90.0%)\n"
            "平局: 0 局\n"
        )

        def fake_run_subprocess(cmd):
            call_count[0] += 1
            if call_count[0] > max_calls * 3:  # 3 subprocess calls per round
                raise KeyboardInterrupt()
            return _make_completed_process(stdout=battle_output)

        with patch.object(self.trainer, '_run_subprocess', side_effect=fake_run_subprocess):
            with patch.object(self.trainer, '_ensure_sample_output_dir'):
                with patch.object(self.trainer, '_validate_samples'):
                    result = self.trainer.run()

        # 验证所有轮次 promoted 均为 False
        for r in result['rounds']:
            self.assertFalse(r['promoted'],
                             f"意图 12 不执行晋升，但发现 promoted=True: {r}")

    def test_round_num_increments_each_cycle(self):
        """意图 12 变更：每轮 i++，_round_num 持续递增"""
        call_count = [0]
        max_calls = 4

        battle_output = "cnn_v3.4 胜: 50 局\ncnn_v3.5 胜: 50 局 (50.0%)\n平局: 0 局\n"

        def fake_run_subprocess(cmd):
            call_count[0] += 1
            if call_count[0] > max_calls * 3:
                raise KeyboardInterrupt()
            return _make_completed_process(stdout=battle_output)

        with patch.object(self.trainer, '_run_subprocess', side_effect=fake_run_subprocess):
            with patch.object(self.trainer, '_ensure_sample_output_dir'):
                with patch.object(self.trainer, '_validate_samples'):
                    result = self.trainer.run()

        # 验证每轮的 round_num 递增
        for i, r in enumerate(result['rounds'], start=1):
            self.assertEqual(r['round_num'], i,
                             f"第 {i} 条记录的 round_num 应为 {i}，实际为 {r['round_num']}")


class TestKeyboardInterruptResult(unittest.TestCase):
    """测试 KeyboardInterrupt 时返回有效结果"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        gen4_dir = os.path.join(self.tmp_dir, MODELS_DIR, f"gen{PARENT_GEN}")
        os.makedirs(gen4_dir, exist_ok=True)
        gen4_path = os.path.join(gen4_dir, "cnn_v3.pth")
        with open(gen4_path, 'w') as f:
            f.write("fake_weights")
        self.trainer = IterativeTrainer(project_root=self.tmp_dir)

    def tearDown(self):
        lock_path = os.path.join(self.tmp_dir, MODELS_DIR, ".iterative_lock")
        if os.path.isfile(lock_path):
            os.remove(lock_path)
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_keyboard_interrupt_returns_dict(self):
        """KeyboardInterrupt 时应返回包含 rounds 的字典，而非抛异常"""
        battle_output = "cnn_v3.4 胜: 50 局\ncnn_v3.5 胜: 50 局 (50.0%)\n平局: 0 局\n"

        def fake_run_subprocess(cmd):
            # 第一轮完成（3 次调用）后立即中断
            if not hasattr(fake_run_subprocess, 'count'):
                fake_run_subprocess.count = 0
            fake_run_subprocess.count += 1
            if fake_run_subprocess.count > 3:
                raise KeyboardInterrupt()
            return _make_completed_process(stdout=battle_output)

        with patch.object(self.trainer, '_run_subprocess', side_effect=fake_run_subprocess):
            with patch.object(self.trainer, '_ensure_sample_output_dir'):
                with patch.object(self.trainer, '_validate_samples'):
                    result = self.trainer.run()

        # 验证返回值类型和关键字段
        self.assertIsInstance(result, dict)
        self.assertIn('rounds', result)
        self.assertIn('final_model_path', result)
        self.assertGreater(len(result['rounds']), 0,
                           "至少应完成 1 轮")


class TestNoPromotionMode(unittest.TestCase):
    """测试不再有晋升（promote）逻辑"""

    def test_promote_method_does_not_exist(self):
        """意图 12 变更：_promote() 方法应已移除"""
        self.tmp_dir = tempfile.mkdtemp()
        gen4_dir = os.path.join(self.tmp_dir, MODELS_DIR, f"gen{PARENT_GEN}")
        os.makedirs(gen4_dir, exist_ok=True)
        gen4_path = os.path.join(gen4_dir, "cnn_v3.pth")
        with open(gen4_path, 'w') as f:
            f.write("fake_weights")

        trainer = IterativeTrainer(project_root=self.tmp_dir)

        # 验证 _promote 方法不存在
        self.assertFalse(
            hasattr(trainer, '_promote'),
            "意图 12 不再需要晋升逻辑，_promote() 方法应已移除"
        )

        import shutil
        lock_path = os.path.join(self.tmp_dir, MODELS_DIR, ".iterative_lock")
        if os.path.isfile(lock_path):
            os.remove(lock_path)
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
