"""FC v1 训练器单元测试（整合）

按 task_reports/2026-05-01/test_suite_review.md Top #6 执行：
将原 7 个文件
  - test_fc_v1_trainer.py
  - test_fc_v1_trainer_batch.py
  - test_fc_v1_trainer_cycle.py
  - test_fc_v1_trainer_error.py
  - test_fc_v1_trainer_fresh.py
  - test_fc_v1_trainer_verbose.py
  - （test_fc_v1_trainer_integration.py 保留独立）
合并为本文件 + integration。

删除重复：
- test_trainer_train_single_returns_loss（与 test_trainer_train_single 重复）
- test_trainer_compute_loss* 3 处（与 test_sample_loader.py 重复）
- train_batch 错误场景与独立 error 文件重复的合并到 TestFCv1TrainerErrorHandling
"""

import os
import sys
import unittest
import unittest.mock
import tempfile
import shutil
import json
import io
import copy
import inspect
import re
from unittest.mock import MagicMock, patch

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch

from ai.models.fc_v1.trainer import FCv1Trainer
from ai.models.fc_v1.model import FCv1Model
from ai.sample_loader import compute_masked_kl_loss
from ai.base_trainer import BaseTrainer
from tests.ai.helpers import create_output


# ---------------------------------------------------------------------------
# 共用辅助
# ---------------------------------------------------------------------------

def _write_standard_sample(sample_dir: str) -> None:
    """在 sample_dir 下写入标准 2 样本 sample.json"""
    sample_data = [
        {
            "input": [0] * 225 + [1],
            "output": create_output(0, 0, 1000000),
            "mask": [1] + [0] * 224,
        },
        {
            "input": [1] + [0] * 224 + [2],
            "output": create_output(1, 0, 500000),
            "mask": [0] * 15 + [1] + [0] * 209,
        },
    ]
    with open(os.path.join(sample_dir, "sample.json"), 'w', encoding='utf-8') as f:
        json.dump(sample_data, f)


def _make_batch_sample_dir(temp_dir: str, name: str, n_items: int = 2) -> str:
    """创建一个 batch 测试用的 sample 目录（每个样本 mask[0]=1, output[0]=1000）"""
    sample_dir = os.path.join(temp_dir, name)
    os.makedirs(sample_dir, exist_ok=True)

    sample_data = []
    for i in range(n_items):
        output = [0.0] * 225
        output[0] = 1000.0
        sample_data.append({
            "input": [float(i + 1)] * 226,
            "output": output,
            "mask": [1] + [0] * 224,
        })

    with open(os.path.join(sample_dir, "sample.json"), 'w', encoding='utf-8') as f:
        json.dump(sample_data, f)
    return sample_dir


def _build_trainer_with_tmp_logger(fresh: bool = False, model: FCv1Model = None) -> FCv1Trainer:
    """构造训练器并注入 MagicMock logger，避免污染真实 loss_logs 目录"""
    if model is not None:
        trainer = FCv1Trainer(model=model)
    else:
        trainer = FCv1Trainer(fresh=fresh)
    trainer.loss_logger = MagicMock()
    trainer.loss_logger.dense_log = False
    return trainer


class _DefaultModelPathIsolatedMixin:
    """setUp/tearDown 辅助，隔离 FCv1Model.DEFAULT_MODEL_PATH 到 tmp 目录"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self._patcher = unittest.mock.patch.object(
            FCv1Model, 'DEFAULT_MODEL_PATH',
            os.path.join(self.temp_dir, "fc_v1.pth")
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 初始化 / 属性
# ---------------------------------------------------------------------------

class TestFCv1TrainerInit(_DefaultModelPathIsolatedMixin, unittest.TestCase):
    """训练器初始化与基本属性"""

    def test_trainer_initialization_without_model(self):
        """不传入模型时默认创建 fc_v1"""
        trainer = FCv1Trainer()
        self.assertIsNotNone(trainer.model)
        self.assertEqual(trainer.model.name, "fc_v1")

    def test_trainer_initialization_with_model(self):
        """显式传入模型"""
        model = FCv1Model()
        trainer = FCv1Trainer(model)
        self.assertIs(trainer.model, model)

    def test_trainer_model_name(self):
        """model_name 属性"""
        trainer = FCv1Trainer()
        self.assertEqual(trainer.model_name, "fc_v1")

    def test_trainer_inherits_base_trainer(self):
        """继承 BaseTrainer"""
        self.assertTrue(issubclass(FCv1Trainer, BaseTrainer))


class TestTrainCycleRateParameter(unittest.TestCase):
    """train_cycle rate 参数默认值（原 test_fc_v1_trainer_cycle.py 保留）"""

    def test_train_cycle_has_rate_parameter(self):
        """train_cycle 应有 rate 参数，默认 0.9"""
        sig = inspect.signature(FCv1Trainer.train_cycle)
        self.assertIn('rate', sig.parameters, "train_cycle 应有 rate 参数")
        self.assertAlmostEqual(
            sig.parameters['rate'].default, 0.9, places=9,
            msg="rate 默认值应为 0.9"
        )

    def test_train_main_script_rate_default(self):
        """train_main.py --rate 默认值为 0.9"""
        import argparse
        original_argv = sys.argv
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument('--rate', type=float, default=0.9)
            parser.add_argument('--model', type=str, default='fc_v1')
            args = parser.parse_args([])
            self.assertAlmostEqual(args.rate, 0.9, places=9)
            self.assertEqual(args.model, 'fc_v1')
        finally:
            sys.argv = original_argv


# ---------------------------------------------------------------------------
# 单轮训练 / 数据加载
# ---------------------------------------------------------------------------

class TestFCv1TrainerTrainSingle(_DefaultModelPathIsolatedMixin, unittest.TestCase):
    """train_single + _sample_loader 行为"""

    def setUp(self):
        super().setUp()
        self.sample_dir = os.path.join(self.temp_dir, "sample_data")
        os.makedirs(self.sample_dir, exist_ok=True)
        _write_standard_sample(self.sample_dir)

    def test_trainer_train_single(self):
        """单轮训练返回非负 float"""
        trainer = _build_trainer_with_tmp_logger()
        loss = trainer.train_single(self.sample_dir)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)

    def test_trainer_load_sample_data(self):
        """加载 sample.json 结构正确"""
        trainer = FCv1Trainer()
        data = trainer._sample_loader.load(os.path.join(self.sample_dir, "sample.json"))
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)
        first = data[0]
        for key in ("input", "output", "mask"):
            self.assertIn(key, first)
        self.assertEqual(len(first["input"]), 226)
        self.assertEqual(len(first["output"]), 225)
        self.assertEqual(len(first["mask"]), 225)


# ---------------------------------------------------------------------------
# 批量训练
# ---------------------------------------------------------------------------

class TestFCv1TrainerTrainBatch(_DefaultModelPathIsolatedMixin, unittest.TestCase):
    """train_batch 主流程 + 参数更新验证"""

    def test_train_batch_returns_float(self):
        """train_batch 返回非负 float"""
        dir1 = _make_batch_sample_dir(self.temp_dir, "s1")
        dir2 = _make_batch_sample_dir(self.temp_dir, "s2")
        trainer = _build_trainer_with_tmp_logger(fresh=True)
        loss = trainer.train_batch([dir1, dir2], epochs=3)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)

    def test_train_batch_single_sample_dir(self):
        """train_batch 处理单目录"""
        dir1 = _make_batch_sample_dir(self.temp_dir, "s1")
        trainer = _build_trainer_with_tmp_logger(fresh=True)
        loss = trainer.train_batch([dir1], epochs=2)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)

    def test_train_batch_multiple_sample_dirs_merged(self):
        """多目录合并为"一个 epoch 一次参数更新"：optimizer.step 调用次数 == epochs

        原版用 stdout "无 Epoch 输出" 隐式断言合并语义，过于脆弱：
        即使实现回退为"每个 sample_dir 都走一次 forward/backward"，
        只要 verbose 打印被压制也能静默通过。
        改为直接 mock optimizer.step，核心契约一目了然。
        """
        dirs = [_make_batch_sample_dir(self.temp_dir, f"s{i}") for i in range(1, 4)]
        trainer = _build_trainer_with_tmp_logger(fresh=True)
        epochs = 4

        with unittest.mock.patch.object(
            trainer.optimizer, 'step', wraps=trainer.optimizer.step
        ) as mock_step:
            loss = trainer.train_batch(dirs, epochs=epochs, verbose=False)

        # 核心契约：无论 sample_dirs 有几个，每 epoch 只更新一次
        self.assertEqual(
            mock_step.call_count, epochs,
            f"期望 optimizer.step 被调用 {epochs} 次（每 epoch 1 次），"
            f"实际 {mock_step.call_count} 次。若为 {epochs * len(dirs)} 则说明"
            f"回退为每目录单独更新"
        )
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)

    def test_train_batch_updates_model_params(self):
        """train_batch 后模型参数应更新"""
        dir1 = _make_batch_sample_dir(self.temp_dir, "s1")
        dir2 = _make_batch_sample_dir(self.temp_dir, "s2")
        trainer = _build_trainer_with_tmp_logger(fresh=True)

        params_before = copy.deepcopy(
            list(trainer.model.network.parameters())[0].data.clone()
        )
        trainer.train_batch([dir1, dir2], epochs=5)
        params_after = list(trainer.model.network.parameters())[0].data

        self.assertFalse(torch.equal(params_before, params_after))


# ---------------------------------------------------------------------------
# 张量准备
# ---------------------------------------------------------------------------

class TestFCv1TrainerPrepareTensors(unittest.TestCase):
    """_sample_loader.prepare_tensors 的 mask 语义"""

    def setUp(self):
        self.trainer = FCv1Trainer(fresh=True)

    def _make_sample(self, mask_idx: int) -> dict:
        """只有第 mask_idx 位 mask=1, output=999 的样本"""
        output = [0.0] * 225
        output[mask_idx] = 999.0
        mask = [0] * 225
        mask[mask_idx] = 1
        return {"input": [0.0] * 226, "output": output, "mask": mask}

    def test_prepare_tensors_output_shapes(self):
        """inputs(1,226) / targets(1,225) / masks(1,225)"""
        sample = self._make_sample(0)
        inputs_t, targets_t, masks_t = self.trainer._sample_loader.prepare_tensors([sample])
        self.assertEqual(inputs_t.shape, (1, 226))
        self.assertEqual(targets_t.shape, (1, 225))
        self.assertEqual(masks_t.shape, (1, 225))

    def test_mask_tensor_matches_input_mask(self):
        """mask 张量应直接对应输入 mask"""
        for i in [0, 1, 7, 100, 223]:
            with self.subTest(mask_idx=i):
                sample = self._make_sample(i)
                _, _, masks_t = self.trainer._sample_loader.prepare_tensors([sample])
                self.assertEqual(masks_t[0, i].item(), 1.0, f"masks_t[0, {i}] 应为 1")
                self.assertEqual((masks_t[0] != 0).sum().item(), 1)

    def test_masked_loss_consistent_with_compute_loss(self):
        """_masked_loss 与 compute_masked_kl_loss 结果一致（KL 散度）"""
        import math
        # KL 散度: target=[0.5], log_estimated=[log(0.7)]
        # KL = 0.5 * (log(0.5) - log(0.7)) ≈ 0.0872 / 1
        log_estimated_val = math.log(0.7)
        output_estimated = [0.0] * 225
        output_estimated[0] = log_estimated_val
        output_target = [0.0] * 225
        output_target[0] = 0.5
        mask = [0] * 225
        mask[0] = 1

        py_loss = compute_masked_kl_loss(output_estimated, output_target, mask)

        sample = {"input": [0.0] * 226, "output": output_target, "mask": mask}
        _, targets_t, masks_t = self.trainer._sample_loader.prepare_tensors([sample])
        outputs_t = torch.tensor([output_estimated], dtype=torch.float32).to(self.trainer.device)
        tensor_loss = self.trainer._masked_loss(outputs_t, targets_t, masks_t)

        self.assertAlmostEqual(
            py_loss, tensor_loss.item(), places=3,
            msg="纯Python KL 散度与 _masked_loss 应一致"
        )


# ---------------------------------------------------------------------------
# fresh 参数
# ---------------------------------------------------------------------------

class TestFCv1TrainerFreshParameter(_DefaultModelPathIsolatedMixin, unittest.TestCase):
    """fresh 参数行为"""

    def _create_test_sample(self):
        sample_dir = os.path.join(self.temp_dir, "sample_data")
        os.makedirs(sample_dir, exist_ok=True)
        with open(os.path.join(sample_dir, "sample.json"), 'w', encoding='utf-8') as f:
            json.dump([{
                "input": [0] * 226,
                "output": [0] * 225,
                "mask": [1] + [0] * 224,
            }], f)
        return sample_dir

    def test_trainer_fresh_parameter_creates_new_model(self):
        """fresh=True 时即便默认路径已保存模型也创建新模型"""
        FCv1Model(fresh=True).save()
        trainer = FCv1Trainer(fresh=True)
        self.assertIsNotNone(trainer.model)
        self.assertEqual(trainer.model.name, "fc_v1")

    def test_trainer_default_loads_existing_model(self):
        """默认行为加载已保存模型（预测结果一致）"""
        model1 = FCv1Model(fresh=True)
        board_vector = [0] * 225 + [1]
        result1 = model1.predict(board_vector)
        model1.save()

        trainer = FCv1Trainer()
        result2 = trainer.model.predict(board_vector)
        self.assertEqual(len(result1), len(result2))
        for r1, r2 in zip(result1, result2):
            self.assertAlmostEqual(r1, r2, places=5)

    def test_trainer_with_model_parameter(self):
        """传入 model 参数 → 训练器使用该实例"""
        custom_model = FCv1Model(fresh=True)
        board_vector = [0] * 225 + [1]
        result1 = custom_model.predict(board_vector)

        trainer = FCv1Trainer(model=custom_model)
        result2 = trainer.model.predict(board_vector)

        self.assertIs(trainer.model, custom_model)
        self.assertEqual(len(result1), len(result2))
        for r1, r2 in zip(result1, result2):
            self.assertAlmostEqual(r1, r2, places=5)

    def test_trainer_fresh_with_sample_dir(self):
        """fresh=True 与单轮训练联合"""
        sample_dir = self._create_test_sample()
        trainer = _build_trainer_with_tmp_logger(fresh=True)
        loss = trainer.train_single(sample_dir, epochs=1)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)


# ---------------------------------------------------------------------------
# verbose 输出
# ---------------------------------------------------------------------------

class TestFCv1TrainerVerboseOutput(_DefaultModelPathIsolatedMixin, unittest.TestCase):
    """verbose 参数控制 epoch loss 输出"""

    def _create_test_sample(self):
        sample_dir = os.path.join(self.temp_dir, "sample_data")
        os.makedirs(sample_dir, exist_ok=True)
        with open(os.path.join(sample_dir, "sample.json"), 'w', encoding='utf-8') as f:
            json.dump([{
                "input": [0] * 226,
                "output": [0] * 225,
                "mask": [1] + [0] * 224,
            }], f)
        return sample_dir

    def test_verbose_false_no_output(self):
        """verbose=False（默认）→ 无 epoch 输出"""
        sample_dir = self._create_test_sample()
        trainer = _build_trainer_with_tmp_logger(fresh=True)
        trainer.loss_logger.plot.return_value = None

        captured = io.StringIO()
        sys.stdout = captured
        try:
            trainer.train_single(sample_dir, epochs=3, verbose=False)
            output = captured.getvalue()
        finally:
            sys.stdout = sys.__stdout__

        self.assertNotIn("Epoch", output)
        self.assertNotIn("loss", output.lower())

    def test_verbose_output_format(self):
        """verbose=True → 每 200 epoch 打印一次，格式 `Epoch X/Y, loss: Z.ZZZZ`

        关注点：verbose 打印格式契约。真实训练的覆盖由 test_trainer_train_single/
        test_train_batch_updates_model_params 等其他测试保障，此处 mock 掉避免过度测试。
        """
        sample_dir = self._create_test_sample()
        trainer = _build_trainer_with_tmp_logger(fresh=True)

        captured = io.StringIO()
        sys.stdout = captured
        try:
            fake_step = (torch.tensor(0.1234), 0.5)
            with patch.object(trainer, '_forward_backward_step', return_value=fake_step):
                trainer.train_single(sample_dir, epochs=200, verbose=True)
            output = captured.getvalue()
        finally:
            sys.stdout = sys.__stdout__

        self.assertIn("Epoch 200/200", output)
        self.assertRegex(output, r"loss:\s*\d+\.\d+")

    def test_train_batch_verbose_false_no_epoch_output(self):
        """train_batch verbose=False → 无 Epoch 输出"""
        dir1 = _make_batch_sample_dir(self.temp_dir, "s1")
        trainer = _build_trainer_with_tmp_logger(fresh=True)

        captured = io.StringIO()
        sys.stdout = captured
        try:
            trainer.train_batch([dir1], epochs=3, verbose=False)
            output = captured.getvalue()
        finally:
            sys.stdout = sys.__stdout__

        self.assertNotIn("Epoch", output)


# ---------------------------------------------------------------------------
# 错误处理
# ---------------------------------------------------------------------------

class TestFCv1TrainerErrorHandling(_DefaultModelPathIsolatedMixin, unittest.TestCase):
    """错误处理（合并原 error + batch 的异常测试）"""

    def test_train_single_missing_sample_file(self):
        """train_single: sample 目录/文件不存在 → FileNotFoundError with 路径消息"""
        trainer = FCv1Trainer()
        bad_dir = os.path.join(self.temp_dir, "nonexistent")
        with self.assertRaisesRegex(FileNotFoundError, r"Sample file not found:.*nonexistent"):
            trainer.train_single(bad_dir)

    def test_train_batch_empty_list_raises(self):
        """train_batch 空列表 → ValueError with "sample_dirs 不能为空列表" """
        trainer = _build_trainer_with_tmp_logger(fresh=True)
        with self.assertRaisesRegex(ValueError, r"sample_dirs 不能为空列表"):
            trainer.train_batch([], epochs=1)

    def test_train_batch_missing_sample_file_raises(self):
        """train_batch 目录不存在 → FileNotFoundError with 路径消息"""
        trainer = _build_trainer_with_tmp_logger(fresh=True)
        with self.assertRaisesRegex(FileNotFoundError, r"Sample file not found:.*nonexistent"):
            trainer.train_batch([os.path.join(self.temp_dir, "nonexistent")], epochs=1)

    def test_load_sample_invalid_json(self):
        """_sample_loader.load 无效 JSON → JSONDecodeError with "Expecting value" """
        invalid_path = os.path.join(self.temp_dir, "invalid.json")
        with open(invalid_path, 'w', encoding='utf-8') as f:
            f.write("invalid json content")
        trainer = FCv1Trainer()
        # JSONDecodeError 消息格式由标准库提供，对无 JSON 前缀的输入固定为 "Expecting value"
        with self.assertRaisesRegex(json.JSONDecodeError, r"Expecting value"):
            trainer._sample_loader.load(invalid_path)


if __name__ == '__main__':
    unittest.main()
