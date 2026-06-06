"""CNN 训练器参数化测试（v1/v2/v3 统一）

按 task_reports/2026-05-01/test_suite_review.md Top #5 执行：
合并原三个文件（test_cnn_trainer.py / test_cnn_v2_trainer.py / test_cnn_v3_trainer.py）
为 Mixin + 三子类结构。

注：原 test_cnn_trainer.py::test_trainer_compute_loss 实为对 compute_masked_kl_loss 的测试，
与 tests/ai/test_sample_loader.py 中同名用例重复，未纳入 Mixin。
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
from unittest.mock import MagicMock

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch

from ai.models.cnn_v1.trainer import CNNV1Trainer
from ai.models.cnn_v1.model import CNNV1Model
from ai.models.cnn_v2.trainer import CNNV2Trainer
from ai.models.cnn_v2.model import CNNV2Model
from ai.models.cnn_v3.trainer import CNNV3Trainer
from ai.models.cnn_v3.model import CNNV3Model
from ai.base_trainer import BaseTrainer


def _make_output(x: int, y: int, reward: int) -> list:
    """创建 225 维输出向量，索引 i 对应位置 (i//15, i%15)"""
    output = [0] * 225
    output[x * 15 + y] = reward
    return output


def _write_test_sample(sample_dir: str) -> None:
    """在 sample_dir 下写入标准双样本 sample.json"""
    sample_data = [
        {
            "input": [0] * 225 + [1],
            "output": _make_output(0, 0, 1000000),
            "mask": [1] + [0] * 224,
        },
        {
            "input": [1] + [0] * 224 + [2],
            "output": _make_output(1, 0, 500000),
            "mask": [0] * 15 + [1] + [0] * 209,
        },
    ]
    with open(os.path.join(sample_dir, "sample.json"), 'w', encoding='utf-8') as f:
        json.dump(sample_data, f)


# ---------------------------------------------------------------------------
# Mixin：所有 CNN Trainer 共有的行为
# ---------------------------------------------------------------------------

class CNNTrainerSharedMixin:
    """所有 CNN Trainer 的共性行为。

    子类需设置类属性：TRAINER_CLS, MODEL_CLS, MODEL_NAME, DEFAULT_FILENAME
    """

    TRAINER_CLS = None
    MODEL_CLS = None
    MODEL_NAME = None
    DEFAULT_FILENAME = None

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_dir = os.path.join(self.temp_dir, "sample_data")
        os.makedirs(self.sample_dir, exist_ok=True)

        self._original_path = self.MODEL_CLS.DEFAULT_MODEL_PATH
        self.MODEL_CLS.DEFAULT_MODEL_PATH = os.path.join(self.temp_dir, self.DEFAULT_FILENAME)

        _write_test_sample(self.sample_dir)

    def tearDown(self):
        self.MODEL_CLS.DEFAULT_MODEL_PATH = self._original_path
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # --- 初始化 ---

    def test_trainer_initialization(self):
        """不传入模型时默认创建"""
        trainer = self.TRAINER_CLS()
        self.assertIsNotNone(trainer.model)
        self.assertEqual(trainer.model.name, self.MODEL_NAME)

    def test_trainer_initialization_with_model(self):
        """显式传入模型"""
        model = self.MODEL_CLS()
        trainer = self.TRAINER_CLS(model)
        self.assertIs(trainer.model, model)

    def test_trainer_model_name(self):
        """model_name 属性"""
        trainer = self.TRAINER_CLS()
        self.assertEqual(trainer.model_name, self.MODEL_NAME)

    def test_trainer_fresh_parameter(self):
        """fresh=True 创建新模型"""
        trainer = self.TRAINER_CLS(fresh=True)
        self.assertIsInstance(trainer.model, self.MODEL_CLS)

    def test_trainer_inherits_base_trainer(self):
        """继承 BaseTrainer"""
        self.assertTrue(issubclass(self.TRAINER_CLS, BaseTrainer))

    # --- 单轮训练 ---

    def test_trainer_train_single(self):
        """单轮训练返回非负 float"""
        trainer = self.TRAINER_CLS()
        trainer.loss_logger = MagicMock()
        trainer.loss_logger.dense_log = False

        loss = trainer.train_single(self.sample_dir)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)

    def test_trainer_train_batch(self):
        """批量训练返回非负 float"""
        trainer = self.TRAINER_CLS()
        trainer.loss_logger = MagicMock()
        trainer.loss_logger.dense_log = False

        # 再造第二个 sample 目录
        sample_dir2 = os.path.join(self.temp_dir, "sample_data2")
        os.makedirs(sample_dir2, exist_ok=True)
        with open(os.path.join(sample_dir2, "sample.json"), 'w', encoding='utf-8') as f:
            json.dump([{
                "input": [0] * 225 + [1],
                "output": _make_output(2, 2, 1000000),
                "mask": [1] + [0] * 224,
            }], f)

        loss = trainer.train_batch([self.sample_dir, sample_dir2], epochs=5)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)

    # --- 数据加载 / 张量准备 ---

    def test_trainer_load_sample_data(self):
        """_sample_loader.load 返回含 input/output/mask 的列表"""
        trainer = self.TRAINER_CLS()
        data = trainer._sample_loader.load(os.path.join(self.sample_dir, "sample.json"))
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)
        for key in ("input", "output", "mask"):
            self.assertIn(key, data[0])

    def test_trainer_prepare_tensors(self):
        """prepare_tensors 输出 CNN 专用的 4D 输入"""
        trainer = self.TRAINER_CLS()
        sample_data = trainer._sample_loader.load(
            os.path.join(self.sample_dir, "sample.json")
        )
        inputs, targets, masks = trainer._sample_loader.prepare_tensors(sample_data)

        self.assertEqual(inputs.shape, (2, 3, 15, 15))
        self.assertEqual(targets.shape, (2, 225))
        self.assertEqual(masks.shape, (2, 225))
        self.assertIsInstance(inputs, torch.Tensor)
        self.assertIsInstance(targets, torch.Tensor)
        self.assertIsInstance(masks, torch.Tensor)

    # --- 边缘场景 ---

    def test_train_single_with_nonexistent_sample(self):
        """不存在的 sample 目录抛 FileNotFoundError"""
        trainer = self.TRAINER_CLS()
        with self.assertRaises(FileNotFoundError):
            trainer.train_single(os.path.join(self.temp_dir, "nonexistent"))

    def test_train_batch_with_empty_list(self):
        """空列表抛 ValueError"""
        trainer = self.TRAINER_CLS()
        with self.assertRaises(ValueError) as ctx:
            trainer.train_batch([])
        self.assertIn("不能为空", str(ctx.exception))

    def test_load_sample_data_with_invalid_json(self):
        """无效 JSON 抛 JSONDecodeError"""
        trainer = self.TRAINER_CLS()
        invalid_path = os.path.join(self.temp_dir, "invalid.json")
        with open(invalid_path, 'w', encoding='utf-8') as f:
            f.write("invalid json content")
        with self.assertRaises(json.JSONDecodeError):
            trainer._sample_loader.load(invalid_path)


# ---------------------------------------------------------------------------
# 三代具体子类
# ---------------------------------------------------------------------------

class TestCNNV1Trainer(CNNTrainerSharedMixin, unittest.TestCase):
    TRAINER_CLS = CNNV1Trainer
    MODEL_CLS = CNNV1Model
    MODEL_NAME = "cnn_v1"
    DEFAULT_FILENAME = "cnn_v1.pth"


class TestCNNV2Trainer(CNNTrainerSharedMixin, unittest.TestCase):
    TRAINER_CLS = CNNV2Trainer
    MODEL_CLS = CNNV2Model
    MODEL_NAME = "cnn_v2"
    DEFAULT_FILENAME = "cnn_v2.pth"


class TestCNNV3Trainer(CNNTrainerSharedMixin, unittest.TestCase):
    TRAINER_CLS = CNNV3Trainer
    MODEL_CLS = CNNV3Model
    MODEL_NAME = "cnn_v3"
    DEFAULT_FILENAME = "cnn_v3.pth"


if __name__ == '__main__':
    unittest.main()
