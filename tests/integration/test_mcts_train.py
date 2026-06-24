"""MCTS 训练集成测试

测试训练流水线：样本加载 → 训练 → 保存。
"""

import sys
import os
import unittest
import tempfile
import shutil
import json

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch


class TestMCTSTrainingIntegration(unittest.TestCase):
    """训练集成测试"""

    @classmethod
    def setUpClass(cls):
        """导入模块"""
        try:
            from ai.dual_head_trainer import DualHeadTrainer
            cls.DualHeadTrainer = DualHeadTrainer
        except ImportError:
            cls.DualHeadTrainer = None
        try:
            from ai.models.cnn_v7.trainer import CNNV7Trainer
            cls.CNNV7Trainer = CNNV7Trainer
        except ImportError:
            cls.CNNV7Trainer = None

    def setUp(self):
        if self.DualHeadTrainer is None:
            self.skipTest("DualHeadTrainer 尚未实现")
        self.temp_dir = tempfile.mkdtemp(prefix="mcts_train_test_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @staticmethod
    def _create_sample_data(output_dir, n_samples=20):
        """创建测试用 sample.json"""
        samples = []
        for i in range(n_samples):
            board = [0] * 225
            # 放一些棋子
            board[i % 225] = 1 if i % 3 != 0 else 2
            board[(i * 7 + 11) % 225] = 2 if i % 3 == 0 else 1
            state = board + [1 if i % 2 == 0 else 2]
            pi = [0.0] * 225
            # 给合法位置概率
            empty_positions = [j for j, v in enumerate(board) if v == 0]
            if empty_positions:
                prob = 1.0 / len(empty_positions)
                for pos in empty_positions:
                    pi[pos] = prob
            else:
                pi = [1.0 / 225] * 225
            z = 1 if i % 3 == 0 else (-1 if i % 3 == 1 else 0)
            samples.append({"state": state, "pi": pi, "z": z})

        os.makedirs(output_dir, exist_ok=True)
        sample_path = os.path.join(output_dir, "sample.json")
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f)
        return sample_path

    def test_cnn_v7_trainer_inherits_dual_head(self):
        """CNNV7Trainer 应继承自 DualHeadTrainer"""
        if self.CNNV7Trainer is None:
            self.skipTest("CNNV7Trainer 尚未实现")
        from ai.dual_head_trainer import DualHeadTrainer
        self.assertTrue(issubclass(self.CNNV7Trainer, DualHeadTrainer))

    def test_convert_inputs_relative_encoding(self):
        """_convert_inputs 应使用相对编码"""
        if self.CNNV7Trainer is None:
            self.skipTest("CNNV7Trainer 尚未实现")
        from ai.base_model import board_vector_to_cnn_tensor_relative

        trainer = self.CNNV7Trainer(fresh=True)
        # 创建一些测试 input
        test_inputs = []
        for i in range(4):
            board = [0] * 225
            board[112] = 1
            board[0] = 2
            test_inputs.append(board + [1 if i % 2 == 0 else 2])

        tensor = trainer._convert_inputs(test_inputs)
        self.assertEqual(tensor.shape[0], 4)
        self.assertEqual(tensor.shape[1], 3)
        self.assertEqual(tensor.shape[2], 15)
        self.assertEqual(tensor.shape[3], 15)

    def test_end_to_end_training(self):
        """端到端训练流程"""
        if self.CNNV7Trainer is None:
            self.skipTest("CNNV7Trainer 尚未实现")

        self._create_sample_data(self.temp_dir, n_samples=16)

        trainer = self.CNNV7Trainer(fresh=True)

        # 训练 2 个 epoch
        if hasattr(trainer, 'train_from_dir'):
            result = trainer.train_from_dir(
                train_from_dir=self.temp_dir,
                epochs=2,
                batch_size=4,
                val_split=0.0,
            )
            self.assertIsNotNone(result)
        elif hasattr(trainer, 'train'):
            # 备用：直接调用 train
            result = trainer.train(
                train_dirs=[self.temp_dir],
                epochs=2,
                batch_size=4,
                val_split=0.0,
            )
            self.assertIsNotNone(result)

    def test_model_save_after_training(self):
        """训练后模型保存"""
        if self.CNNV7Trainer is None:
            self.skipTest("CNNV7Trainer 尚未实现")

        self._create_sample_data(self.temp_dir, n_samples=8)

        trainer = self.CNNV7Trainer(fresh=True)

        # 训练 1 个 epoch
        if hasattr(trainer, 'train_from_dir'):
            trainer.train_from_dir(
                train_from_dir=self.temp_dir,
                epochs=1,
                batch_size=4,
                val_split=0.0,
            )

        output_path = os.path.join(self.temp_dir, "trained.pth")
        trainer.model.save(output_path)
        self.assertTrue(os.path.exists(output_path))


if __name__ == "__main__":
    unittest.main()
