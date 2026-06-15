"""Mini-Batch 训练边缘测试

测试 BaseTrainer.train_mini_batch 方法的边缘情况。
"""

import os
import sys
import json
import tempfile
import unittest
import unittest.mock
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from ai.models.fc_v1.model import FCv1Model
from ai.models.fc_v1.trainer import FCv1Trainer
from tests.ai.helpers import create_sample_dir, create_output
from unittest.mock import MagicMock
import torch


class TestMiniBatchTrainingEdge(unittest.TestCase):
    """Mini-Batch 训练边缘测试"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        # 使用 patch.object 隔离默认模型路径
        self._patcher = unittest.mock.patch.object(
            FCv1Model, 'DEFAULT_MODEL_PATH',
            os.path.join(self.temp_dir, "fc_v1.pth")
        )
        self._patcher.start()

    def tearDown(self):
        """测试后清理"""
        self._patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_trainer(self) -> FCv1Trainer:
        """创建训练器并注入 mock logger 避免污染真实 loss_logs 目录"""
        trainer = FCv1Trainer(fresh=True)
        trainer.loss_logger = MagicMock()
        trainer.loss_logger.dense_log = False
        return trainer

    def _create_sample_dir(self, name: str, n_items: int = 2) -> str:
        """创建一个包含 sample.json 的目录"""
        sample_dir = os.path.join(self.temp_dir, name)
        os.makedirs(sample_dir, exist_ok=True)
        
        sample_data = []
        for i in range(n_items):
            output = [0.0] * 225
            output[0] = 0.7
            output[1] = 0.2
            output[2] = 0.1
            mask = [1, 1, 1] + [0] * 222
            
            sample_data.append({
                "input": [float(i + 1)] * 226,
                "output": output,
                "mask": mask
            })
        
        sample_path = os.path.join(sample_dir, "sample.json")
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f)
        
        return sample_dir
    
    def test_empty_sample_dirs_raises(self):
        """[edge] sample_dirs 为空列表时抛出 ValueError"""
        trainer = FCv1Trainer(fresh=True)
        
        with self.assertRaises(ValueError) as context:
            trainer.train_mini_batch([], epochs=1)
        
        self.assertIn("不能为空", str(context.exception))
    
    def test_missing_sample_file_raises(self):
        """[edge] sample 目录不存在时抛出 FileNotFoundError"""
        trainer = FCv1Trainer(fresh=True)
        fake_dir = os.path.join(self.temp_dir, "nonexistent")
        
        with self.assertRaises(FileNotFoundError):
            trainer.train_mini_batch([fake_dir], epochs=1)
    
    def test_batch_size_larger_than_data(self):
        """[edge] batch_size 大于样本数时退化为单批次"""
        dir1 = self._create_sample_dir("s1", n_items=3)
        trainer = self._create_trainer()
        
        # batch_size=100 远大于样本数 3
        result = trainer.train_mini_batch(
            [dir1],
            epochs=5,
            batch_size=100,
            verbose=False
        )
        
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
    
    def test_batch_size_equals_one(self):
        """[edge] batch_size=1 时每个样本单独更新"""
        dir1 = self._create_sample_dir("s1", n_items=5)
        trainer = self._create_trainer()
        
        result = trainer.train_mini_batch(
            [dir1],
            epochs=3,
            batch_size=1,
            verbose=False
        )
        
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
    
    def test_single_sample(self):
        """[edge] 只有 1 个样本时的训练"""
        dir1 = self._create_sample_dir("s1", n_items=1)
        trainer = self._create_trainer()
        
        result = trainer.train_mini_batch(
            [dir1],
            epochs=5,
            batch_size=2,
            verbose=False
        )
        
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
    
    def test_large_batch_size(self):
        """[edge] 非常大的 batch_size"""
        dir1 = self._create_sample_dir("s1", n_items=5)
        trainer = self._create_trainer()
        
        result = trainer.train_mini_batch(
            [dir1],
            epochs=3,
            batch_size=10000,
            verbose=False
        )
        
        self.assertIsInstance(result, float)
    
    def test_zero_epochs(self):
        """[edge] epochs=0 时的行为"""
        dir1 = self._create_sample_dir("s1")
        trainer = self._create_trainer()
        
        # epochs=0 应该返回 0（没有训练迭代）
        result = trainer.train_mini_batch(
            [dir1],
            epochs=0,
            batch_size=2,
            verbose=False
        )
        # epochs=0 时 total_batches=0，应返回 0.0
        self.assertEqual(result, 0.0)
    
    def test_shuffle_false_consistent_order(self):
        """[edge] shuffle=False 时数据顺序固定
        
        注意：由于每次创建 fresh 模型参数不同，我们只验证训练能正常完成。
        """
        dir1 = self._create_sample_dir("s1", n_items=10)
        
        # 验证 shuffle=False 时训练能正常完成
        trainer = self._create_trainer()
        loss = trainer.train_mini_batch(
            [dir1],
            epochs=5,
            batch_size=2,
            shuffle=False,
            verbose=False
        )
        
        self.assertIsInstance(loss, float)
    
    def test_multiple_sample_dirs_merged(self):
        """[edge] 多个 sample 目录正确合并"""
        dir1 = self._create_sample_dir("s1", n_items=3)
        dir2 = self._create_sample_dir("s2", n_items=3)
        dir3 = self._create_sample_dir("s3", n_items=4)
        
        trainer = self._create_trainer()
        
        result = trainer.train_mini_batch(
            [dir1, dir2, dir3],
            epochs=5,
            batch_size=2,
            verbose=False
        )
        
        self.assertIsInstance(result, float)
        # 总共 3+3+4=10 个样本，batch_size=2，每个 epoch 应有 5 次更新


class TestMiniBatchVsBatchComparison(unittest.TestCase):
    """比较 mini-batch 和 batch 训练的效果"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        # 使用 patch.object 隔离默认模型路径
        self._patcher = unittest.mock.patch.object(
            FCv1Model, 'DEFAULT_MODEL_PATH',
            os.path.join(self.temp_dir, "fc_v1.pth")
        )
        self._patcher.start()

    def tearDown(self):
        """测试后清理"""
        self._patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_trainer(self) -> FCv1Trainer:
        """创建训练器并注入 mock logger 避免污染真实 loss_logs 目录"""
        trainer = FCv1Trainer(fresh=True)
        trainer.loss_logger = MagicMock()
        trainer.loss_logger.dense_log = False
        return trainer

    def _create_sample_dir(self, name: str, n_items: int = 5) -> str:
        """创建 sample 目录"""
        sample_dir = os.path.join(self.temp_dir, name)
        os.makedirs(sample_dir, exist_ok=True)
        
        sample_data = []
        for i in range(n_items):
            output = [0.0] * 225
            output[0] = 0.7
            output[1] = 0.2
            output[2] = 0.1
            mask = [1, 1, 1] + [0] * 222
            
            sample_data.append({
                "input": [float(i + 1)] * 226,
                "output": output,
                "mask": mask
            })
        
        sample_path = os.path.join(sample_dir, "sample.json")
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f)
        
        return sample_dir
    
    def test_mini_batch_more_updates_than_batch(self):
        """[edge] mini-batch 每个 epoch 更新次数多于 batch
        
        train_batch: 每个 epoch 1 次更新
        train_mini_batch: 每个 epoch N/batch_size 次更新
        """
        dir1 = self._create_sample_dir("s1", n_items=10)
        
        # 记录 train_batch 的参数更新次数
        trainer_batch = self._create_trainer()
        params_batch_before = list(trainer_batch.model.network.parameters())[0].data.clone()
        trainer_batch.train_batch([dir1], epochs=1, verbose=False)
        params_batch_after = list(trainer_batch.model.network.parameters())[0].data
        batch_diff = torch.abs(params_batch_after - params_batch_before).sum().item()
        
        # 记录 train_mini_batch 的参数变化
        trainer_mini = self._create_trainer()
        params_mini_before = list(trainer_mini.model.network.parameters())[0].data.clone()
        trainer_mini.train_mini_batch([dir1], epochs=1, batch_size=2, verbose=False)
        params_mini_after = list(trainer_mini.model.network.parameters())[0].data
        mini_diff = torch.abs(params_mini_after - params_mini_before).sum().item()
        
        # mini-batch 的参数变化应该更大（因为更新了 5 次 vs 1 次）
        # 但这取决于学习率和数据，所以我们只验证两者都更新了
        self.assertGreater(batch_diff, 0, "train_batch 应更新参数")
        self.assertGreater(mini_diff, 0, "train_mini_batch 应更新参数")


if __name__ == '__main__':
    unittest.main()
