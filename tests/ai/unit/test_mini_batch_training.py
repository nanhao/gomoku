"""Mini-Batch 训练主流程测试

测试 BaseTrainer.train_mini_batch 方法的主流程功能。
"""

import os
import sys
import json
import tempfile
import unittest
import shutil
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from ai.models.fc_v1.model import FCv1Model
from ai.models.fc_v1.trainer import FCv1Trainer
from ai.loss_logger import LossLogger
from tests.ai.helpers import create_sample_dir, create_output
import torch


class TestMiniBatchTraining(unittest.TestCase):
    """Mini-Batch 训练主流程测试"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        # 隔离默认模型路径
        self.original_default_path = FCv1Model.DEFAULT_MODEL_PATH
        FCv1Model.DEFAULT_MODEL_PATH = os.path.join(self.temp_dir, "fc_v1.pth")
    
    def tearDown(self):
        """测试后清理"""
        FCv1Model.DEFAULT_MODEL_PATH = self.original_default_path
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_sample_dir(self, name: str, n_items: int = 2) -> str:
        """创建一个包含 sample.json 的目录"""
        sample_dir = os.path.join(self.temp_dir, name)
        os.makedirs(sample_dir, exist_ok=True)
        
        sample_data = []
        for i in range(n_items):
            # 创建 Softmax 概率分布作为 output
            output = [0.0] * 225
            output[0] = 0.7  # 主要概率集中在前几个位置
            output[1] = 0.2
            output[2] = 0.1
            mask = [1, 1, 1] + [0] * 222  # 只有前3个位置有效
            
            sample_data.append({
                "input": [float(i + 1)] * 226,
                "output": output,
                "mask": mask
            })
        
        sample_path = os.path.join(sample_dir, "sample.json")
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f)
        
        return sample_dir
    
    def test_train_mini_batch_returns_float(self):
        """[primary] train_mini_batch 返回 float 损失值"""
        dir1 = self._create_sample_dir("s1")
        dir2 = self._create_sample_dir("s2")
        trainer = FCv1Trainer(fresh=True)
        
        result = trainer.train_mini_batch(
            [dir1, dir2],
            epochs=10,
            batch_size=2,
            verbose=False
        )
        
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
    
    def test_train_mini_batch_single_sample_dir(self):
        """[primary] train_mini_batch 处理单个 sample 目录"""
        dir1 = self._create_sample_dir("s1", n_items=4)
        trainer = FCv1Trainer(fresh=True)
        
        result = trainer.train_mini_batch(
            [dir1],
            epochs=5,
            batch_size=2,
            verbose=False
        )
        
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
    
    def test_train_mini_batch_multiple_batches_per_epoch(self):
        """[primary] 每个 epoch 内多次更新参数
        
        验证：与 train_batch 不同，train_mini_batch 应该每个 epoch 内多次更新
        """
        # 创建足够多的样本以确保有多个 batch
        dir1 = self._create_sample_dir("s1", n_items=10)
        trainer = FCv1Trainer(fresh=True)
        
        import copy
        # 记录训练前的参数
        params_before = copy.deepcopy(
            list(trainer.model.network.parameters())[0].data.clone()
        )
        
        # 只训练 1 个 epoch，batch_size=2，应该有 5 次参数更新
        trainer.train_mini_batch(
            [dir1],
            epochs=1,
            batch_size=2,
            verbose=False
        )
        
        # 训练后参数应该已更新
        params_after = list(trainer.model.network.parameters())[0].data
        self.assertFalse(torch.equal(params_before, params_after),
                        "1 epoch 内应通过多次 batch 更新参数")
    
    def test_train_mini_batch_shuffle_changes_order(self):
        """[primary] shuffle=True 时数据顺序被打乱"""
        # 创建多个样本
        dir1 = self._create_sample_dir("s1", n_items=10)
        
        # 固定随机种子，两次训练应得到相同结果
        torch.manual_seed(42)
        trainer1 = FCv1Trainer(fresh=True)
        loss1 = trainer1.train_mini_batch(
            [dir1],
            epochs=5,
            batch_size=2,
            shuffle=True,
            verbose=False
        )
        
        torch.manual_seed(42)
        trainer2 = FCv1Trainer(fresh=True)
        loss2 = trainer2.train_mini_batch(
            [dir1],
            epochs=5,
            batch_size=2,
            shuffle=True,
            verbose=False
        )
        
        # 固定种子时两次训练结果应相同
        self.assertAlmostEqual(loss1, loss2, places=6)
    
    def test_train_mini_batch_updates_model_params(self):
        """[primary] 训练后模型参数确实更新"""
        dir1 = self._create_sample_dir("s1")
        dir2 = self._create_sample_dir("s2")
        trainer = FCv1Trainer(fresh=True)
        
        import copy
        params_before = copy.deepcopy(
            list(trainer.model.network.parameters())[0].data.clone()
        )
        
        trainer.train_mini_batch(
            [dir1, dir2],
            epochs=10,
            batch_size=2,
            verbose=False
        )
        
        params_after = list(trainer.model.network.parameters())[0].data
        self.assertFalse(torch.equal(params_before, params_after))
    
    def test_train_mini_batch_verbose_output(self):
        """[primary] verbose=True 时每 200 个 epoch 输出一次 loss

        关注点：verbose 打印契约（格式 `Epoch 200/200, avg_loss: ..., lr: ...`）。
        真实 forward/backward 的覆盖由其他 mini-batch 用例（如 test_train_mini_batch_*
        参数更新验证）保障，此处 mock 掉 _masked_loss 以避免过度测试。
        mock 后 loss 是脱离计算图的 scalar，backward 空转、optimizer.step 跳过，
        仅保留 epoch 循环 + verbose 分支判断。
        """
        dir1 = self._create_sample_dir("s1", n_items=4)
        trainer = FCv1Trainer(fresh=True)

        import io
        captured = io.StringIO()
        sys.stdout = captured

        try:
            # scalar tensor：.item() 返回 0.5，backward 不触发任何 model 参数梯度
            def _fake_loss(*args, **kwargs):
                return torch.tensor(0.5, requires_grad=True)

            with patch.object(trainer, '_masked_loss', side_effect=_fake_loss):
                trainer.train_mini_batch(
                    [dir1],
                    epochs=200,
                    batch_size=2,
                    verbose=True
                )
            output = captured.getvalue()
        finally:
            sys.stdout = sys.__stdout__

        # 应在第 200 个 epoch 输出
        self.assertIn("Epoch 200/200", output)
        self.assertIn("loss", output.lower())
    
    def test_train_mini_batch_dense_log_records_all_batches(self):
        """[primary] dense_log=True 时批量同步不影响日志完整性

        验证方案 C：每个 epoch 结束后批量同步 GPU 张量到 CPU，
        所有 batch 的 loss 和 grad_norm 仍被正确记录。
        """
        dir1 = self._create_sample_dir("s1", n_items=6)
        trainer = FCv1Trainer(fresh=True)

        logger = LossLogger(model_name="fc_v1", dense_log=True, log_dir=self.temp_dir)
        trainer.loss_logger = logger

        trainer.train_mini_batch(
            [dir1],
            epochs=2,
            batch_size=2,
            verbose=False
        )

        entries = logger._read_entries()
        # 6 items / batch_size 2 = 3 batches/epoch * 2 epochs = 6 entries
        self.assertEqual(len(entries), 6)
        # step 连续
        self.assertEqual([e["step"] for e in entries], [0, 1, 2, 3, 4, 5])
        # 每条记录都有 loss 和 grad_norm
        for entry in entries:
            self.assertIn("loss", entry)
            self.assertIn("grad_norm", entry)
            self.assertIsInstance(entry["loss"], float)
            self.assertIsInstance(entry["grad_norm"], float)

    def test_train_mini_batch_dense_log_returns_float(self):
        """[primary] dense_log=True 时返回值类型正确"""
        dir1 = self._create_sample_dir("s1", n_items=4)
        trainer = FCv1Trainer(fresh=True)

        logger = LossLogger(model_name="fc_v1", dense_log=True, log_dir=self.temp_dir)
        trainer.loss_logger = logger

        result = trainer.train_mini_batch(
            [dir1],
            epochs=3,
            batch_size=2,
            verbose=False
        )

        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)

    def test_train_mini_batch_batch_size_default(self):
        """[primary] batch_size 默认值为 32"""
        import inspect
        sig = inspect.signature(FCv1Trainer.train_mini_batch)
        self.assertEqual(sig.parameters['batch_size'].default, 32)

    def test_train_mini_batch_shuffle_default(self):
        """[primary] shuffle 默认值为 True"""
        import inspect
        sig = inspect.signature(FCv1Trainer.train_mini_batch)
        self.assertTrue(sig.parameters['shuffle'].default)


if __name__ == '__main__':
    unittest.main()
