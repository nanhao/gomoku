"""LossLogger log_val() 和 plot() val_loss 测试

测试验证集 loss 的记录和绘图功能，包括向后兼容性。
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.loss_logger import LossLogger


class TestLogVal(unittest.TestCase):
    """测试 log_val() 方法"""

    def test_log_val_writes_entry(self):
        """log_val 写入正确格式的条目"""
        with tempfile.TemporaryDirectory() as tmp_path:
            logger = LossLogger(model_name="test_model", log_dir=tmp_path)
            logger.log_val(200, 0.45)

            log_path = logger.get_log_path()
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            self.assertEqual(len(lines), 1)
            entry = json.loads(lines[0])
            self.assertEqual(entry["epoch"], 200)
            self.assertEqual(entry["val_loss"], 0.45)
            self.assertNotIn("loss", entry)
            self.assertNotIn("grad_norm", entry)

    def test_log_val_sparse_mode(self):
        """稀疏模式使用 epoch 作为 x 轴"""
        with tempfile.TemporaryDirectory() as tmp_path:
            logger = LossLogger(model_name="test_model", log_dir=tmp_path, dense_log=False)
            logger.log_val(400, 0.33)

            log_path = logger.get_log_path()
            with open(log_path, 'r', encoding='utf-8') as f:
                entry = json.loads(f.readline())

            self.assertEqual(entry["epoch"], 400)
            self.assertEqual(entry["val_loss"], 0.33)
            self.assertNotIn("step", entry)

    def test_log_val_dense_mode(self):
        """密集模式使用 step 作为 x 轴"""
        with tempfile.TemporaryDirectory() as tmp_path:
            logger = LossLogger(model_name="test_model", log_dir=tmp_path, dense_log=True)
            logger.log_val(1000, 0.22)

            log_path = logger.get_log_path()
            with open(log_path, 'r', encoding='utf-8') as f:
                entry = json.loads(f.readline())

            self.assertEqual(entry["step"], 1000)
            self.assertEqual(entry["val_loss"], 0.22)
            self.assertNotIn("epoch", entry)


class TestPlotWithValLoss(unittest.TestCase):
    """测试 plot() 在含 val_loss 条目时的行为"""

    @patch('ai.loss_logger.plt')
    def test_plot_with_val_loss(self, mock_plt):
        """含 val_loss 条目时图表包含 Val Loss 折线"""
        # 配置 mock，使 fig, ax1 = plt.subplots() 可正常解包
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax1)
        mock_ax1.twinx.return_value = mock_ax2

        with tempfile.TemporaryDirectory() as tmp_path:
            logger = LossLogger(model_name="test_model", log_dir=tmp_path)
            # 训练 loss（含 grad_norm，触发双 Y 轴路径）
            logger.log(200, 0.5, 1.2)
            logger.log(400, 0.3, 0.8)
            # 验证 loss
            logger.log_val(200, 0.45)
            logger.log_val(400, 0.35)

            logger.plot()

            # ax1.plot 被调用：train loss (b-) 和 val loss (g-)
            call_args_list = mock_ax1.plot.call_args_list
            # 查找绿色 val loss 折线
            val_loss_plotted = False
            for call in call_args_list:
                args = call[0]
                kwargs = call[1] if len(call) > 1 else {}
                color = args[2] if len(args) > 2 else kwargs.get('color', '')
                label = kwargs.get('label', '')
                if 'g' in str(color) or label == 'Val Loss':
                    val_loss_plotted = True
                    break
            self.assertTrue(val_loss_plotted, "应绘制 Val Loss 折线")

    def test_plot_without_val_loss(self):
        """无 val_loss 条目时图表与原逻辑一致"""
        with tempfile.TemporaryDirectory() as tmp_path:
            logger = LossLogger(model_name="test_model", log_dir=tmp_path)
            logger.log(1, 0.5)
            logger.log(200, 0.3)

            img_path = logger.plot()
            self.assertIsNotNone(img_path)
            self.assertTrue(os.path.exists(img_path))

    @patch('ai.loss_logger.plt')
    def test_plot_val_loss_same_x_axis(self, mock_plt):
        """val loss 与 train loss 共享 x 轴（epoch 或 step）"""
        # 配置 mock，使 fig, ax1 = plt.subplots() 可正常解包
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax1)
        mock_ax1.twinx.return_value = mock_ax2

        with tempfile.TemporaryDirectory() as tmp_path:
            # 使用密集模式（含 grad_norm）以触发双 Y 轴路径
            logger = LossLogger(model_name="test_model", log_dir=tmp_path, dense_log=True)
            logger.log_dense(200, 0.5, 1.2)
            logger.flush()
            logger.log_val(200, 0.45)

            logger.plot()

            # 所有 plot 调用应使用相同的 x 轴值（200）
            x_values = []
            for call in mock_ax1.plot.call_args_list:
                args = call[0]
                if args:
                    x_values.append(args[0])
            # train loss 和 val loss 都应在 step=200 处绘制
            self.assertIn([200], x_values)


if __name__ == '__main__':
    unittest.main()
