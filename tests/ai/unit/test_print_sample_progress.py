"""print_sample_progress 函数单元测试

覆盖设计文档 test_cases.md C-01 ~ C-04：
验证进度输出格式: 进度: 样本局 {sample_count}/{target} ({pct}%), 已对弈 {total_games} 局
"""

import unittest
from io import StringIO
from unittest.mock import patch

from ai.create_samples import print_sample_progress


class TestPrintSampleProgress(unittest.TestCase):
    """进度输出格式测试"""

    @patch('sys.stdout', new_callable=StringIO)
    def test_first_10pct(self, mock_stdout):
        """C-01: 首次10%进度"""
        print_sample_progress(10, 100, 23)
        self.assertEqual(
            mock_stdout.getvalue().strip(),
            "进度: 样本局 10/100 (10%), 已对弈 23 局"
        )

    @patch('sys.stdout', new_callable=StringIO)
    def test_50pct(self, mock_stdout):
        """C-02: 50%进度"""
        print_sample_progress(50, 100, 114)
        self.assertEqual(
            mock_stdout.getvalue().strip(),
            "进度: 样本局 50/100 (50%), 已对弈 114 局"
        )

    @patch('sys.stdout', new_callable=StringIO)
    def test_100pct(self, mock_stdout):
        """C-03: 100%达标"""
        print_sample_progress(100, 100, 224)
        self.assertEqual(
            mock_stdout.getvalue().strip(),
            "进度: 样本局 100/100 (100%), 已对弈 224 局"
        )

    @patch('sys.stdout', new_callable=StringIO)
    def test_small_target(self, mock_stdout):
        """C-04: 小值达标"""
        print_sample_progress(5, 5, 5)
        self.assertEqual(
            mock_stdout.getvalue().strip(),
            "进度: 样本局 5/5 (100%), 已对弈 5 局"
        )


if __name__ == "__main__":
    unittest.main()
