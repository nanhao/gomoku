"""测试运行器 - 统一调用 pytest

保留此文件作为兼容性入口：
  python tests/run_tests.py

也可直接调用 pytest：
  python -m pytest tests/ -v

注意：默认禁用输出捕获 (-s/--capture=no)，因为 pytest 的捕获机制
在 Windows 上与 Tcl/Tk 存在已知兼容性问题，会导致 GUI 测试间歇性失败。
"""

import sys
import os
import tempfile
import shutil
import atexit

# 兼容性兜底：设置测试专用的 loss 日志目录
# 实际由 tests/conftest.py 接管，此处保留以支持直接 python tests/run_tests.py
_test_loss_log_dir = tempfile.mkdtemp()
os.environ['GOMOKU_LOSS_LOG_DIR'] = _test_loss_log_dir


def _cleanup_test_loss_logs():
    """测试结束后清理临时日志目录"""
    if os.path.exists(_test_loss_log_dir):
        shutil.rmtree(_test_loss_log_dir, ignore_errors=True)


atexit.register(_cleanup_test_loss_logs)

# 获取项目根目录（tests 的父目录）
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)

# 将项目根目录添加到 sys.path 最前面
if parent_dir in sys.path:
    sys.path.remove(parent_dir)
sys.path.insert(0, parent_dir)

# 移除空字符串（当前目录）避免冲突
if '' in sys.path:
    sys.path.remove('')
if '.' in sys.path:
    sys.path.remove('.')


def main():
    """调用 pytest 运行全部测试"""
    import pytest
    # 默认禁用捕获，避免 Windows 上 Tcl/Tk 初始化失败
    # 用户仍可通过追加 --capture=fd 等参数覆盖此默认值
    args = ['tests/', '-v', '-s'] + sys.argv[1:]
    return pytest.main(args)


if __name__ == "__main__":
    sys.exit(main())
