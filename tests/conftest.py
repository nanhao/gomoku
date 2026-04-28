"""pytest 全局配置

在 pytest 收集测试之前执行环境初始化：
- 设置测试专用的 loss 日志目录，避免污染真实数据
- 配置 sys.path 确保项目根目录可导入
"""

import sys
import os
import tempfile
import shutil

# 设置测试专用的 loss 日志目录，避免污染真实数据
# 必须在导入任何 ai 模块之前设置，因为 ai/constants.py 在导入时就读取此环境变量
_test_loss_log_dir = tempfile.mkdtemp()
os.environ['GOMOKU_LOSS_LOG_DIR'] = _test_loss_log_dir


def pytest_sessionfinish(session, exitstatus):
    """测试会话结束后清理临时日志目录"""
    if os.path.exists(_test_loss_log_dir):
        shutil.rmtree(_test_loss_log_dir, ignore_errors=True)


# 获取项目根目录（tests 的父目录）
_current_dir = os.path.abspath(os.path.dirname(__file__))
_parent_dir = os.path.dirname(_current_dir)

# 将项目根目录添加到 sys.path 最前面
if _parent_dir in sys.path:
    sys.path.remove(_parent_dir)
sys.path.insert(0, _parent_dir)

# 移除空字符串（当前目录）避免冲突
if '' in sys.path:
    sys.path.remove('')
if '.' in sys.path:
    sys.path.remove('.')
