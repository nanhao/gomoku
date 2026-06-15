"""pytest 全局配置

在 pytest 收集测试之前执行环境初始化：
- 设置测试专用的 loss 日志目录，避免污染真实数据
- 安装写入保护 hook，禁止测试向受保护目录写入文件
- 配置 sys.path 确保项目根目录可导入
"""

import sys
import os
import tempfile
import shutil
import builtins
import io as _io_module

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


# === 写入保护 Hook ===
# 测试期间禁止向以下目录写入文件，防止测试污染生产数据

_PROJECT_ROOT = _parent_dir

PROTECTED_DIRS = [
    os.path.join(_PROJECT_ROOT, 'ai', 'ai_data'),
    os.path.join(_PROJECT_ROOT, 'ai', 'ai_data', 'loss_logs'),
    os.path.join(_PROJECT_ROOT, 'ai', 'models'),
    os.path.join(_PROJECT_ROOT, 'ai', 'train_data'),
    os.path.join(_PROJECT_ROOT, 'ai', 'merge_samples'),
    os.path.join(_PROJECT_ROOT, 'game', 'save'),
    os.path.join(_PROJECT_ROOT, 'design'),
    os.path.join(_PROJECT_ROOT, 'requirements'),
]

# 保存原始 open 函数
_original_open = builtins.open


def _is_write_mode(mode):
    """判断文件打开模式是否涉及写入操作

    包括：w(截断写入)、a(追加)、x(排他创建)、+(读写模式)
    """
    if not mode:
        return False
    mode_str = str(mode).lower()
    return any(c in mode_str for c in ('w', 'a', 'x', '+'))


def _is_protected_path(filepath):
    """判断文件路径是否在受保护目录下

    使用 os.sep 确保精确的目录前缀匹配，
    避免 /project/ai_data 匹配 /project/ai_data_backup 等情况
    """
    # 文件描述符（int）不可能是路径，直接放行
    if isinstance(filepath, int):
        return False
    try:
        abs_path = os.path.abspath(str(filepath))
        for protected in PROTECTED_DIRS:
            protected_prefix = protected + os.sep
            if abs_path == protected or abs_path.startswith(protected_prefix):
                return True
        return False
    except (TypeError, ValueError):
        return False


def _guarded_open(file, mode='r', *args, **kwargs):
    """带写入保护的 open() 包装器

    在 pytest 运行期间拦截所有文件写入操作，
    禁止向受保护目录写入文件，防止测试污染生产数据。
    允许读取受保护目录中的文件。
    """
    if _is_write_mode(mode) and _is_protected_path(file):
        raise AssertionError(
            f"[WRITE GUARD] 测试试图写入受保护目录: {os.path.abspath(str(file))}\n"
            f"受保护目录列表:\n"
            + '\n'.join(f'  - {d}' for d in PROTECTED_DIRS)
            + "\n如需写入，请使用 tempfile 或 tmp_path fixture"
        )
    return _original_open(file, mode, *args, **kwargs)


def pytest_configure(config):
    """安装写入保护 hook：替换 builtins.open 和 io.open"""
    builtins.open = _guarded_open
    _io_module.open = _guarded_open
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


def pytest_unconfigure(config):
    """卸载写入保护 hook：恢复原始 open 函数"""
    builtins.open = _original_open
    _io_module.open = _original_open
