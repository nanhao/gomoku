"""GUI 测试共享 conftest

根治 Windows + Python 3.13 下反复 tk.Tk() 导致 Tcl 解释器偶发退化的问题：
症状 `couldn't read file "init.tcl": No error`，随机漂移、非确定性失败。

策略：
1. 在 conftest 模块导入时（早于任何 TestCase.setUpClass 和 pytest fixture）立即
   创建唯一真实 Tk root，并把 tkinter.Tk 全局替换为返回 Toplevel(shared_root) 的工厂。
2. session 结束时尝试销毁 shared root（失败也无所谓，进程要退了）。

效果：
- 整个 gui 测试过程只初始化一次 Tcl 解释器（一次 tk.Tk()）
- 所有测试代码 / 生产代码中的 tk.Tk() 调用都被透明重定向为 Toplevel
- Toplevel 支持 title / geometry / configure / withdraw / destroy 等测试涉及的全部 API

影响范围：仅限 tests/gui/ 目录下的测试（conftest.py 的目录作用域）。
"""
import tkinter as _tk

import pytest


# ---------------------------------------------------------------------------
# 模块导入时立即完成 Tk 替换：必须早于任何 unittest.TestCase.setUpClass 执行
# ---------------------------------------------------------------------------
_GUI_SHARED_ROOT = _tk.Tk()
_GUI_SHARED_ROOT.withdraw()

# 保留原始 Tk 类，以防极少数测试需要真实 Tk 时显式恢复使用
_ORIGINAL_TK = _tk.Tk


def _toplevel_factory(*args, **kwargs):
    """伪装成 tk.Tk()：每次调用返回一个新 Toplevel，复用全局 Tcl 解释器。

    不接受 args/kwargs 的语义校验（Tk() 常用不带参数），多余参数一律忽略。
    """
    return _tk.Toplevel(_GUI_SHARED_ROOT)


# 全局替换 tk.Tk，所有测试 / 被测代码的 `tk.Tk()` 调用都命中此工厂
_tk.Tk = _toplevel_factory


@pytest.fixture(scope="session", autouse=True)
def _gui_session_cleanup():
    """session 结束时销毁共享 root，尽力而为"""
    yield
    try:
        _GUI_SHARED_ROOT.destroy()
    except Exception:
        pass
