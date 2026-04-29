"""鼠标防休眠脚本（可暂停模式）

持续随机移动鼠标，防止 NVIDIA 独显因空闲而休眠。
使用 SendInput 发送真实鼠标事件，确保驱动能识别为用户活动。

交互:
    空格键  - 切换 暂停/恢复
    Ctrl+C  - 退出

用法:
    python ai/keep_gpu_alive.py
    python ai/keep_gpu_alive.py --min-interval 0.05   # 最快移动间隔
    python ai/keep_gpu_alive.py --max-step 80        # 最大单步移动像素

按 空格键 暂停/恢复，按 Ctrl+C 停止。
"""

import ctypes
import time
import random
import argparse
import threading
from ctypes import wintypes


# ============ Windows API 常量和结构体 ============

MOUSEEVENTF_MOVE = 0x0001
INPUT_MOUSE = 0
VK_SPACE = 0x20  # 空格键虚拟键码

user32 = ctypes.windll.user32


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]

    _fields_ = [
        ("type", wintypes.DWORD),
        ("input", _INPUT),
    ]


def send_mouse_move(dx, dy):
    """发送真实鼠标移动事件（相对移动）

    使用 SendInput API，产生 WM_MOUSEMOVE 消息，
    NVIDIA 驱动能识别为用户活动。

    Args:
        dx: 水平相对移动量（像素）
        dy: 垂直相对移动量（像素）
    """
    extra = ctypes.c_ulong(0)
    ii_ = INPUT()
    ii_.type = INPUT_MOUSE
    ii_.input.mi = MOUSEINPUT(
        dx=dx,
        dy=dy,
        mouseData=0,
        dwFlags=MOUSEEVENTF_MOVE,
        time=0,
        dwExtraInfo=ctypes.pointer(extra),
    )
    user32.SendInput(1, ctypes.byref(ii_), ctypes.sizeof(INPUT))


def get_mouse_pos():
    """获取当前鼠标位置"""
    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
    point = POINT()
    user32.GetCursorPos(ctypes.byref(point))
    return point.x, point.y


class PauseController:
    """空格键暂停/恢复控制器

    在独立线程中持续监听空格键，通过 threading.Event
    与主线程通信。主线程无论在做什么（包括 sleep），
    都不会错过空格键事件。
    """

    def __init__(self):
        # 暂停状态标志
        self.paused = False
        # 用于通知主线程暂停状态变化
        self._toggle_event = threading.Event()
        # 控制监听线程退出
        self._stop_flag = threading.Event()
        # 用于线程安全的暂停状态切换
        self._lock = threading.Lock()

    def start_listener(self):
        """启动键盘监听线程"""
        t = threading.Thread(target=self._listen_loop, daemon=True)
        t.start()

    def stop_listener(self):
        """停止键盘监听线程"""
        self._stop_flag.set()

    def wait_for_toggle(self, timeout=None):
        """阻塞等待暂停状态变化（由键盘线程触发）

        Args:
            timeout: 超时秒数，None 表示无限等待
        Returns:
            True 如果状态发生了变化，False 如果超时
        """
        result = self._toggle_event.wait(timeout=timeout)
        if result:
            self._toggle_event.clear()
        return result

    def is_paused(self):
        """返回当前是否暂停"""
        with self._lock:
            return self.paused

    def _toggle_pause(self):
        """切换暂停状态并通知主线程"""
        with self._lock:
            self.paused = not self.paused
        self._toggle_event.set()

    def _listen_loop(self):
        """键盘监听循环（在独立线程中运行）

        持续轮询空格键状态，检测 按下→松开 的完整动作，
        触发一次暂停/恢复切换。
        """
        was_down = False
        while not self._stop_flag.is_set():
            state = user32.GetAsyncKeyState(VK_SPACE)
            is_down = bool(state & 0x8000)

            # 检测「按下后松开」的完整动作
            if was_down and not is_down:
                self._toggle_pause()

            was_down = is_down
            # 短暂休眠，避免 CPU 空转
            time.sleep(0.01)


def move_mouse_random_step(max_step=50):
    """执行一次随机鼠标小步移动

    在当前位置附近做小幅随机偏移，模拟自然的手部抖动或微移。
    这种小幅移动足以让驱动检测到活动，又不会严重干扰用户操作。

    Args:
        max_step: 最大单步偏移像素数
    """
    dx = random.randint(-max_step, max_step)
    dy = random.randint(-max_step, max_step)

    # 确保不会移出屏幕
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)
    cur_x, cur_y = get_mouse_pos()

    new_x = max(0, min(screen_w - 1, cur_x + dx))
    new_y = max(0, min(screen_h - 1, cur_y + dy))

    actual_dx = new_x - cur_x
    actual_dy = new_y - cur_y

    if actual_dx != 0 or actual_dy != 0:
        send_mouse_move(actual_dx, actual_dy)


def main():
    parser = argparse.ArgumentParser(description="鼠标防休眠（可暂停模式）")
    parser.add_argument(
        "--min-interval", type=float, default=0.08,
        help="最小移动间隔（秒），默认 0.08"
    )
    parser.add_argument(
        "--max-interval", type=float, default=0.5,
        help="最大移动间隔（秒），默认 0.5"
    )
    parser.add_argument(
        "--max-step", type=int, default=50,
        help="最大单步移动像素，默认 50"
    )
    args = parser.parse_args()

    min_interval = args.min_interval
    max_interval = args.max_interval
    max_step = args.max_step

    print("=" * 50)
    print("  鼠标防休眠已启动（可暂停模式）")
    print(f"  移动间隔: {min_interval}~{max_interval} 秒")
    print(f"  最大步幅: {max_step} 像素")
    print("  空格键: 暂停/恢复")
    print("  Ctrl+C: 退出")
    print("=" * 50)

    ctrl = PauseController()
    ctrl.start_listener()
    move_count = 0

    try:
        while True:
            if ctrl.is_paused():
                # 暂停中：阻塞等待空格键恢复，不消耗 CPU
                print("\n  [ 暂停中 ] 按空格键恢复", end="", flush=True)
                ctrl.wait_for_toggle()  # 无限等待，直到空格键按下
                print("\n  [ 运行中 ] 按空格键暂停", flush=True)
            else:
                # 运行中：移动鼠标，用 wait_for_toggle 替代 sleep
                # 这样即使正在等待下次移动，空格键也能立即响应
                move_mouse_random_step(max_step=max_step)
                move_count += 1

                interval = random.uniform(min_interval, max_interval)
                # 等待移动间隔，但如果期间按了空格键立即返回
                toggled = ctrl.wait_for_toggle(timeout=interval)
                if toggled:
                    print("\n  [ 暂停中 ] 按空格键恢复", end="", flush=True)

    except KeyboardInterrupt:
        ctrl.stop_listener()
        status = "暂停中" if ctrl.is_paused() else "运行中"
        print(f"\n\n已停止（共移动 {move_count} 次，退出时状态: {status}）")


if __name__ == "__main__":
    main()
