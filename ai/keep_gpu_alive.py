"""鼠标防休眠脚本

每隔 10 秒进行一次鼠标大范围随机移动，持续 5 秒，
防止 NVIDIA 独显因空闲而休眠。
使用 SendInput 发送真实鼠标事件，确保驱动能识别为用户活动。

用法:
    python temp/keep_gpu_alive.py
    python temp/keep_gpu_alive.py --interval 5       # 每5秒触发一次
    python temp/keep_gpu_alive.py --move-duration 3  # 每次移动持续3秒

按 Ctrl+C 停止。
"""

import ctypes
import time
import random
import argparse
from ctypes import wintypes


# ============ Windows API 常量和结构体 ============

MOUSEEVENTF_MOVE = 0x0001
INPUT_MOUSE = 0

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


def move_mouse_to(target_x, target_y, steps=50, total_delay=0.5):
    """平滑移动鼠标到目标位置（使用 SendInput）

    Args:
        target_x: 目标 X 坐标
        target_y: 目标 Y 坐标
        steps: 移动步数
        total_delay: 总移动时间（秒）
    """
    start_x, start_y = get_mouse_pos()
    dx_total = target_x - start_x
    dy_total = target_y - start_y
    step_delay = total_delay / steps

    for step in range(1, steps + 1):
        t = step / steps
        t_ease = t * t * (3 - 2 * t)  # smoothstep 缓动
        new_x = start_x + dx_total * t_ease
        new_y = start_y + dy_total * t_ease

        # 计算相对当前位置的偏移量
        cur_x, cur_y = get_mouse_pos()
        rel_dx = int(new_x) - cur_x
        rel_dy = int(new_y) - cur_y

        if rel_dx != 0 or rel_dy != 0:
            send_mouse_move(rel_dx, rel_dy)

        time.sleep(step_delay)


def move_mouse_randomly(duration: float = 5.0):
    """大范围随机移动鼠标，持续指定秒数

    在屏幕范围内随机选择目标点，平滑移动过去，
    使用真实鼠标事件，模拟用户的大范围鼠标操作。

    Args:
        duration: 移动持续时间（秒）
    """
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)
    margin = 100

    # 随机选择 3-6 个目标点，依次移动过去
    num_targets = random.randint(3, 6)
    segment_duration = duration / num_targets

    for i in range(num_targets):
        target_x = random.randint(margin, screen_w - margin)
        target_y = random.randint(margin, screen_h - margin)
        move_mouse_to(target_x, target_y, steps=40, total_delay=segment_duration)


def main():
    parser = argparse.ArgumentParser(description="鼠标防休眠 - 保持 GPU 活跃")
    parser.add_argument(
        "--interval", type=int, default=10,
        help="触发间隔（秒），默认 10"
    )
    parser.add_argument(
        "--move-duration", type=float, default=5.0,
        help="每次移动持续时间（秒），默认 5"
    )
    args = parser.parse_args()

    interval = args.interval
    move_duration = args.move_duration

    print("鼠标防休眠已启动（SendInput 模式）")
    print(f"  触发间隔: {interval} 秒")
    print(f"  移动持续: {move_duration} 秒")
    print("按 Ctrl+C 停止")
    print("-" * 40)

    count = 0
    try:
        while True:
            count += 1
            print(f"第 {count} 次移动中...", flush=True)
            move_mouse_randomly(duration=move_duration)
            print(f"第 {count} 次完成，等待 {interval} 秒...", flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\n已停止，共执行 {count} 次")


if __name__ == "__main__":
    main()
