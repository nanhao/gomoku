"""交互式命令行主程序"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from game import GomokuController


# 棋子显示符号
PIECE_EMPTY = "."
PIECE_BLACK = "●"
PIECE_WHITE = "○"


def main():
    """程序入口，启动 REPL 循环"""
    controller = GomokuController()
    
    print("五子棋游戏 v1.0")
    print("输入 'help' 查看命令列表")
    print()
    
    while True:
        try:
            user_input = input("> ").strip()
            if not user_input:
                continue
            
            result = parse_command(controller, user_input)
            if result == "quit":
                print("再见！")
                break
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except EOFError:
            print("\n再见！")
            break


def parse_command(controller, input_str):
    """解析用户输入的命令"""
    parts = input_str.split()
    if not parts:
        return None
    
    cmd = parts[0].lower()
    args = parts[1:]
    
    if cmd in ("quit", "exit"):
        return "quit"
    elif cmd == "new":
        handle_new(controller, args)
    elif cmd == "put":
        handle_put(controller, args)
    elif cmd == "list":
        handle_list(controller, args)
    elif cmd == "print":
        handle_print(controller, args)
    elif cmd == "show":
        handle_show(controller, args)
    elif cmd == "short":
        handle_short(controller, args)
    elif cmd == "help":
        handle_help(controller, args)
    else:
        print(f"未知命令: {cmd}，输入 'help' 查看帮助")
    
    return None


def handle_new(controller, args):
    """处理 new 命令"""
    save_name = args[0] if args else None
    
    result = controller.new_game(save_name)
    
    if "error" in result:
        print(f"错误: {result['error']}")
        return
    
    state = result["state"]
    print(f"游戏已创建，黑方先手")
    print(f"当前存档: {state['name']}")
    print()
    print_board(state["board"], state["current_player"], None)


def handle_put(controller, args):
    """处理 put 命令"""
    if len(args) != 2:
        print("错误: 用法: put <x> <y>")
        return
    
    try:
        x = int(args[0])
        y = int(args[1])
    except ValueError:
        print("错误: 坐标必须是整数")
        return
    
    result = controller.make_move(x, y)
    
    if "error" in result:
        print(f"错误: {result['error']}")
        return
    
    # 显示落子信息
    player_name = "黑方" if result["current_player"] == 2 else "白方"  # 已经切换了
    last_player = 2 if result["current_player"] == 1 else 1
    last_player_name = "黑方" if last_player == 1 else "白方"
    print(f"{last_player_name}在 ({x}, {y}) 落子")
    
    # 检查游戏是否结束
    if result.get("game_over"):
        print(f"\n游戏结束！{result['result']}！")
        print_board(result["board"], result["current_player"], (x, y))
    else:
        next_player = "黑方" if result["current_player"] == 1 else "白方"
        print(f"{next_player}回合")
        print()
        print_board(result["board"], result["current_player"], (x, y))


def handle_list(controller, args):
    """处理 list 命令"""
    saves = controller.list_saves()
    
    if not saves:
        print("暂无存档")
        return
    
    print("存档列表:")
    for save_name in saves:
        print(f"  {save_name}")


def handle_print(controller, args):
    """处理 print 命令"""
    if controller.game is None:
        print("错误: 请先创建或加载游戏")
        return
    
    save_name = controller.current_save_name or "未知"
    print(f"当前存档: {save_name}")
    print()
    
    state = controller.get_state()
    last_move = None
    if state.get("history"):
        last = state["history"][-1]
        last_move = (last["x"], last["y"])
    
    print_board(state["board"], state["current_player"], last_move)
    
    # 显示游戏状态
    if state["winner"] == 0:
        player_name = "黑方" if state["current_player"] == 1 else "白方"
        piece = PIECE_BLACK if state["current_player"] == 1 else PIECE_WHITE
        print(f"\n当前玩家: {player_name} ({piece})")
        if last_move:
            print(f"上次落子: {last_move}")
        print("游戏状态: 进行中")
    elif state["winner"] == 3:
        print("\n游戏状态: 平局")
    else:
        winner_name = "黑方" if state["winner"] == 1 else "白方"
        print(f"\n游戏状态: {winner_name}获胜")


def handle_show(controller, args):
    """处理 show 命令"""
    if len(args) != 1:
        print("错误: 用法: show <存档名称>")
        return
    
    save_name = args[0]
    json_str = controller.get_state_json(short=False)
    
    # 如果当前没有游戏，尝试加载指定存档
    if controller.game is None or controller.current_save_name != save_name:
        result = controller.load_game(save_name)
        if "error" in result:
            print(f"错误: {result['error']}")
            return
        json_str = controller.get_state_json(short=False)
    
    print(json_str)


def handle_short(controller, args):
    """处理 short 命令"""
    if len(args) != 1:
        print("错误: 用法: short <存档名称>")
        return
    
    save_name = args[0]
    
    # 如果当前没有游戏，尝试加载指定存档
    if controller.game is None or controller.current_save_name != save_name:
        result = controller.load_game(save_name)
        if "error" in result:
            print(f"错误: {result['error']}")
            return
    
    json_str = controller.get_state_json(short=True)
    print(json_str)


def handle_help(controller, args):
    """处理 help 命令"""
    print("五子棋游戏 v1.0 命令列表:")
    print()
    print("new [名称]      - 新建游戏")
    print("put <x> <y>     - 在当前存档落子")
    print("list            - 列出所有存档")
    print("print           - 显示当前棋盘")
    print("show <名称>     - 显示完整存档信息")
    print("short <名称>    - 显示简要存档信息")
    print("help            - 显示本帮助信息")
    print("quit / exit     - 退出程序")
    print()
    print("坐标范围: 0-14，例如: put 7 7 表示在中心落子")


def print_board(board, current_player, last_move):
    """打印棋盘
    
    Args:
        board: 15x15 二维数组
        current_player: 当前玩家 (1=黑, 2=白)
        last_move: 上次落子坐标 (x, y) 或 None
    """
    # 打印列号
    print("    ", end="")
    for y in range(15):
        print(f"{y:2}", end=" ")
    print()
    
    # 打印每一行
    for x in range(15):
        print(f"{x:2}  ", end="")
        for y in range(15):
            cell = board[x][y]
            if cell == 0:
                symbol = PIECE_EMPTY
            elif cell == 1:
                symbol = PIECE_BLACK
            else:
                symbol = PIECE_WHITE
            print(f"{symbol:2}", end=" ")
        print()
    print()


if __name__ == "__main__":
    main()
