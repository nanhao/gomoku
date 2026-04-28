"""冲四棋型训练样本生成器 v4

生成 4096 个包含冲四棋型的训练样本。

核心思路：
- 棋盘上已有4颗同色棋子连珠（冲四已形成）
- 一端被阻挡（对手棋子或棋盘边界）
- 另一端开放（空位）→ 落子即形成连五 → reward = MAX
- 使用 RewardCalculator 和 masked_softmax 计算正确的标签
"""

import sys
import os
import json
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai.reward_calculator import RewardCalculator, RewardConfig
from ai.shape_detector import detect_shapes
from ai.probability_utils import masked_softmax

NUM_CELLS = 225
NUM_SAMPLES = 40960

DIRECTIONS = [
    (0, 1),   # 横向
    (1, 0),   # 纵向
    (1, 1),   # 主对角线
    (1, -1),  # 副对角线
]


def make_empty_board():
    return [0] * NUM_CELLS


def set_piece(board, x, y, player):
    board[x * 15 + y] = player


def is_valid_pos(x, y):
    return 0 <= x < 15 and 0 <= y < 15


def is_empty(board, x, y):
    return board[x * 15 + y] == 0


def generate_chong4_board(random_gen):
    """生成一个棋盘，上面已存在冲四棋型（4子连珠一端被挡）

    返回 (board, open_end_pos, current_player) 或 None
    - board: 棋盘状态
    - open_end_pos: 冲四开放端坐标 (x, y)，该位置落子可形成连五
    - current_player: 当前落子方（即拥有冲四的一方）
    """
    for _ in range(300):
        player = random_gen.choice([1, 2])
        opponent = 3 - player
        dx, dy = random_gen.choice(DIRECTIONS)

        # 选择冲四线的起始位置（4颗棋子中偏移最小的那颗）
        # 然后在两端分别检查阻挡和开放

        # 随机选择4颗连珠棋子的起始位置
        start_x = random_gen.randint(0, 14)
        start_y = random_gen.randint(0, 14)

        # 检查4颗棋子位置是否都在棋盘内
        stone_positions = []
        for i in range(4):
            sx = start_x + i * dx
            sy = start_y + i * dy
            if not is_valid_pos(sx, sy):
                break
            stone_positions.append((sx, sy))

        if len(stone_positions) != 4:
            continue

        # 4颗棋子的两端
        # 负方向端（起始位置之前）
        neg_end_x = start_x - dx
        neg_end_y = start_y - dy
        neg_end_valid = is_valid_pos(neg_end_x, neg_end_y)

        # 正方向端（第4颗棋子之后）
        last_x, last_y = stone_positions[3]
        pos_end_x = last_x + dx
        pos_end_y = last_y + dy
        pos_end_valid = is_valid_pos(pos_end_x, pos_end_y)

        # 需要恰好一端被阻挡，另一端开放（空位）
        # 开放端就是落子可形成连五的位置
        blocked_end = None
        open_end = None

        if not neg_end_valid and pos_end_valid:
            # 负方向被边界阻挡，正方向开放
            blocked_end = 'neg_edge'
            open_end = (pos_end_x, pos_end_y)
        elif not pos_end_valid and neg_end_valid:
            # 正方向被边界阻挡，负方向开放
            blocked_end = 'pos_edge'
            open_end = (neg_end_x, neg_end_y)
        elif neg_end_valid and pos_end_valid:
            # 两端都在棋盘内，用对手棋子阻挡一端
            block_choice = random_gen.choice(['neg', 'pos'])
            if block_choice == 'neg':
                blocked_end = 'neg_opponent'
                open_end = (pos_end_x, pos_end_y)
            else:
                blocked_end = 'pos_opponent'
                open_end = (neg_end_x, neg_end_y)
        else:
            # 两端都是边界 → 活四（两端都开放）或无意义
            continue

        # 构建棋盘
        board = make_empty_board()

        # 放置4颗同色棋子
        for sx, sy in stone_positions:
            set_piece(board, sx, sy, player)

        # 放置阻挡棋子
        if blocked_end == 'neg_opponent':
            set_piece(board, neg_end_x, neg_end_y, opponent)
        elif blocked_end == 'pos_opponent':
            set_piece(board, pos_end_x, pos_end_y, opponent)

        # 验证开放端为空
        ox, oy = open_end
        if board[ox * 15 + oy] != 0:
            continue

        # 收集需要保护的位置
        protected = set()
        protected.add((ox, oy))  # 开放端必须为空
        for sx, sy in stone_positions:
            protected.add((sx, sy))
        # 阻挡位置也保护
        if blocked_end == 'neg_opponent':
            protected.add((neg_end_x, neg_end_y))
        elif blocked_end == 'pos_opponent':
            protected.add((pos_end_x, pos_end_y))

        # 添加少量干扰棋子 (0-4颗)
        num_extra = random_gen.randint(0, 4)
        for _ in range(num_extra):
            for _ in range(50):
                rx, ry = random_gen.randint(0, 14), random_gen.randint(0, 14)
                if is_empty(board, rx, ry) and (rx, ry) not in protected:
                    extra_player = random_gen.choice([player, opponent])
                    set_piece(board, rx, ry, extra_player)
                    break

        # 再次验证开放端为空
        if board[ox * 15 + oy] != 0:
            continue

        # 验证开放端落子确实形成连五
        board_copy = list(board)
        board_copy[ox * 15 + oy] = player
        shapes = detect_shapes(board_copy, (ox, oy), player)
        shape_names = [s["shape"] for s in shapes]

        if "连五" in shape_names:
            return board, open_end, player

    return None


def create_sample(board, current_player, reward_calculator):
    """根据棋盘状态创建训练样本"""
    input_vector = [float(x) for x in board] + [float(current_player)]

    mask = [0] * NUM_CELLS
    raw_rewards = [0.0] * NUM_CELLS

    for i in range(NUM_CELLS):
        if board[i] != 0:
            mask[i] = 0
            raw_rewards[i] = 0.0
        else:
            mask[i] = 1
            x, y = i // 15, i % 15
            raw_rewards[i] = float(reward_calculator.compute_position_reward(
                board, x, y, current_player))

    output = masked_softmax(raw_rewards, mask)

    return {
        "input": input_vector,
        "output": output,
        "mask": mask
    }


def main():
    random_gen = random.Random(42)
    reward_calculator = RewardCalculator(RewardConfig())

    samples = []

    print(f"开始生成 {NUM_SAMPLES} 个冲四棋型样本（开放端=连五位）...")

    while len(samples) < NUM_SAMPLES:
        result = generate_chong4_board(random_gen)
        if result is None:
            continue

        board, open_end, current_player = result

        sample = create_sample(board, current_player, reward_calculator)
        samples.append(sample)

        if len(samples) % 500 == 0:
            print(f"  已生成 {len(samples)}/{NUM_SAMPLES}")

    print(f"\n生成完成!")

    # 验证：开放端是否为最高概率位置
    print("\n验证开放端（连五位）概率排名...")
    top1 = top3 = top5 = 0
    open_end_rewards = []

    for sample in samples:
        board = [int(x) for x in sample["input"][:225]]
        current_player = int(sample["input"][225])
        output = sample["output"]

        # 找到连五位置（开放端）
        lian5_positions = []
        for i in range(NUM_CELLS):
            if board[i] == 0:
                x, y = i // 15, i % 15
                board_copy = list(board)
                board_copy[i] = current_player
                shapes = detect_shapes(board_copy, (x, y), current_player)
                for s in shapes:
                    if s["shape"] == "连五":
                        lian5_positions.append(i)
                        break

        if lian5_positions:
            sorted_indices = sorted(range(NUM_CELLS),
                                     key=lambda idx: output[idx], reverse=True)
            best_rank = min(sorted_indices.index(p) for p in lian5_positions)
            if best_rank == 0:
                top1 += 1
            if best_rank < 3:
                top3 += 1
            if best_rank < 5:
                top5 += 1

            # 记录开放端的 reward
            for p in lian5_positions:
                x, y = p // 15, p % 15
                r = reward_calculator.compute_position_reward(board, x, y, current_player)
                open_end_rewards.append(r)

    total = len(samples)
    print(f"  连五位 Top1: {top1}/{total} ({100*top1/total:.1f}%)")
    print(f"  连五位 Top3: {top3}/{total} ({100*top3/total:.1f}%)")
    print(f"  连五位 Top5: {top5}/{total} ({100*top5/total:.1f}%)")
    if open_end_rewards:
        print(f"  开放端 reward: min={min(open_end_rewards)}, max={max(open_end_rewards)}, "
              f"avg={sum(open_end_rewards)/len(open_end_rewards):.0f}")

    # 保存
    output_path = os.path.join(
        os.path.dirname(__file__), '..', 'ai', 'merge_samples', 'chong4.json')
    output_path = os.path.normpath(output_path)
    print(f"\n保存到 {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False)

    file_size = os.path.getsize(output_path)
    print(f"保存完成! 文件大小: {file_size / 1024 / 1024:.1f} MB")
    print(f"样本数量: {len(samples)}")


if __name__ == '__main__':
    main()
