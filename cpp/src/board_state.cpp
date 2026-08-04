#include "board_state.h"

#include <algorithm>
#include <cstddef>

// 四方向扫描偏移量：{正向步进, 反向步进}，均为 flat 索引偏移
// 水平方向: 列偏移 ±1
// 垂直方向: 行偏移 ±15
// 主对角线: ±16
// 副对角线: ±14
const int BoardState::DIRECTION_OFFSETS[4][2] = {
    {1, -1},        // 水平: (0,+1), (0,-1) -> flat: +1, -1
    {15, -15},      // 垂直: (+1,0), (-1,0) -> flat: +15, -15
    {16, -16},      // 主对角线: (+1,+1), (-1,-1) -> flat: +16, -16
    {14, -14},      // 副对角线: (+1,-1), (-1,+1) -> flat: +14, -14
};

// 用于 validate_move 的辅助：检查 flat 索引在同一方向上的步进是否越行/越列
namespace {

/** 判断从 pos 沿 offset 移动一步是否仍在棋盘内且不跨行/列边界。
 *
 * 使用 (row, col) 坐标确保对角线方向不会从行尾跳到下一行。
 */
bool is_valid_step(int pos, int offset) {
    int new_pos = pos + offset;
    if (new_pos < 0 || new_pos >= 225) return false;

    int row = pos / 15;
    int col = pos % 15;
    int new_row = new_pos / 15;
    int new_col = new_pos % 15;

    // 行/列跨度不能超过 1（防止对角线穿越边界）
    int dr = new_row - row;
    int dc = new_col - col;
    return (dr >= -1 && dr <= 1) && (dc >= -1 && dc <= 1);
}

}  // namespace

BoardState::BoardState(std::vector<int> board, int current_player)
    : board_(std::move(board)), current_player_(current_player) {}

bool BoardState::has_winner() const {
    return get_winner() != 0;
}

int BoardState::get_winner() const {
    if (cached_winner_ >= 0) return cached_winner_;

    for (int pos = 0; pos < 225; ++pos) {
        if (board_[pos] != 0) {
            int w = check_winner_at(board_, pos);
            if (w != 0) {
                cached_winner_ = w;
                return w;
            }
        }
    }
    cached_winner_ = 0;
    return 0;
}

bool BoardState::is_full() const {
    return std::all_of(board_.begin(), board_.end(),
                       [](int cell) { return cell != 0; });
}

bool BoardState::is_terminal() const {
    return has_winner() || is_full();
}

float BoardState::terminal_value() const {
    int winner = get_winner();
    if (winner != 0) {
        // 从当前玩家视角：胜=1.0, 负=-1.0
        return (winner == current_player_) ? 1.0f : -1.0f;
    }
    // 平局
    return 0.0f;
}

const std::vector<int>& BoardState::get_board() const {
    return board_;
}

std::vector<int> BoardState::get_legal_actions() const {
    std::vector<int> actions;
    actions.reserve(225);
    for (int i = 0; i < 225; ++i) {
        if (board_[i] == 0) {
            actions.push_back(i);
        }
    }
    return actions;
}

BoardState BoardState::apply_action(int action) const {
    std::vector<int> new_board = board_;
    new_board[action] = current_player_;
    int next_player = (current_player_ == 1) ? 2 : 1;
    return BoardState(std::move(new_board), next_player);
}

int BoardState::check_winner_at(const std::vector<int>& board, int pos) {
    int player = board[pos];
    if (player == 0) return 0;

    int row = pos / 15;
    int col = pos % 15;

    // 四个方向扫描
    for (int d = 0; d < 4; ++d) {
        int count = 1;  // 包含当前位置

        int offset_fwd = DIRECTION_OFFSETS[d][0];
        int offset_bwd = DIRECTION_OFFSETS[d][1];

        // 正向扫描
        int cur = pos;
        while (is_valid_step(cur, offset_fwd)) {
            cur += offset_fwd;
            if (board[cur] == player) {
                ++count;
            } else {
                break;
            }
        }

        // 反向扫描
        cur = pos;
        while (is_valid_step(cur, offset_bwd)) {
            cur += offset_bwd;
            if (board[cur] == player) {
                ++count;
            } else {
                break;
            }
        }

        if (count >= 5) return player;
    }

    return 0;
}

std::vector<float> BoardState::encode_for_network() const {
    // 3 通道 × 15 行 × 15 列 = 675，channel-major 排列
    std::vector<float> tensor(675, 0.0f);

    int opponent = (current_player_ == 1) ? 2 : 1;
    float color_indicator = (current_player_ == 1) ? 1.0f : 0.0f;

    for (int i = 0; i < 225; ++i) {
        int row = i / 15;
        int col = i % 15;

        // 通道 0：我方棋子（offset = 0 * 225 + row * 15 + col）
        if (board_[i] == current_player_) {
            tensor[0 * 225 + row * 15 + col] = 1.0f;
        }

        // 通道 1：对方棋子（offset = 1 * 225 + row * 15 + col）
        if (board_[i] == opponent) {
            tensor[1 * 225 + row * 15 + col] = 1.0f;
        }

        // 通道 2：我方颜色指示器（offset = 2 * 225 + row * 15 + col）
        tensor[2 * 225 + row * 15 + col] = color_indicator;
    }

    return tensor;
}
