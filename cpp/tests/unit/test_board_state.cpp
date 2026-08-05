/** BoardState 单元测试
 *
 * 覆盖 design/46/main_docs/test_cases.md 中 TC-BS-01 至 TC-BS-11 的所有用例。
 *
 * 运行方式：编译后直接执行，exit code 0 = 全部通过。
 */
#include "../src/board_state.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name)                                         \
    do {                                                   \
        std::printf("  TEST %s ... ", name);               \
    } while (0)

#define PASS()                                             \
    do {                                                   \
        std::printf("PASS\n");                             \
        ++tests_passed;                                    \
    } while (0)

#define FAIL(msg)                                          \
    do {                                                   \
        std::printf("FAIL: %s\n", msg);                    \
        ++tests_failed;                                    \
    } while (0)

#define ASSERT_TRUE(cond)                                  \
    do {                                                   \
        if (!(cond)) { FAIL(#cond); return; }              \
    } while (0)

#define ASSERT_FALSE(cond)                                 \
    do {                                                   \
        if (cond) { FAIL(#cond " (expected false)"); return; } \
    } while (0)

#define ASSERT_EQ(a, b)                                    \
    do {                                                   \
        if ((a) != (b)) {                                  \
            std::printf("FAIL: %s == %s, expected %d got %d\n", \
                        #a, #b, (int)(b), (int)(a));       \
            ++tests_failed;                                \
            return;                                        \
        }                                                  \
    } while (0)

#define ASSERT_FLOAT_EQ(a, b, eps)                         \
    do {                                                   \
        if (std::fabs((a) - (b)) > (eps)) {                \
            std::printf("FAIL: %s ~= %s, expected %.6f got %.6f\n", \
                        #a, #b, (float)(b), (float)(a));   \
            ++tests_failed;                                \
            return;                                        \
        }                                                  \
    } while (0)


// --- TC-BS-01: 空棋盘检测 ---
static void test_empty_board() {
    TEST("TC-BS-01 empty board");
    std::vector<int> board(225, 0);
    BoardState bs(board, 1);

    ASSERT_FALSE(bs.has_winner());
    ASSERT_FALSE(bs.is_full());
    ASSERT_FALSE(bs.is_terminal());
    ASSERT_EQ((int)bs.get_legal_actions().size(), 225);
    PASS();
}

// --- TC-BS-02: 满棋盘平局 ---
static void test_full_board_draw() {
    TEST("TC-BS-02 full board draw");
    // 构造一个满棋盘但无五连的局面
    // 使用 modulo 6 模式 (阈值 3): 0-2→1(黑), 3-5→2(白)
    // 已验证四个方向 (水平步进1, 垂直步进15, 主对角线步进16, 反对角线步进14)
    // 5步序列均包含两种颜色，保证无五连
    std::vector<int> board(225, 0);
    for (int i = 0; i < 225; ++i) {
        board[i] = (i % 6 < 3) ? 1 : 2;
    }
    BoardState bs(board, 1);

    ASSERT_FALSE(bs.has_winner());
    ASSERT_TRUE(bs.is_full());
    ASSERT_TRUE(bs.is_terminal());
    ASSERT_FLOAT_EQ(bs.terminal_value(), 0.0f, 1e-6);
    PASS();
}

// --- TC-BS-03: 水平五连（黑胜）---
static void test_horizontal_win_black() {
    TEST("TC-BS-03 horizontal five black");
    std::vector<int> board(225, 0);
    // 在第 0 行放置黑子五连 [0,1,2,3,4]
    board[0] = 1;
    board[1] = 1;
    board[2] = 1;
    board[3] = 1;
    board[4] = 1;
    BoardState bs(board, 1);

    ASSERT_TRUE(bs.has_winner());
    ASSERT_EQ(bs.get_winner(), 1);
    ASSERT_FLOAT_EQ(bs.terminal_value(), 1.0f, 1e-6);
    PASS();
}

// --- TC-BS-04: 垂直五连（白胜）---
static void test_vertical_win_white() {
    TEST("TC-BS-04 vertical five white");
    std::vector<int> board(225, 0);
    // 在第 0 列放置白子五连 [0,15,30,45,60]
    board[0] = 2;
    board[15] = 2;
    board[30] = 2;
    board[45] = 2;
    board[60] = 2;
    BoardState bs(board, 2);

    ASSERT_EQ(bs.get_winner(), 2);
    PASS();
}

// --- TC-BS-05: 主对角线五连 ---
static void test_main_diag_win() {
    TEST("TC-BS-05 main diagonal five");
    std::vector<int> board(225, 0);
    // 位置 [0,16,32,48,64] 均为黑子
    board[0] = 1;
    board[16] = 1;
    board[32] = 1;
    board[48] = 1;
    board[64] = 1;
    BoardState bs(board, 1);

    ASSERT_EQ(bs.get_winner(), 1);
    PASS();
}

// --- TC-BS-06: 副对角线五连 ---
static void test_anti_diag_win() {
    TEST("TC-BS-06 anti diagonal five");
    std::vector<int> board(225, 0);
    // 位置 [4,18,32,46,60] 均为白子
    board[4] = 2;
    board[18] = 2;
    board[32] = 2;
    board[46] = 2;
    board[60] = 2;
    BoardState bs(board, 2);

    ASSERT_EQ(bs.get_winner(), 2);
    PASS();
}

// --- TC-BS-07: 边缘对角线（最后一组）---
static void test_edge_diag_win() {
    TEST("TC-BS-07 edge diagonal five");
    std::vector<int> board(225, 0);
    // 位置 [210,196,182,168,154] 均为黑子
    board[210] = 1;
    board[196] = 1;
    board[182] = 1;
    board[168] = 1;
    board[154] = 1;
    BoardState bs(board, 1);

    ASSERT_EQ(bs.get_winner(), 1);
    PASS();
}

// --- TC-BS-08: 四连无胜者 ---
static void test_four_in_row_no_win() {
    TEST("TC-BS-08 four in a row no win");
    std::vector<int> board(225, 0);
    // 位置 [0,1,2,3] 为黑子，[4] 为空
    board[0] = 1;
    board[1] = 1;
    board[2] = 1;
    board[3] = 1;
    board[4] = 0;
    BoardState bs(board, 1);

    ASSERT_FALSE(bs.has_winner());
    PASS();
}

// --- TC-BS-09: apply_action 创建新状态 ---
static void test_apply_action() {
    TEST("TC-BS-09 apply_action immutable");
    std::vector<int> board(225, 0);
    BoardState original(board, 1);

    BoardState next = original.apply_action(0);

    // 新状态 board[0]=1, player=2
    ASSERT_EQ(next.get_board()[0], 1);
    ASSERT_EQ((int)next.get_board().size(), 225);
    // 原状态不变
    ASSERT_EQ(original.get_board()[0], 0);

    // 验证玩家翻转：apply_action 后的新实例已经翻转到对方
    // 通过再落一颗对方棋子来验证
    BoardState after_white = next.apply_action(1);
    ASSERT_EQ(after_white.get_board()[1], 2);
    PASS();
}

// --- TC-BS-10: 终局状态 terminal_value（对方视角）---
static void test_terminal_value_opponent_view() {
    TEST("TC-BS-10 terminal_value opponent view");
    std::vector<int> board(225, 0);
    // 黑子五连
    board[0] = 1;
    board[1] = 1;
    board[2] = 1;
    board[3] = 1;
    board[4] = 1;
    // 当前玩家=2（白方视角）
    BoardState bs(board, 2);

    ASSERT_TRUE(bs.is_terminal());
    // 白方视角下黑胜=白负 → -1.0
    ASSERT_FLOAT_EQ(bs.terminal_value(), -1.0f, 1e-6);
    PASS();
}

// --- TC-BS-11: 编码一致性 ---
static void test_encode_for_network() {
    TEST("TC-BS-11 encode_for_network");
    std::vector<int> board(225, 0);
    board[0] = 1;  // 黑子
    board[1] = 2;  // 白子
    BoardState bs(board, 1);  // current_player=1(黑)

    auto tensor = bs.encode_for_network();
    ASSERT_EQ((int)tensor.size(), 675);

    // 通道0 (我方=黑): 位置 0=1.0
    ASSERT_FLOAT_EQ(tensor[0], 1.0f, 1e-6);
    // 通道0 位置 1: 对方棋子 → 0.0
    ASSERT_FLOAT_EQ(tensor[1], 0.0f, 1e-6);

    // 通道1 (对方=白): 位置 1=1.0
    ASSERT_FLOAT_EQ(tensor[1 * 225 + 1], 1.0f, 1e-6);
    // 通道1 位置 0: 我方棋子 → 0.0
    ASSERT_FLOAT_EQ(tensor[1 * 225 + 0], 0.0f, 1e-6);

    // 通道2 (我方颜色=黑): 全 1.0
    ASSERT_FLOAT_EQ(tensor[2 * 225 + 0], 1.0f, 1e-6);
    ASSERT_FLOAT_EQ(tensor[2 * 225 + 100], 1.0f, 1e-6);
    ASSERT_FLOAT_EQ(tensor[2 * 225 + 224], 1.0f, 1e-6);

    // 验证 current_player=2 时通道2 全 0.0
    BoardState bs2(board, 2);
    auto tensor2 = bs2.encode_for_network();
    ASSERT_FLOAT_EQ(tensor2[2 * 225 + 0], 0.0f, 1e-6);
    ASSERT_FLOAT_EQ(tensor2[2 * 225 + 100], 0.0f, 1e-6);

    PASS();
}


int main() {
    std::printf("=== BoardState Unit Tests ===\n\n");

    test_empty_board();
    test_full_board_draw();
    test_horizontal_win_black();
    test_vertical_win_white();
    test_main_diag_win();
    test_anti_diag_win();
    test_edge_diag_win();
    test_four_in_row_no_win();
    test_apply_action();
    test_terminal_value_opponent_view();
    test_encode_for_network();

    std::printf("\n=== Results: %d passed, %d failed ===\n",
                tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
