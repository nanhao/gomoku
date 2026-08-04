#pragma once

#include <vector>

/** 棋盘状态模块
 *
 * 提供 15×15 五子棋盘的胜负判定与状态查询。
 * 棋盘编码：0=空, 1=黑, 2=白，225 维平坦向量（按行展开）。
 *
 * 不可变设计：apply_action 返回新实例，不修改自身。
 */
class BoardState {
public:
    /** 从 225 维棋盘向量和当前玩家构造
     *
     * @param board 225 维 int 向量，0=空, 1=黑, 2=白
     * @param current_player 当前落子方，1=黑, 2=白
     */
    BoardState(std::vector<int> board, int current_player);

    /** 检查棋盘上是否存在五连 */
    bool has_winner() const;

    /** 返回胜者，1=黑胜, 2=白胜，无胜者返回 0 */
    int get_winner() const;

    /** 检查棋盘是否已满（平局条件） */
    bool is_full() const;

    /** 是否终局：有人胜或满棋盘 */
    bool is_terminal() const;

    /** 终局价值（从当前玩家视角）。
     *
     * 胜=1.0, 负=-1.0, 平=0.0。
     * 调用方应确保仅在 is_terminal() 为 true 时调用此方法。
     */
    float terminal_value() const;

    /** 获取棋盘向量常量引用（225 维） */
    const std::vector<int>& get_board() const;

    /** 获取所有空位索引列表 */
    std::vector<int> get_legal_actions() const;

    /** 在指定位置落子，返回新 BoardState。
     *
     * 不修改当前实例。新状态的 current_player 自动翻转。
     *
     * @param action 落子索引 0-224
     * @return 落子后的新棋盘状态
     */
    BoardState apply_action(int action) const;

    /** 输出相对视角编码张量（与 Python board_vector_to_cnn_tensor_relative 一致）。
     *
     * 返回 3×15×15 = 675 个 float，排列方式：[channel][row][col]：
     *   - 通道 0：我方棋子位置
     *   - 通道 1：对方棋子位置
     *   - 通道 2：我方颜色指示器
     *
     * 张量按 channel_major 排列：先遍历通道，再行，再列。
     */
    std::vector<float> encode_for_network() const;

private:
    std::vector<int> board_;
    int current_player_;
    mutable int cached_winner_ = -1;  // -1=未计算, 0=无胜者, 1/2=胜者

    /** 检查从指定位置出发是否存在五连 */
    static int check_winner_at(const std::vector<int>& board, int pos);

    /** 四方向扫描：水平、垂直、主对角线、副对角线 */
    static const int DIRECTION_OFFSETS[4][2];
};
