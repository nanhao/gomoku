/** 训练样本与 D4 数据增强模块
 *
 * 定义 AGZ 格式训练样本 GameSample 和 D4 群 8 种对称变换工具类 SampleAugmentor。
 *
 * GameSample 格式：{state[226], pi[225], z}
 * - state: cnn_v11 相对视角编码（225 棋盘 + 1 玩家标识）
 * - pi: MCTS visit_counts 归一化概率分布
 * - z: 终局标签（+1=当前玩家胜, -1=当前玩家负, 0=平局）
 *
 * SampleAugmentor 提供静态方法对单条样本执行 8 种棋盘对称变换，
 * 返回 8 条增强变体。变换仅影响 state[0:225] 和 pi[0:225]，
 * state[225]（玩家标识）和 z 值保持不变。
 */
#pragma once

#include <cstdint>
#include <vector>


// =====================================================================
// GameSample — 训练样本结构体
// =====================================================================

struct GameSample {
    std::vector<float> state;   // 226 维：[225 棋盘编码 + 1 当前玩家标识]
    std::vector<float> pi;      // 225 维：MCTS visit counts 归一化后的概率分布
    float z;                    // 终局标签：+1（当前玩家胜）、-1（当前玩家负）、0（平局）
};


// =====================================================================
// SampleAugmentor — D4 数据增强
// =====================================================================

class SampleAugmentor {
public:
    /** 对单条样本执行 D4 增强，返回 8 条变体样本
     *
     * 返回样本中 state[225] 和 z 与原始一致；state[0:225] 和 pi 按变换映射。
     *
     * @param sample 原始样本
     * @return 8 条增强变体（包含恒等变换作为第一条）
     */
    static std::vector<GameSample> augment(const GameSample& sample);

    /** 对 225 维向量执行指定的棋盘对称变换
     *
     * 变换仅重新排列向量元素，不修改任何元素值。
     *
     * @param vec 225 维向量（棋盘编码或策略分布）
     * @param tid 变换 ID（0~7）
     * @return 变换后的 225 维向量
     */
    static std::vector<float> transform_225(const std::vector<float>& vec, int tid);

    /** 返回变换 tid 将原始棋盘索引映射到的新索引
     *
     * 将 15×15 棋盘上原始坐标 (r, c) 经 D4 变换后对应的位置索引。
     *
     * 变换定义：
     *   tid 0 — 恒等：(r, c)
     *   tid 1 — 顺时针 90°：(c, 14 - r)
     *   tid 2 — 180°：(14 - r, 14 - c)
     *   tid 3 — 顺时针 270°：(14 - c, r)
     *   tid 4 — 左右镜像：(r, 14 - c)
     *   tid 5 — 上下镜像：(14 - r, c)
     *   tid 6 — 主对角线翻转：(c, r)
     *   tid 7 — 反对角线翻转：(14 - c, 14 - r)
     *
     * @param idx 原始棋盘索引 0~224（idx = r * 15 + c）
     * @param tid 变换 ID（0~7）
     * @return 变换后的棋盘索引 0~224
     */
    static int transform_index(int idx, int tid);

private:
    /** 将棋盘索引 (idx) 转换为行列坐标 (r, c) */
    static void _idx_to_rc(int idx, int& r, int& c);

    /** 将行列坐标 (r, c) 转换为棋盘索引 */
    static int _rc_to_idx(int r, int c);
};


// =====================================================================
// 内联实现（工具方法，足够简单可内联）
// =====================================================================

inline void SampleAugmentor::_idx_to_rc(int idx, int& r, int& c) {
    r = idx / 15;
    c = idx % 15;
}

inline int SampleAugmentor::_rc_to_idx(int r, int c) {
    return r * 15 + c;
}

inline int SampleAugmentor::transform_index(int idx, int tid) {
    int r, c;
    _idx_to_rc(idx, r, c);

    int nr, nc;
    switch (tid) {
        case 0:  // 恒等
            nr = r; nc = c; break;
        case 1:  // 顺时针 90°
            nr = c; nc = 14 - r; break;
        case 2:  // 180°
            nr = 14 - r; nc = 14 - c; break;
        case 3:  // 顺时针 270°
            nr = 14 - c; nc = r; break;
        case 4:  // 左右镜像
            nr = r; nc = 14 - c; break;
        case 5:  // 上下镜像
            nr = 14 - r; nc = c; break;
        case 6:  // 主对角线翻转
            nr = c; nc = r; break;
        case 7:  // 反对角线翻转
            nr = 14 - c; nc = 14 - r; break;
        default:
            nr = r; nc = c; break;  // 退化：无效 tid 视为恒等
    }
    return _rc_to_idx(nr, nc);
}

inline std::vector<float> SampleAugmentor::transform_225(
        const std::vector<float>& vec, int tid) {
    std::vector<float> result(225);
    for (int i = 0; i < 225; ++i) {
        result[transform_index(i, tid)] = vec[i];
    }
    return result;
}

inline std::vector<GameSample> SampleAugmentor::augment(const GameSample& sample) {
    std::vector<GameSample> result;
    result.reserve(8);

    for (int tid = 0; tid < 8; ++tid) {
        GameSample aug;
        // state[0:225] 和 pi[0:225] 按变换映射
        std::vector<float> state225(sample.state.begin(), sample.state.begin() + 225);
        aug.state = transform_225(state225, tid);
        aug.state.push_back(sample.state[225]);  // 玩家标识不变

        aug.pi = transform_225(sample.pi, tid);
        aug.z = sample.z;  // z 值不变

        result.push_back(std::move(aug));
    }
    return result;
}
