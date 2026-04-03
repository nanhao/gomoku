# 热力图鼠标悬停 tooltip 不显示修复

## 1. 需求内容

用户反馈：回放界面热力图鼠标悬停时没有显示 reward 值。

## 2. 解决思路

**根本原因**：`HeatmapCanvas._show_tooltip` 方法只调用了外部回调（`callback(reward, row, col, x_root, y_root)`），但从未调用 `create_tooltip` 创建 tooltip 窗口；而 `replay_viewer._on_cell_hover` 回调本身只有 `pass`，因此鼠标悬停完全没有任何视觉反馈。

**修复方案**：在 `_show_tooltip` 内直接调用 `self.create_tooltip`，让 tooltip 由热力图自身管理，外部回调保留供扩展。

## 3. 执行过程

1. 阅读 `heatmap.py` 定位问题：`_show_tooltip` 只调用回调，`create_tooltip` 存在但从未被调用
2. 阅读 `replay_viewer._on_cell_hover`：回调为空实现（`pass`）
3. 分析两种方案：
   - 方案A：在 `replay_viewer._on_cell_hover` 里调用对应热力图的 `create_tooltip`（回调里无法区分 AI/Sample 热力图，架构不清晰）
   - 方案B：在 `_show_tooltip` 内直接调用 `self.create_tooltip`（tooltip 归属热力图自身，更内聚）
4. 选择方案B，修改 `_show_tooltip`：先调用 `create_tooltip` 显示 `(row, col) reward: value`，再调用外部回调
5. 添加 4 个边缘测试用例，含回归用例 `test_show_tooltip_creates_window`
6. 运行测试：19 passed，2 errors（均为 Tkinter 环境偶发故障，与代码无关）

## 4. 执行结果

- 鼠标悬停热力图格子时，弹出 tooltip 显示 `(row, col)  reward: value`
- 新增 4 个测试用例全部通过

## 5. 涉及文件

| 文件 | 操作 |
|------|------|
| `gui/components/heatmap.py` | 修复 `_show_tooltip`，在内部调用 `create_tooltip` |
| `tests/gui/test_heatmap.py` | 添加 tooltip 相关边缘测试用例 |
