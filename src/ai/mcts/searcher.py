"""MCTS 搜索器核心模块

实现 AlphaGo Zero 风格的蒙特卡洛树搜索：
- PUCT 选择 + Virtual Loss
- Batch 推理（多叶节点合并评估）
- 回传取反（每层切换玩家视角）
- 狄利克雷噪声
- 子树复用（advance_root）

日志功能委托给 search_logger.MCTSSearchLogger。
"""

import math
import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np

from game.game import check_winner, is_board_full
from ai.mcts.search_logger import MCTSSearchLogger
from config.mcts import MCTS_LOG_ENABLED, MCTS_LOG_DIR

# 哨兵值：区分"未传参"和"显式传 None"
_LOG_UNSET = object()


class MCTSSearcher:
    """MCTS 搜索器

    对给定棋盘状态执行指定次数的模拟（simulation），
    每次模拟经历 selection → expansion → evaluation → backup 四阶段。

    使用 Batch 推理 + Virtual Loss 机制：
    - 选择阶段对已选边施加 Virtual Loss 惩罚
    - 收集多个叶节点后合并为一个 batch 调用模型
    - 回传前撤销 Virtual Loss
    """

    def __init__(
        self,
        model,
        num_simulations: int = 800,
        c_puct: float = 2.0,
        dirichlet_epsilon: float = 0.25,
        dirichlet_alpha: float = 0.03,
        virtual_loss: float = 3,
        mcts_batch_size: int = 16,
        log_dir: Optional[str] = _LOG_UNSET,
    ):
        """初始化 MCTS 搜索器

        Args:
            model: CNN v7 模型实例，需支持 predict(board_vector) -> (policy, value)
            num_simulations: 每次搜索的总模拟次数，默认 800
            c_puct: PUCT 探索系数，默认 2.0
            dirichlet_epsilon: 狄利克雷噪声比例，默认 0.25
            dirichlet_alpha: 狄利克雷分布参数，默认 0.03
            virtual_loss: Virtual Loss 值，默认 3
            mcts_batch_size: 单 batch 最大叶节点数，默认 64
            log_dir: 搜索日志输出目录。未传参时读取 config/mcts.py 配置；
                    显式传 None 禁用日志
        """
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_epsilon = dirichlet_epsilon
        self.dirichlet_alpha = dirichlet_alpha
        self.virtual_loss = virtual_loss
        self.mcts_batch_size = mcts_batch_size
        self.log_dir: Optional[str] = None
        if log_dir is _LOG_UNSET:
            # 未传参：读取全局配置
            self.log_dir = MCTS_LOG_DIR if MCTS_LOG_ENABLED else None
        else:
            self.log_dir = log_dir

        # 搜索树：state_hash -> Node
        self.tree: Dict[str, dict] = {}
        self._root_hash: Optional[str] = None

        # 根边 N 快照（子树复用时记录初始值，用于 get_result 中计算增量）
        self._root_edge_N_snapshot: Dict[int, int] = {}

        # 日志记录器
        self._logger = MCTSSearchLogger(self.log_dir, c_puct, virtual_loss) if self.log_dir else None

    # ---------- 公共 API ----------

    def search(
        self,
        state: List[int],
        num_simulations: Optional[int] = None,
        log_key: Optional[str] = None,
    ) -> Tuple[List[int], float, List[float]]:
        """执行 MCTS 搜索

        Args:
            state: 226 维 board_vector（前 225=棋盘按行展开，末位=当前落子方）
            num_simulations: 覆盖默认模拟次数，None 时使用构造参数
            log_key: 日志标识，非 None 且 log_dir 已配置时输出日志 JSON

        Returns:
            (visit_counts, root_value, root_policy) 三元组：
            - visit_counts: 225 维各动作原始访问计数
            - root_value: 根节点平均价值（从当前玩家视角）
            - root_policy: 根节点先验策略分布（施加噪声前的模型原始 softmax，225 维）
        """
        sims = num_simulations if num_simulations is not None else self.num_simulations
        self._init_search(state)

        # 初始化日志
        log_data = None
        if self._logger is not None and log_key is not None:
            log_data = self._logger.init_log_data(
                board=state[:225],
                current_player=state[225],
                simulations=sims,
                dirichlet_epsilon=self.dirichlet_epsilon,
                dirichlet_alpha=self.dirichlet_alpha,
                root_noise=self._get_root_noise(),
            )

        # 主搜索循环
        simulation_logs = []
        sim_id = 0

        while sim_id < sims:
            # 第一阶段：选择 → 收集叶节点（应用 Virtual Loss）
            leaf_nodes = self._select_and_collect(sims - sim_id)

            # 第二阶段：Batch 推理 + 扩展
            self._batch_evaluate(leaf_nodes)

            # 第三阶段：回传（撤销 Virtual Loss + Backup）
            for leaf_info in leaf_nodes:
                self._backup(leaf_info)

            # 记录日志
            if log_data is not None:
                for leaf_info in leaf_nodes:
                    sim_log = self._logger.build_simulation_log(
                        sim_id, leaf_info.get("path", []), leaf_info, self.tree
                    )
                    simulation_logs.append(sim_log)
                    sim_id += 1
            else:
                sim_id += len(leaf_nodes)

        # 组装结果
        visit_counts, root_value, root_policy = self._get_result()

        # 写入日志
        if log_data is not None and log_key is not None:
            log_data["simulation_logs"] = simulation_logs
            log_data["result"] = {
                "visit_counts": visit_counts,
                "root_value": root_value,
                "pi": self._compute_pi(visit_counts, temperature=1.0),
            }
            self._logger.write_log(log_key, log_data)

        return visit_counts, root_value, root_policy

    def advance_root(self, action: int) -> bool:
        """将 action 对应子节点提升为新根，丢弃兄弟子树。

        Args:
            action: 要推进的动作索引 0-224

        Returns:
            True 如果子节点存在并成功提升
        """
        if self._root_hash is None or self._root_hash not in self.tree:
            return False

        root = self.tree[self._root_hash]
        if action not in root["edges"]:
            return False

        edge = root["edges"][action]
        child_hash = edge.get("child_hash")
        if child_hash is None or child_hash not in self.tree:
            return False

        # 收集子树中所有节点哈希（递归）
        subtree_hashes = self._collect_subtree(child_hash)

        # 清除不在子树中的节点
        old_hashes = list(self.tree.keys())
        for h in old_hashes:
            if h not in subtree_hashes:
                del self.tree[h]

        self._root_hash = child_hash
        return True

    def _collect_subtree(self, root_hash: str) -> set:
        """收集以 root_hash 为根的子树中所有节点哈希（DFS）"""
        result = set()
        stack = [root_hash]
        while stack:
            h = stack.pop()
            if h in result or h not in self.tree:
                continue
            result.add(h)
            for edge in self.tree[h].get("edges", {}).values():
                ch = edge.get("child_hash")
                if ch is not None and ch not in result:
                    stack.append(ch)
        return result

    def reset(self):
        """清空搜索树"""
        self.tree.clear()
        self._root_hash = None

    # ---------- 内部：初始化 ----------

    @staticmethod
    def _state_hash(board: List[int], player: int) -> str:
        """计算状态确定性哈希（MD5 前 8 字符）"""
        data = bytes(board) + bytes([player])
        return hashlib.md5(data).hexdigest()[:8]

    def _init_search(self, state: List[int]):
        """初始化搜索：创建根节点 + 狄利克雷噪声
        
        若根节点已存在于树中（子树复用场景），仍重新施加新鲜噪声。
        """
        board = state[:225]
        player = state[225]
        root_hash = self._state_hash(board, player)

        # 记录根边 N 快照（子树复用时，边可能带有上轮搜索的 N）
        self._root_edge_N_snapshot = {}

        if root_hash not in self.tree:
            self.tree.clear()
            self._root_hash = root_hash

            # 模型预测
            policy, value = self.model.predict(state)

            # 创建根节点
            self.tree[root_hash] = {
                "board": list(board),
                "player": player,
                "N": 0,
                "W": 0.0,
                "Q": 0.0,
                "edges": {},
                "_root_policy": list(policy),
            }
            root = self.tree[root_hash]

            # 为所有空位创建边
            for action in range(225):
                if board[action] == 0:
                    root["edges"][action] = {
                        "N": 0, "W": 0.0, "Q": 0.0,
                        "P": policy[action], "child_hash": None,
                    }
            # 全新树：所有边初始 N=0，快照均为 0
            self._root_edge_N_snapshot = {a: 0 for a in range(225) if board[a] == 0}
        else:
            self._root_hash = root_hash
            # 子树复用：记录当前 N 快照，get_result 中只报告增量
            root = self.tree[root_hash]
            for action, edge in root.get("edges", {}).items():
                self._root_edge_N_snapshot[action] = edge["N"]

        # 始终施加狄利克雷噪声（包括子树复用场景）
        self._add_dirichlet_noise(root_hash)

    def _add_dirichlet_noise(self, node_hash: str):
        """对根节点的合法动作施加狄利克雷噪声
        
        以节点存储的 _root_policy（模型原始 softmax 输出）为基底混合噪声。
        支持子树复用后重新施加噪声（_root_policy 在 _expand_leaf 和 _init_search 中均会存储）。
        """
        node = self.tree[node_hash]
        edges = node["edges"]
        if not edges:
            return
        actions = list(edges.keys())
        n_actions = len(actions)
        noise = np.random.dirichlet([self.dirichlet_alpha] * n_actions)
        # 以 _root_policy 为基底（模型原始输出），若非根节点回退到当前 P 值
        root_policy = node.get("_root_policy")
        for i, action in enumerate(actions):
            base_P = root_policy[action] if root_policy else edges[action]["P"]
            edges[action]["P"] = (
                (1 - self.dirichlet_epsilon) * base_P
                + self.dirichlet_epsilon * noise[i]
            )

    def _get_root_noise(self) -> List[float]:
        """获取根节点噪声施加后的 P' 值（日志用）"""
        if self._root_hash is None or self._root_hash not in self.tree:
            return []
        root = self.tree[self._root_hash]
        return [round(root["edges"].get(a, {}).get("P", 0.0), 6) for a in range(225)]

    # ---------- 内部：选择阶段 ----------

    def _select_and_collect(self, remaining_sims: int) -> List[dict]:
        """选择阶段：沿 PUCT 下降收集叶节点，同时应用 Virtual Loss。

        Args:
            remaining_sims: 本轮剩余模拟次数

        Returns:
            叶节点信息列表，每项含 path, board, player, is_terminal 等
        """
        batch_size = min(self.mcts_batch_size, remaining_sims)
        leaf_nodes = []

        for _ in range(batch_size):
            path = []
            current_hash = self._root_hash

            while True:
                node = self.tree[current_hash]
                board = node["board"]
                player = node["player"]
                edges = node["edges"]

                # PUCT 选择最优子节点到叶节点
                best_action = self._puct_select(current_hash)
                if best_action is None:
                    # 无合法动作（终局或满棋盘）
                    leaf = self._make_terminal_leaf(current_hash, board, player, path)
                    leaf_nodes.append(leaf)
                    break

                edge = edges[best_action]
                path.append({
                    "state_hash": current_hash,
                    "action": best_action,
                    "player": player,
                })

                # 应用 Virtual Loss
                self._apply_virtual_loss(current_hash, best_action)

                child_hash = edge.get("child_hash")
                if child_hash is None:
                    # 子节点未展开 → 叶节点
                    new_board = list(board)
                    new_board[best_action] = player
                    new_player = 2 if player == 1 else 1

                    leaf_nodes.append({
                        "path": path,
                        "state_hash": self._state_hash(new_board, new_player),
                        "action": best_action,
                        "board": new_board,
                        "player": new_player,
                        "is_terminal": False,
                        "terminal_value": None,
                    })
                    break

                current_hash = child_hash

        return leaf_nodes

    def _puct_select(self, node_hash: str) -> Optional[int]:
        """PUCT 选择：a = argmax(Q + c_puct * P * sqrt(sum_N) / (1 + N))

        Returns:
            最优动作索引，无合法动作时返回 None
        """
        node = self.tree[node_hash]
        edges = node["edges"]
        board = node["board"]
        sum_N = sum(e["N"] for e in edges.values())
        sqrt_sum_N = math.sqrt(max(sum_N, 1))

        best_action = None
        best_score = float("-inf")

        for action, edge in edges.items():
            if board[action] != 0:
                continue
            Q = edge["Q"]
            U = self.c_puct * edge["P"] * sqrt_sum_N / (1 + edge["N"])
            score = Q + U
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    @staticmethod
    def _make_terminal_leaf(node_hash, board, player, path):
        """构造终局叶节点信息"""
        # 检测终局类型
        winner = 0
        for a in range(225):
            if board[a] != 0:
                w = check_winner(board, a)
                if w != 0:
                    winner = w
                    break
        if winner != 0:
            return {
                "path": path, "state_hash": node_hash,
                "board": list(board), "player": player,
                "is_terminal": True, "terminal_value": -1.0,
            }
        if is_board_full(board):
            return {
                "path": path, "state_hash": node_hash,
                "board": list(board), "player": player,
                "is_terminal": True, "terminal_value": 0.0,
            }
        return {
            "path": path, "state_hash": node_hash,
            "board": list(board), "player": player,
            "is_terminal": False, "terminal_value": None,
        }

    # ---------- 内部：Virtual Loss ----------

    def _apply_virtual_loss(self, node_hash: str, action: int):
        """对指定边施加 Virtual Loss"""
        edge = self.tree[node_hash]["edges"][action]
        edge["N"] += self.virtual_loss
        edge["W"] -= self.virtual_loss
        edge["Q"] = edge["W"] / edge["N"] if edge["N"] > 0 else 0.0

    def _undo_virtual_loss(self, node_hash: str, action: int):
        """撤销指定边的 Virtual Loss"""
        edge = self.tree[node_hash]["edges"][action]
        edge["N"] -= self.virtual_loss
        edge["W"] += self.virtual_loss
        edge["Q"] = edge["W"] / edge["N"] if edge["N"] > 0 else 0.0

    # ---------- 内部：Batch 推理 + 扩展 ----------

    def _batch_evaluate(self, leaf_nodes: List[dict]):
        """Batch 推理：对叶节点调用模型评估，并展开子边。

        终局节点由规则判定 value，非终局节点调用 model.predict。
        """
        # 分类：终局 vs 需要模型评估
        for leaf in leaf_nodes:
            if leaf.get("is_terminal", False):
                leaf["value"] = leaf.get("terminal_value", 0.0)
                continue

            # 检查棋盘状态
            winner = self._check_board_winner(leaf["board"])
            if winner != 0:
                leaf["is_terminal"] = True
                leaf["value"] = -1.0 if winner != leaf["player"] else 1.0
                continue
            if is_board_full(leaf["board"]):
                leaf["is_terminal"] = True
                leaf["value"] = 0.0
                continue

        # 对需要模型评估的节点逐个调用 predict
        for leaf in leaf_nodes:
            if leaf.get("is_terminal", False) or "value" in leaf:
                continue

            bv = leaf["board"] + [leaf["player"]]
            policy, value = self.model.predict(bv)
            leaf["policy"] = policy
            leaf["value"] = value

            # 扩展子节点
            self._expand_leaf(leaf, policy)

    @staticmethod
    def _check_board_winner(board: List[int]) -> int:
        """检查棋盘上是否有胜者"""
        for a in range(225):
            if board[a] != 0:
                w = check_winner(board, a)
                if w != 0:
                    return w
        return 0

    def _expand_leaf(self, leaf: dict, policy: List[float]):
        """为叶节点创建搜索树节点和边"""
        child_hash = self._state_hash(leaf["board"], leaf["player"])
        if child_hash not in self.tree:
            self.tree[child_hash] = {
                "board": list(leaf["board"]),
                "player": leaf["player"],
                "N": 0, "W": 0.0, "Q": 0.0,
                "edges": {},
                "_root_policy": list(policy),
            }
            child_node = self.tree[child_hash]
            for action in range(225):
                if leaf["board"][action] == 0:
                    child_node["edges"][action] = {
                        "N": 0, "W": 0.0, "Q": 0.0,
                        "P": policy[action], "child_hash": None,
                    }

        leaf["child_hash"] = child_hash

        # 在父节点边中记录 child_hash
        if "action" in leaf and leaf.get("path"):
            parent_info = leaf["path"][-1]
            parent_hash = parent_info["state_hash"]
            parent_action = leaf["action"]
            if parent_hash in self.tree and parent_action in self.tree[parent_hash]["edges"]:
                self.tree[parent_hash]["edges"][parent_action]["child_hash"] = child_hash

    # ---------- 内部：回传 ----------

    def _backup(self, leaf_info: dict):
        """回传阶段：沿路径向上更新 N/W/Q，每层取反 value。

        对路径上的每条边：
        1. 撤销 Virtual Loss
        2. v = -v（翻转为当前节点玩家视角）
        3. N += 1, W += v, Q = W / N
        """
        v = leaf_info.get("value", 0.0)

        for step in reversed(leaf_info.get("path", [])):
            node_hash = step["state_hash"]
            action = step["action"]

            # 撤销 Virtual Loss
            self._undo_virtual_loss(node_hash, action)

            # 翻转为当前节点玩家视角（叶节点 value 是对方视角）
            v = -v

            # 更新边统计量
            if node_hash in self.tree and action in self.tree[node_hash]["edges"]:
                edge = self.tree[node_hash]["edges"][action]
                edge["N"] += 1
                edge["W"] += v
                edge["Q"] = edge["W"] / edge["N"] if edge["N"] > 0 else 0.0

                # 更新节点统计量
                node = self.tree[node_hash]
                node["N"] += 1
                node["W"] += v
                node["Q"] = node["W"] / node["N"] if node["N"] > 0 else 0.0

    # ---------- 内部：结果组装 ----------

    def _get_result(self) -> Tuple[List[int], float, List[float]]:
        """从搜索树中提取搜索结果

        使用 N 增量计算 visit_counts，避免子树复用带来的跨步 N 残留。
        """
        root = self.tree.get(self._root_hash, {})
        edges = root.get("edges", {})

        visit_counts = [0] * 225
        for action, edge in edges.items():
            # 只报告本搜索新增的访问计数
            visit_counts[action] = edge["N"] - self._root_edge_N_snapshot.get(action, 0)

        root_value = root.get("Q", 0.0) if root.get("N", 0) > 0 else 0.0
        root_policy = root.get("_root_policy", [1.0 / 225] * 225)

        return visit_counts, root_value, root_policy

    @staticmethod
    def _compute_pi(visit_counts: List[int], temperature: float = 1.0) -> List[float]:
        """从访问计数计算 π 分布：π(a) = N(a)^(1/τ) / Σ N(b)^(1/τ)"""
        if temperature <= 0:
            max_count = max(visit_counts)
            max_indices = [i for i, c in enumerate(visit_counts) if c == max_count]
            pi = [0.0] * 225
            for i in max_indices:
                pi[i] = 1.0 / len(max_indices)
            return pi

        total = sum(c ** (1.0 / temperature) for c in visit_counts)
        if total > 0:
            return [(c ** (1.0 / temperature)) / total for c in visit_counts]
        return [1.0 / 225] * 225
