# export_opening_book_winprob.py
# 读取你训练得到的 q-table（支持两种 key 形式），生成“对弈树/开局书”，并把每一步的 Q 值与“己方胜率预测”写入 txt。
#
# 用法：
#   python export_opening_book_winprob.py
#
# 你可以在下面修改：
#   QTABLE_PATH, OUT_PATH, DEPTH, TOP_K, WIN_REWARD, LOSE_REWARD, DRAW_REWARD

import os
import math
import pickle
from typing import Dict, Tuple, Any, List, Optional

# ======================
# 配置
# ======================
QTABLE_PATH = "q.pkl"          # 你的模型文件：可能叫 q.pkl 或 qtable.pkl
OUT_PATH    = "opening_book.txt"
DEPTH       = 9               # 展开深度（步数），井字棋建议 5~9
TOP_K       = 0               # 每个节点最多展开前 K 个动作（按 win_prob 排序）。设为 0 表示展开全部动作。
WIN_REWARD  = 1.0             # 训练时 win 的奖励（用于映射胜率）
LOSE_REWARD = -1.0            # 训练时 lose 的奖励（用于映射胜率）
DRAW_REWARD = 0.0             # 训练时 draw 的奖励（不直接用于映射，但用于理解范围）


# ======================
# 环境
# ======================
class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9
        self.current_winner: Optional[str] = None

    def reset(self) -> str:
        self.board = [' '] * 9
        self.current_winner = None
        return self.get_state()

    def get_state(self) -> str:
        return ''.join(self.board)

    def available_moves(self) -> List[int]:
        return [i for i, s in enumerate(self.board) if s == ' ']

    def make_move(self, square: int, letter: str) -> bool:
        if self.board[square] != ' ':
            return False
        self.board[square] = letter
        if self.check_winner(square, letter):
            self.current_winner = letter
        return True

    def undo_move(self, square: int) -> None:
        self.board[square] = ' '
        self.current_winner = None  # 重新计算 winner 太麻烦，这里简单置空（对生成树足够）

    def check_winner(self, square: int, letter: str) -> bool:
        r = square // 3
        if all(self.board[r * 3 + i] == letter for i in range(3)):
            return True
        c = square % 3
        if all(self.board[c + i * 3] == letter for i in range(3)):
            return True
        if square % 2 == 0:
            if all(self.board[i] == letter for i in [0, 4, 8]):
                return True
            if all(self.board[i] == letter for i in [2, 4, 6]):
                return True
        return False

    def is_draw(self) -> bool:
        return (' ' not in self.board) and (self.current_winner is None)


# ======================
# 对称归一化（建议与你训练时保持一致）
# 下面实现包含 8 种对称中的 6 种也能工作，但最好用完整 8 种。
# 我这里给你完整 8 种（旋转+镜像）版本。
# ======================
def _rot90(s: str) -> str:
    # 0 1 2      6 3 0
    # 3 4 5  ->  7 4 1
    # 6 7 8      8 5 2
    idx = [6,3,0,7,4,1,8,5,2]
    return ''.join(s[i] for i in idx)

def _flip_h(s: str) -> str:
    # 水平翻转（左右镜像）
    # 0 1 2      2 1 0
    # 3 4 5  ->  5 4 3
    # 6 7 8      8 7 6
    idx = [2,1,0,5,4,3,8,7,6]
    return ''.join(s[i] for i in idx)

def normalize_state(state: str) -> str:
    s0 = state
    s1 = _rot90(s0)
    s2 = _rot90(s1)
    s3 = _rot90(s2)
    f0 = _flip_h(s0)
    f1 = _rot90(f0)
    f2 = _rot90(f1)
    f3 = _rot90(f2)
    return min([s0,s1,s2,s3,f0,f1,f2,f3])


# ======================
# Q 表适配器：兼容两种常见格式
# 1) key=(player, state, action)  -> 这是你后来修复后的格式
# 2) key=(state, action)          -> 你早期代码可能是这种（不区分 player）
# 并且 state 可能已经被 normalize 过，也可能没 normalize
# ======================
class QTableAdapter:
    def __init__(self, q: Dict[Tuple[Any, ...], float]):
        self.q = q
        self.mode = self._detect_mode()

        # 预计算 8 种对称的 index 映射（new_pos -> old_pos），与 normalize_state 使用的变换一致。
        base = '012345678'
        s0 = base
        s1 = _rot90(s0)
        s2 = _rot90(s1)
        s3 = _rot90(s2)
        f0 = _flip_h(s0)
        f1 = _rot90(f0)
        f2 = _rot90(f1)
        f3 = _rot90(f2)
        self._sym_idx_maps = []
        for s in [s0, s1, s2, s3, f0, f1, f2, f3]:
            self._sym_idx_maps.append([int(ch) for ch in s])

    def _detect_mode(self) -> str:
        # 粗检测：取一个 key 看长度
        for k in self.q.keys():
            if isinstance(k, tuple):
                if len(k) == 3:
                    return "player_state_action"
                if len(k) == 2:
                    return "state_action"
        # 空表或奇怪结构
        return "unknown"

    def q_value(self, player: str, state: str, action: int) -> float:
        # 试几种可能的 key：考虑到训练时可能对 state 做了对称化并同时变换了 action 的索引，
        # 我们对 8 种对称都尝试 state+action 的映射以兼容各种保存方式。
        ns = normalize_state(state)

        # 1) 如果表是 (player,state,action) 形式，优先尝试在 8 个对称下查找
        if self.mode == "player_state_action":
            # 尝试对称变换后的 (player, state_sym, action_sym)
            for idx_map in self._sym_idx_maps:
                state_sym = ''.join(state[i] for i in idx_map)
                try:
                    action_sym = idx_map.index(action)
                except ValueError:
                    action_sym = action
                k = (player, state_sym, action_sym)
                if k in self.q:
                    return self.q[k]
            # 退回到未变换的查找
            return self.q.get((player, ns, action), self.q.get((player, state, action), 0.0))

        # 2) 如果表是 (state,action) 形式，同样尝试对称映射
        if self.mode == "state_action":
            for idx_map in self._sym_idx_maps:
                state_sym = ''.join(state[i] for i in idx_map)
                try:
                    action_sym = idx_map.index(action)
                except ValueError:
                    action_sym = action
                k = (state_sym, action_sym)
                if k in self.q:
                    return self.q[k]
            return self.q.get((ns, action), self.q.get((state, action), 0.0))

        return 0.0


# ======================
# Q -> 己方胜率映射
# 最稳妥：线性映射到 [0,1]，并裁剪：
#   p = (Q - LOSE) / (WIN - LOSE)
# 当 WIN=1, LOSE=-1 时就是 (Q+1)/2
# 注意：如果你的奖励把 draw 设成 100、-100 之类，会让 Q 远超范围，
# 这时我们会裁剪到 [0,1]（仍可作为“相对强弱”指标）。
# ======================
def q_to_win_prob(q: float, win: float = WIN_REWARD, lose: float = LOSE_REWARD) -> float:
    if abs(win - lose) < 1e-9:
        return 0.5
    p = (q - lose) / (win - lose)
    return max(0.0, min(1.0, p))


# ======================
# 生成树（可控制每层展开 TOP_K）
# ======================
def build_opening_tree(qtab: QTableAdapter, env: TicTacToe, depth: int, top_k: int) -> Dict[str, Any]:
    visited = {}

    def expand(player: str, d: int) -> Dict[int, Any]:
        state = env.get_state()
        key = (normalize_state(state), player, d)
        if key in visited:
            return visited[key]

        # 终局
        if env.current_winner is not None or env.is_draw() or d == 0:
            visited[key] = {}
            return {}

        moves = env.available_moves()
        scored = []
        for a in moves:
            q = qtab.q_value(player, state, a)
            p = q_to_win_prob(q)
            scored.append((a, q, p))

        # 按“己方胜率”排序（高到低），再按 action 作为稳定 tie-break
        scored.sort(key=lambda x: (-x[2], -x[1], x[0]))

        if top_k and top_k > 0:
            scored = scored[:top_k]

        node: Dict[int, Any] = {}
        for a, q, p in scored:
            # 执行动作
            env.make_move(a, player)
            next_player = 'O' if player == 'X' else 'X'
            child = expand(next_player, d - 1)
            # 回退
            env.undo_move(a)

            node[a] = {
                "Q": q,
                "win_prob": p,
                "child": child,
            }

        visited[key] = node
        return node

    env.reset()
    return {
        "root_player": "X",
        "depth": depth,
        "top_k": top_k,
        "tree": expand("X", depth),
    }


# ======================
# 写入 txt：包含棋盘 + 每步 Q + 己方胜率
# ======================
def format_board(state: str) -> str:
    def ch(i: int) -> str:
        return state[i] if state[i] != ' ' else '.'
    rows = [
        f"{ch(0)} {ch(1)} {ch(2)}",
        f"{ch(3)} {ch(4)} {ch(5)}",
        f"{ch(6)} {ch(7)} {ch(8)}",
    ]
    return "\n".join(rows)

def write_tree_txt(payload: Dict[str, Any], out_path: str) -> None:
    tree = payload["tree"]
    depth = payload["depth"]
    top_k = payload["top_k"]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# opening book exported from Q-table\n")
        f.write(f"# depth={depth}, top_k={top_k} (0 means all actions)\n")
        f.write(f"# win/lose used for win_prob mapping: WIN={WIN_REWARD}, LOSE={LOSE_REWARD}\n\n")

        def rec(node: Dict[int, Any], env: TicTacToe, player: str, indent: int) -> None:
            state = env.get_state()
            f.write(" " * indent + f"Player {player} to move\n")
            f.write(" " * indent + format_board(state).replace("\n", "\n" + " " * indent) + "\n")

            if not node:
                # 终局或截断
                if env.current_winner:
                    f.write(" " * indent + f"==> Terminal: {env.current_winner} wins\n\n")
                elif env.is_draw():
                    f.write(" " * indent + f"==> Terminal: Draw\n\n")
                else:
                    f.write(" " * indent + f"==> Truncated\n\n")
                return

            # 输出本节点的所有候选动作（已经按 win_prob 排序）
            for a, info in node.items():
                f.write(" " * indent + f"- move {a}  Q={info['Q']:.6f}  win_prob(self)={info['win_prob']:.3f}\n")
            f.write("\n")

            # 继续递归展开每个子节点
            for a, info in node.items():
                f.write(" " * indent + f"-> choose move {a}\n")
                env.make_move(a, player)
                next_player = 'O' if player == 'X' else 'X'
                rec(info["child"], env, next_player, indent + 4)
                env.undo_move(a)

        env = TicTacToe()
        env.reset()
        rec(tree, env, "X", 0)


# ======================
# main
# ======================
def main():
    # if not os.path.exists(QTABLE_PATH):
    #     # 兼容你可能用的文件名
    #     alt = "q.pkl"
    #     if os.path.exists(alt):
    #         global QTABLE_PATH
    #         QTABLE_PATH = alt
    #     else:
    #         raise FileNotFoundError(f"Cannot find Q-table file: {QTABLE_PATH} (or {alt})")

    with open(QTABLE_PATH, "rb") as f:
        q = pickle.load(f)

    qtab = QTableAdapter(q)
    env = TicTacToe()

    payload = build_opening_tree(qtab, env, depth=DEPTH, top_k=TOP_K)
    write_tree_txt(payload, OUT_PATH)
    print(f"Saved opening book to: {OUT_PATH}")
    print(f"Detected Q-table mode: {qtab.mode}")


if __name__ == "__main__":
    main()
