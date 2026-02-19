# export_opening_book_wdl_q_winprob_fast.py
# 目标：用你的 Q 表导出开局书，并同时输出 WDL + Q + win_prob
# 优化点：
# 1) Minimax 带 memo（每个(state,player)只算一次）→ 速度提升数量级
# 2) 全程纯函数 state-string 操作（不需要 env/make/undo/winner 回滚）→ 更快更稳
# 3) WDL 来自 minimax（稳定）；Q/win_prob 仅作附加信息；排序 W>D>L, 再按Q, 再按action稳定

import pickle
from typing import Dict, Tuple, Any, List, Optional

# ======================
# 配置
# ======================
QTABLE_PATH = "q.pkl"
OUT_PATH    = "opening_book.txt"
DEPTH       = 9      # 井字棋建议 9（完整）
TOP_K       = 0      # 每节点最多展开前K个动作（0=全部）

WIN_REWARD  = 1.0
LOSE_REWARD = -1.0

# 如果你训练时对 state/action 做了规范化（对称归一化），可在这里启用并实现对应映射。
# 下面默认关闭：按原始 board 字符串查Q
NORMALIZE_FOR_LOOKUP = False


# ======================
# 棋盘与规则（纯函数）
# state: 长度9的字符串，字符为 'X','O',' '
# ======================
WIN_LINES = [
    (0,1,2),(3,4,5),(6,7,8),
    (0,3,6),(1,4,7),(2,5,8),
    (0,4,8),(2,4,6),
]

def other(player: str) -> str:
    return 'O' if player == 'X' else 'X'

def available_moves(state: str) -> List[int]:
    return [i for i,ch in enumerate(state) if ch == ' ']

def apply_move(state: str, action: int, player: str) -> str:
    # 假设 action 合法
    lst = list(state)
    lst[action] = player
    return ''.join(lst)

def winner(state: str) -> Optional[str]:
    for a,b,c in WIN_LINES:
        if state[a] != ' ' and state[a] == state[b] == state[c]:
            return state[a]
    return None

def is_draw(state: str) -> bool:
    return (' ' not in state) and (winner(state) is None)


# ======================
# （可选）对称归一化
# 默认关闭。若你训练时确实对 state/action 做过对称化，建议你把这里与训练保持一致。
# ======================
def _rot90(s: str) -> str:
    idx = [6,3,0,7,4,1,8,5,2]
    return ''.join(s[i] for i in idx)

def _flip_h(s: str) -> str:
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
# Q 表适配器（仅做“格式兼容”）
# 兼容两种常见 key：
# 1) (player, state, action)
# 2) (state, action)
# 注意：这里默认不做“8种对称试探”，因为那会带来不稳定和额外开销。
# 若你训练时确实规范化了 state/action，请开启 NORMALIZE_FOR_LOOKUP 并确保训练&导出一致。
# ======================
class QTableAdapter:
    def __init__(self, q: Dict[Tuple[Any, ...], float]):
        self.q = q
        self.mode = self._detect_mode()

    def _detect_mode(self) -> str:
        for k in self.q.keys():
            if isinstance(k, tuple):
                if len(k) == 3:
                    return "player_state_action"
                if len(k) == 2:
                    return "state_action"
        return "unknown"

    def q_value(self, player: str, state: str, action: int) -> float:
        s = normalize_state(state) if NORMALIZE_FOR_LOOKUP else state

        if self.mode == "player_state_action":
            return float(self.q.get((player, s, action), 0.0))
        if self.mode == "state_action":
            return float(self.q.get((s, action), 0.0))
        return 0.0


# ======================
# Q -> win_prob（仅展示用；不是“真实胜率”）
# WIN=1, LOSE=-1 时，win_prob=(Q+1)/2 裁剪到[0,1]
# ======================
def q_to_win_prob(q: float) -> float:
    if abs(WIN_REWARD - LOSE_REWARD) < 1e-12:
        return 0.5
    p = (q - LOSE_REWARD) / (WIN_REWARD - LOSE_REWARD)
    return max(0.0, min(1.0, p))


# ======================
# Minimax（带 memo，稳定且快）
# 返回值：从“当前 player（要走的人）视角”出发的结果
# +1 必胜, 0 至少平, -1 必败
# ======================
_minimax_memo: Dict[Tuple[str, str], int] = {}

def minimax_value(state: str, player: str) -> int:
    key = (state, player)
    if key in _minimax_memo:
        return _minimax_memo[key]

    w = winner(state)
    if w is not None:
        v = 1 if w == player else -1
        _minimax_memo[key] = v
        return v
    if is_draw(state):
        _minimax_memo[key] = 0
        return 0

    best = -2
    for a in available_moves(state):
        s2 = apply_move(state, a, player)
        v = -minimax_value(s2, other(player))
        if v > best:
            best = v
            if best == 1:
                break  # 已经找到必胜走法，不必继续
    _minimax_memo[key] = best
    return best


# ======================
# 构建开局树：WDL + Q + win_prob
# 注意：WDL来自minimax（稳定）；Q/win_prob仅作注释信息
# ======================
def build_tree(state: str, player: str, depth: int, qtab: QTableAdapter) -> Dict[int, Any]:
    if depth <= 0:
        return {}
    if winner(state) is not None or is_draw(state):
        return {}

    node: Dict[int, Any] = {}
    moves = available_moves(state)

    for a in moves:
        q = qtab.q_value(player, state, a)
        wp = q_to_win_prob(q)

        s2 = apply_move(state, a, player)
        v = -minimax_value(s2, other(player))  # 因为换边

        wdl = "W" if v == 1 else ("D" if v == 0 else "L")

        child = build_tree(s2, other(player), depth - 1, qtab)
        node[a] = {
            "WDL": wdl,
            "value": v,
            "Q": q,
            "win_prob": wp,
            "child": child
        }

    # 排序：W > D > L，再按 Q（大优先），再按 action（小优先）稳定
    order = {"W": 2, "D": 1, "L": 0}
    items = sorted(
        node.items(),
        key=lambda kv: (-order[kv[1]["WDL"]], -kv[1]["Q"], kv[0])
    )
    if TOP_K and TOP_K > 0:
        items = items[:TOP_K]
    return dict(items)


# ======================
# 输出
# ======================
def format_board(state: str) -> str:
    def ch(i: int) -> str:
        return state[i] if state[i] != ' ' else '.'
    return "\n".join([
        f"{ch(0)} {ch(1)} {ch(2)}",
        f"{ch(3)} {ch(4)} {ch(5)}",
        f"{ch(6)} {ch(7)} {ch(8)}",
    ])

def write_tree_txt(tree: Dict[int, Any], state: str, player: str, f, indent: int = 0) -> None:
    pad = " " * indent
    f.write(pad + f"Player {player} to move\n")
    f.write(pad + format_board(state).replace("\n", "\n" + pad) + "\n\n")

    if not tree:
        w = winner(state)
        if w:
            f.write(pad + f"==> Terminal: {w} wins\n\n")
        elif is_draw(state):
            f.write(pad + "==> Terminal: Draw\n\n")
        else:
            f.write(pad + "==> Truncated\n\n")
        return

    for a, info in tree.items():
        f.write(
            pad + f"- move {a}  WDL={info['WDL']}  "
                  f"Q={info['Q']:.6f}  win_prob={info['win_prob']:.3f}\n"
        )
    f.write("\n")

    for a, info in tree.items():
        f.write(pad + f"-> choose move {a}\n")
        s2 = apply_move(state, a, player)
        write_tree_txt(info["child"], s2, other(player), f, indent + 4)


# ======================
# main
# ======================
def main():
    with open(QTABLE_PATH, "rb") as f:
        q = pickle.load(f)

    qtab = QTableAdapter(q)

    root = " " * 9
    # 清空 minimax memo（可选）
    _minimax_memo.clear()

    tree = build_tree(root, "X", DEPTH, qtab)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("# Opening Book (WDL + Q + win_prob)\n")
        f.write(f"# depth={DEPTH}, top_k={TOP_K} (0 means all)\n")
        f.write(f"# Q-table mode={qtab.mode}\n")
        f.write(f"# NOTE: WDL is from minimax (stable). win_prob is just a linear map of Q.\n\n")
        write_tree_txt(tree, root, "X", f, 0)

    print("Opening book exported:", OUT_PATH)
    print("Detected Q-table mode:", qtab.mode)
    print("Minimax memo size:", len(_minimax_memo))

if __name__ == "__main__":
    main()
