import pickle
import time


# ========= 环境 =========
class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9
        self.current_winner = None

    def print_board(self):
        for r in [self.board[i*3:(i+1)*3] for i in range(3)]:
            print('| ' + ' | '.join(r) + ' |')
        print()

    def available_moves(self):
        return [i for i,s in enumerate(self.board) if s==' ']

    def make_move(self, square, letter):
        if self.board[square]==' ':
            self.board[square]=letter
            if self.check_winner(square,letter):
                self.current_winner=letter
            return True
        return False

    def check_winner(self,sq,l):
        r=sq//3
        if all(self.board[r*3+i]==l for i in range(3)): return True
        c=sq%3
        if all(self.board[c+i*3]==l for i in range(3)): return True
        if sq%2==0:
            if all(self.board[i]==l for i in [0,4,8]): return True
            if all(self.board[i]==l for i in [2,4,6]): return True
        return False

    def is_draw(self):
        return ' ' not in self.board and self.current_winner is None

    def get_state(self):
        return ''.join(self.board)


# ========= AI =========
class AI:
    def __init__(self, qfile):
        q = pickle.load(open(qfile,'rb'))
        from book import QTableAdapter
        self.qtab = QTableAdapter(q)

    def get_q(self, player, s, a):
        return self.qtab.q_value(player, s, a)

    def choose(self,player,env):
        s = env.get_state()
        acts = env.available_moves()
        qs = [self.get_q(player, s, a) for a in acts]
        m = max(qs)
        best = [a for a, qv in zip(acts, qs) if qv == m]
        return best[0]


# ========= 双AI对弈 =========
def play(delay=1.0):
    env=TicTacToe()
    ai=AI("q.pkl")

    player='X'

    while True:
        env.print_board()
        time.sleep(delay)

        move=ai.choose(player,env)
        print(player,"moves to",move+1)

        env.make_move(move,player)

        if env.current_winner:
            env.print_board()
            print(player,"wins!")
            break

        if env.is_draw():
            env.print_board()
            print("Draw!")
            break

        player='O' if player=='X' else 'X'


if __name__=="__main__":
    play(delay=0.2)   # 每步0.8秒
