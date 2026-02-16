import random
import pickle
import os


# ======================
# 环境
# ======================
class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9
        self.current_winner = None

    def reset(self):
        self.board = [' '] * 9
        self.current_winner = None
        return self.get_state()

    def get_state(self):
        return ''.join(self.board)

    def available_moves(self):
        return [i for i,s in enumerate(self.board) if s==' ']

    def make_move(self, square, letter):
        if self.board[square] != ' ':
            return False
        self.board[square] = letter
        if self.check_winner(square, letter):
            self.current_winner = letter
        return True

    def check_winner(self, square, letter):
        r = square//3
        if all(self.board[r*3+i]==letter for i in range(3)):
            return True
        c = square%3
        if all(self.board[c+i*3]==letter for i in range(3)):
            return True
        if square%2==0:
            if all(self.board[i]==letter for i in [0,4,8]): return True
            if all(self.board[i]==letter for i in [2,4,6]): return True
        return False

    def is_draw(self):
        return ' ' not in self.board and self.current_winner is None


# ======================
# Agent (Minimax-Q)
# ======================
class QAgent:
    def __init__(self, epsilon=1.0, alpha=0.1, gamma=0.95):
        self.q={}
        self.epsilon=epsilon
        self.alpha=alpha
        self.gamma=gamma

    def get_q(self,p,s,a):
        return self.q.get((p,s,a),0)

    def choose(self,p,s,acts):
        if random.random()<self.epsilon:
            return random.choice(acts)
        qs=[self.get_q(p,s,a) for a in acts]
        m=max(qs)
        best=[a for a in acts if self.get_q(p,s,a)==m]
        return random.choice(best)

    def update(self,p,s,a,r,s2,next_acts,opp,done):
        if done:
            target=r
        else:
            # opponent chooses worst for us
            opp_vals=[max(self.get_q(opp,s2,oa) for oa in next_acts)]
            target=r+self.gamma*(-max(opp_vals))  # minimax

        old=self.get_q(p,s,a)
        self.q[(p,s,a)]=old+self.alpha*(target-old)

    def decay(self):
        self.epsilon=max(0.05,self.epsilon*0.9999)

    def save(self,f):
        pickle.dump(self.q,open(f,'wb'))
    def load(self,f):
        if os.path.exists(f):
            self.q=pickle.load(open(f,'rb'))


# ======================
# self-play
# ======================
def play(agent,env,train=True):
    s=env.reset()
    p='X'

    while True:
        acts=env.available_moves()
        a=agent.choose(p,s,acts)
        env.make_move(a,p)
        s2=env.get_state()

        done=env.current_winner or env.is_draw()

        if done:
            if env.current_winner==p: r=1
            elif env.is_draw(): r=0
            else: r=-1
            next_acts=[]
        else:
            r=0
            next_acts=env.available_moves()

        if train:
            opp='O' if p=='X' else 'X'
            agent.update(p,s,a,r,s2,next_acts,opp,done)

        if done:
            return env.current_winner

        s=s2
        p='O' if p=='X' else 'X'


# ======================
# evaluate
# ======================
def eval(agent,env,n=100):
    e=agent.epsilon
    agent.epsilon=0
    res={'X':0,'O':0,'D':0}
    for _ in range(n):
        w=play(agent,env,False)
        if w: res[w]+=1
        else: res['D']+=1
    agent.epsilon=e
    return res


# ======================
# train
# ======================
def train(agent,env,eps):
    for i in range(1,eps+1):
        play(agent,env,True)
        agent.decay()
        if i%1000==0:
            print(i,eval(agent,env))


# ======================
if __name__=="__main__":
    env=TicTacToe()
    ag=QAgent()
    ag.load("q.pkl")

    train(ag,env,1000000)

    ag.save("q.pkl")
    print("Training completed.")