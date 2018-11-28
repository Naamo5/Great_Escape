import random
import time
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

class Game():
    """docstring for ClassName"""
    def __init__(self, player, enemy):
        self.player = player
        self.enemy = enemy
        self.exit_pos = [4,4]
        self.S_dim = 5*6*5*6+2 # state dimension all combinaison of position + eaten + escaped
        self.init_v = np.zeros(self.S_dim) # initial state values
        self.r_not_escaped = 0.0
        self.r_eaten = 0.0
        self.r_escaped = 1.0
        self.pij = self.calc_pij()
        self.rewards = self.calc_rewards()

    def tostate(self, pos_p, pos_e):
        return 30*(pos_p[0]*6 + pos_p[1]) + pos_e[0]*6 + pos_e[1]

    def fromstate(self, s):
        if s<self.S_dim-2:
            pos_p = s // 30
            y_p = pos_p // 6
            x_p = pos_p % 6
            rem = s % 30
            y_e = rem // 6
            x_e = rem % 6
            ret = [y_p, x_p], [y_e, x_e]
        else:
            ret = s
        return ret

    def calc_pij(self):
        pij = np.zeros((self.S_dim,self.S_dim,len(self.player.actions)))
        ''' Iterates through player_pos, enemy_pos and new_enemy_pos'''
        iters = [5, 6, 5, 6, 5, 6]
        ranges = [range(x) for x in iters]
        for y_p, x_p, y_e, x_e, yn_e, xn_e in product(*ranges):
            for idx, action in enumerate(self.player.actions):
                S = self.tostate([y_p,x_p],[y_e,x_e])
                posn_p = self.player.transition([y_p,x_p], action)
                prob = self.enemy.transition([y_e,x_e],[yn_e,xn_e])
                Sn = self.tostate(posn_p, [yn_e,xn_e])
                pij[S,Sn,idx] = prob
                ''' terminal states are recursive: final time step, minotaur kills player, player escapes maze '''
                if [y_p, x_p] == self.exit_pos:
                    pij[S,Sn,idx] = 0
                    pij[S,self.S_dim-1,idx] = 1.0
                if y_p == y_e and x_p == x_e:
                    pij[S,Sn,idx] = 0
                    pij[S,self.S_dim-2,idx] = 1.0
        """ When the player is already eaten or already escaped, whatever he does, he will stay in the same state"""
        for idx, action in enumerate(self.player.actions):
            #When he was eaten
            pij[self.S_dim-2,self.S_dim-2,idx] = 1.0
            #When he escaped
            pij[self.S_dim-1,self.S_dim-1,idx] = 1.0
        return pij

    def calc_rewards(self):
        rewards = np.zeros((self.S_dim, self.player.A_dim))
        for s in range(self.S_dim-2):
            [y_p,x_p],[y_e,x_e] = self.fromstate(s)
            if [y_p, x_p] == self.exit_pos:
                rewards[s,4] = self.r_escaped
            if y_p == y_e and x_p == x_e:
                rewards[s] = [self.r_eaten]*self.player.A_dim
        return rewards

    def test_rewards(self):
        for S in range(self.S_dim):
            if S < self.S_dim-2:
                [y_p, x_p], [y_e, x_e] = self.fromstate(S)
                print('player = [{},{}], minotaur = [{},{}], reward = {}'.format(y_p, x_p, y_e, x_e, self.rewards[S]))
            elif S == self.S_dim-2:
                print('Eaten, reward = {}'.format(self.rewards[S]))
            else:
                print('Escaped, reward = {}'.format(self.rewards[S]))

    def display_board(self):
        vis_board = np.empty((5,6), dtype='str')
        vis_board[:] = ' '
        if self.player.pos == self.enemy.pos:
            vis_board[tuple(self.player.pos)] = 'B' # Both in same position
        else:
            vis_board[tuple(self.player.pos)] = 'P' # Player
            vis_board[tuple(self.enemy.pos)] = 'M' # Minotaur
        print('╭' + '─'*5 +'┬' + '─'*11 + '╮') #╭─────────────────╮
        print('│ {} {} │ {} {}   {} {} │'.format(*vis_board[0,:])) #│
        print('│ {} {} │ {} {} │ {} {} │'.format(*vis_board[1,:]))
        print('│     │     ├─────┤')
        print('│ {} {} │ {} {} │ {} {} │'.format(*vis_board[2,:]))
        print('│ {} {}   {} {}   {} {} │'.format(*vis_board[3,:]))
        print('│   ────────┬───  │')
        print('│ {} {}   {} {} │ {} {} │'.format(
            *vis_board[4,:]))
        print('╰' + '─'*11 +'┴'+'─'*5+ '╯') #╰─────────────────╯
        time.sleep(0.6)

    def BW_induction(self, time):
        """ Back ward induction algorithm for a finite horizon MDP problem """
        u2 = np.zeros((self.S_dim, time))
        """ Setting of the last colomn (for final time) of the value function  """
        u2[:,-1] = [np.max([self.rewards[s,a] for a in range(self.player.A_dim)]) for s in range(self.S_dim)]
        """ The initial policy is to Stay """
        policy = 5*np.ones((self.S_dim, time-1))
        """ Iterates through time, state and action """
        for t in reversed(range(1,time)):
            """ Starting from indice T-1 to 1 """
            utemp = np.zeros((self.S_dim,self.player.A_dim))
            for a in range(self.player.A_dim):
                utemp[:,a] = self.rewards[:,a] + np.matmul(self.pij[:,:,a],u2[:,t])
            u1 = [np.max(utemp[s,:]) for s in range(self.S_dim)] #value function for s1 at time t-1
            policy[:,t-1] = [np.argmax(utemp[s,:]) for s in range(self.S_dim)] #optimal policy for s1 at time-1-t
            u2[:,t-1] = np.copy(u1)
        return policy,u2
 
    def opt_policy(self,policy, min_pos, verbose):
        """ Find the optimal policy for a given set of minautor positions """
        T = len(policy[0])+1
        actions = [0]*(T-1)
        self.enemy.pos = min_pos[0]
        self.player.pos = [0,0]
        actions[0] = policy[self.tostate(self.player.pos, self.enemy.pos),0]
        if verbose:
            print("\n \n Time = 1")
            self.display_board()
        for t in range(1,T):
            self.enemy.pos = min_pos[t]
            self.player.pos = self.player.transition(self.player.pos,self.player.actions[int(actions[t-1])])
            if self.enemy.pos == self.player.pos:
                if verbose:
                    print("\n \n Time = {}".format(t+1))
                    self.display_board()
                    print("\n \n Time = {}".format(t+2))
                    print(' _____           _                  ')
                    print('| ____|   __ _  | |_    ___   _ __  ')
                    print('|  _|    / _  | | __|  / _ \ |  _ \ ')
                    print('| |___  | (_| | | |_  |  __/ | | | |')
                    print('|_____|  \__,_|  \__|  \___| |_| |_|')
                    time.sleep(0.7)
                return False
            elif self.player.pos == self.exit_pos and self.enemy.pos != self.player.pos:
                if verbose:
                    print("\n \n Time = {}".format(t+1))
                    self.display_board()
                    print("\n \n Time = {}".format(t+2))
                    print(' _____                                           _ ')
                    print('| ____|  ___    ___    __ _   _ __     ___    __| |')
                    print('|  _|   / __|  / __|  / _  | |  _ \   / _ \  / _  |')
                    print('| |___  \__ \ | (__  | (_| | | |_) | |  __/ | (_| |')
                    print('|_____| |___/  \___|  \__,_| | .__/   \___|  \__,_|')
                    print('                             |_|                   ')
                    time.sleep(0.7)
                return True
            if t<len(actions):
                actions[t] = policy[self.tostate(self.player.pos, self.enemy.pos),t]
            if verbose:
                    print("\n \n Time = {}".format(t+1))
                    self.display_board()
        if verbose:
                    print("\n \n Time = {}".format(t+2))
                    print('  _____           _   _              _ ')
                    print(' |  ___|   __ _  (_) | |   ___    __| |')
                    print(' | |_     / _` | | | | |  / _ \  / _` |')
                    print(' |  _|   | (_| | | | | | |  __/ | (_| |')
                    print(' |_|      \__,_| |_| |_|  \___|  \__,_|')
                    time.sleep(0.7)
        return False
        
    def proba(self, T, verbose):
        policies = self.BW_induction(T)[0]
        iter = 10000
        win=0
        for i in range(iter):
            minotaur_pos = Minotaur.random_path(T)
            if TheGame.opt_policy(policies, minotaur_pos, verbose):
                win = win +1
        return win/iter



class Player():
    """docstring for ClassName"""
    def __init__(self):
        self.pos = [0,0]
        self.actions = ['U','D','L','R','S']
        self.A_dim = len(self.actions)
        self.actdict = {'U': (-1,0),
                        'D': (1,0),
                        'L': (0,-1),
                        'R': (0,1),
                        'S': (0,0)}

    def transition(self, pos, action):
        # ACCEPTS: old position, action
        # RETURNS: new position
        # NOTE: assumes invalid action returns same position!
        y, x = pos[0], pos[1]
        # edge proximity actions
        if (y == 0 and action == 'U') or \
           (y == 4 and action == 'D') or \
           (x == 0 and action == 'L') or \
           (x == 5 and action == 'R'):
           pos_new = [y,x]
        # wall proximity actions going left to right, top to bottom
        elif (x == 1 and y < 3 and action == 'R') or \
             (x == 2 and y < 3 and action == 'L') or \
             (x == 3 and y > 0 and y < 3 and action == 'R') or \
             (x == 4 and y > 0 and y < 3 and action == 'L') or \
             (y == 1 and x > 3 and action == 'D') or \
             (y == 2 and x > 3 and action == 'U') or \
             (y == 3 and x > 0 and x < 5 and action == 'D') or \
             (y == 4 and x > 0 and x < 5 and action == 'U') or \
             (y == 4 and x == 3 and action == 'R') or \
             (y == 4 and x == 4 and action == 'L'):
             pos_new = [y,x]
        else:
            pos_new = [y + x for y, x in zip(pos, self.actdict[action])]
        return pos_new

    def test_transition(self):
        for y in range(5):
            for x in range(6):
                for action in self.actions:
                    yn, xn = Gary.transition([y, x], action)
                    print('[{},{}] > [{},{}] via {}'.format(y,x,yn,xn,action))


class Enemy():
    """docstring  for ClassName"""
    def __init__(self, can_same):
        self.pos = [4,4]
        self.can_same = can_same # Can it remain in same place, see transition

    def transition(self, oldpos, newpos):
        # ACCEPTS: old position, new position
        # RETURNS: probability of transition
        # NOTE: assumes invalid action results in same position!
        y, x = oldpos[0], oldpos[1]
        yn, xn = newpos[0], newpos[1]
        # CENTRAL RECTANGLE POSITIONS
        if y > 0 and y < 4 and x > 0 and x < 5:
            if (yn == (y + 1) and xn == x) or \
               (yn == (y - 1) and xn == x) or \
               (yn == y and xn == (x + 1)) or \
               (yn == y and xn == (x - 1)):
                prob = 1.0/(4 + self.can_same)
            elif (yn == y and xn == x):
                prob = 1 - (4.0/(4 + self.can_same))
            else:
                prob = 0.0
        # EDGE POSITIONS
        elif x == 0 and y > 0 and y < 4:
            if (yn == (y + 1) and xn == x) or \
               (yn == (y - 1) and xn == x) or \
               (yn == y and xn == (x + 1)):
                prob = 1.0/(3 + self.can_same)
            elif (yn == y and xn == x):
                prob = self.can_same/(3 + self.can_same)
            else:
                prob = 0.0
        elif x == 5 and y > 0 and y < 4:
            if (yn == (y + 1) and xn == x) or \
               (yn == (y - 1) and xn == x) or \
               (yn == y and xn == (x - 1)):
                prob = 1.0/(3 + self.can_same)
            elif (yn == y and xn == x):
                prob = self.can_same/(3 + self.can_same)
            else:
                prob = 0.0
        elif y == 0 and x > 0 and x < 5:
            if (yn == (y + 1) and xn == x) or \
               (yn == y and xn == (x - 1)) or \
               (yn == y and xn == (x + 1)):
                prob = 1.0/(3 + self.can_same)
            elif (yn == y and xn == x):
                prob = self.can_same/(3 + self.can_same)
            else:
                prob = 0.0
        elif y == 4 and x > 0 and x < 5:
            if (yn == (y - 1) and xn == x) or \
               (yn == y and xn == (x - 1)) or \
               (yn == y and xn == (x + 1)):
                prob = 1.0/(3 + self.can_same)
            elif (yn == y and xn == x):
                prob = self.can_same/(3 + self.can_same)
            else:
                prob = 0.0
        # CORNER POSITIONS
        elif y==0 and x==0:
            if (yn == (y + 1) and xn == x) or \
               (yn == y and xn == (x + 1)):
                prob = 1.0/(2 + self.can_same)
            elif (yn == y and xn == x):
                prob = self.can_same/(2 + self.can_same)
            else:
                prob = 0.0
        elif y==4 and x==0:
            if (yn == (y - 1) and xn == x) or \
               (yn == y and xn == (x + 1)):
                prob = 1.0/(2 + self.can_same)
            elif (yn == y and xn == x):
                prob = self.can_same/(2 + self.can_same)
            else:
                prob = 0.0
        elif y==0 and x==5:
            if (yn == (y + 1) and xn == x) or \
               (yn == y and xn == (x - 1)):
                prob = 1.0/(2 + self.can_same)
            elif (yn == y and xn == x):
                prob = self.can_same/(2 + self.can_same)
            else:
                prob = 0.0
        elif y==4 and x==5:
            if (yn == (y - 1) and xn == x) or \
               (yn == y and xn == (x - 1)):
                prob = 1.0/(2 + self.can_same)
            elif (yn == y and xn == x):
                prob = self.can_same/(2 + self.can_same)
            else:
                prob = 0.0
        return prob
    
    def neighbors_box(self, pos):
        y, x = pos[0], pos[1]
        u = [y-1,x]
        d = [y+1,x] 
        l = [y,x-1]
        r = [y,x+1]
        if self.can_same:
            boxes = [u, d, l, r, pos]
        else:
            boxes = [u, d, l, r]
        return boxes

    def random_path(self, T):
        path = [[4,4]]
        oldpos = path[0]
        for t in range(T-1):
            newpos = random.choice(self.neighbors_box(oldpos))
            while self.transition(oldpos, newpos) == 0:
                newpos = random.choice(self.neighbors_box(oldpos))
            path.append(newpos)
            oldpos = newpos
        return path
    
    def test_transition(self):
        iters = [5, 6, 5, 6]
        ranges = [range(x) for x in iters]
        for y, x, yn, xn in product(*ranges):
            prob = self.transition([y,x],[yn,xn])
            print('p{{({},{})|({},{})}} = {}'.format(yn,xn,y,x,prob))

Gary = Player()
Minotaur = Enemy(True)
TheGame = Game(Gary, Minotaur)

p,u2 = TheGame.BW_induction(11)

#min_pos = [[4,4],[3,4],[3,3],[3,4], [3,3],[3,4], [3,3],[3,4], [3,3],[3,4], [3,3],[3,4], [3,3],[3,4], [3,3]]
#TheGame.opt_policy(p, min_pos, True)


# Plot the graphs

proba=[]
Tmin = 2
Tmax = 40
for t in range(Tmin,Tmax+1):
    print(t)
    proba.append(TheGame.proba(t, False))

if Minotaur.can_same:
    plt.plot(np.arange(Tmin,Tmax+1),proba)
    plt.xticks(np.arange(1,Tmax+2,4))
    plt.xlabel(r'Time available $T$')
    plt.ylabel(r'Probability to escape the maze $P_T ( \{ Escaped \} )$')
    plt.savefig('pro_can.ps')
else:
    L = [1.0]*30
    L[0:9] = [0.0]*9
    L[9:13] = proba 
    plt.plot(np.arange(2,32),L)
    plt.plot([12],[L[11]],'ro')
    plt.text(12,L[11], r'  $(T = 11, P_T=${})'.format(L[11]))
    plt.xticks(np.arange(1,31,2))
    plt.xlabel(r'Time available $T$')
    plt.ylabel(r'Probability to escape the maze $P_T ( \{ Escaped \} )$')
    plt.savefig('pro_cannot.ps')


