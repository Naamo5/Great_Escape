import random
import time
import numpy as np
from itertools import product

class Game():
    """docstring for ClassName"""
    def __init__(self, player, enemy):
        self.player = player
        self.enemy = enemy
        self.exit_pos = [4,4]
        self.S_dim = 5*6*5*6+2 # state dimension all combinaison of position + eaten + escaped
        self.init_v = np.zeros(self.S_dim) # initial state values
        self.r_not_escaped = 0.0
        self.r_eaten = -2
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
                if y_p == y_e and x_p == x_e:
                    pij[S,self.S_dim-2,idx] = 1.0
                elif [y_p, x_p] == self.exit_pos and action == 'S':
                    pij[S,self.S_dim-1,idx] = 1.0
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
                rewards[s, -1] = self.r_escaped
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
        print(' ' + '_'*8 + ' ')
        print('|{}{}|{}{} {}{}|'.format(*vis_board[0,:]))
        print('|{}{}|{}{}|{}{}|'.format(*vis_board[1,:]))
        print('|{}{}|{}{}|{}\u0305{}\u0305|'.format(*vis_board[2,:]))
        print('|{}{} {}{} {}{}|'.format(*vis_board[3,:]))
        print('|{}{}\u203e{}\u0305{}\u0305|\u0305{}\u0305{}\u0305|'.format(
            *vis_board[4,:]))
        print(' ' + '\u203e'*8 + ' ')
        time.sleep(0.5)

    def BW_induction(self, time):
        """ Back ward induction algorithm for a finite horizon MDP problem """
        u2 = np.zeros((self.S_dim, time))
        """ Setting of the last colomn (for final time) of the value function  """
        u2[:,time-1] = [np.max([self.rewards[s,a] for a in range(self.player.A_dim)]) for s in range(self.S_dim)]
        """ The initial policy is to Stay """
        policy = 4*np.ones((self.S_dim, time))
        """ Iterates through time, state and action """
        for t in range(1,time):
            """ Starting from T-1 to 0 """
            print(time-1-t)
            u1 = np.zeros(self.S_dim)
            for s1 in range(self.S_dim):
                u_temp = np.zeros(self.player.A_dim)
                for a in range(self.player.A_dim):
                    u_temp[a] = sum([self.pij[s1,s2,a] * (self.rewards[s1,a] + u2[s2,time-t]) for s2 in range(self.S_dim)])
                u1[s1] = max(u_temp) #value function for s1 at time-1-t
                policy[s1,time-1-t] = np.argmax(u_temp) #optimal policy for s1 at time-1-t
            u2[:,time-1-t] = np.copy(u1)
        return policy,u2
 
    def opt_policy(self,policy, min_pos):
        """ Find the optimal policy for a given set of minautor positions """
        time = len(policy[0])
        actions = [0]*15
        self.enemy.pos = min_pos[0]
        self.player.pos = [0,0]
        actions[0] = policy[self.tostate(self.player.pos, self.player.pos),0]
        self.display_board()
        for t in range(0,time-1):
            self.enemy.pos = min_pos[t+1]
            self.player.pos = self.player.transition(self.player.pos,self.player.actions[int(actions[t])])
            actions[t+1] = policy[self.tostate(self.player.pos, self.enemy.pos),t+1]
            self.display_board()
        return actions
    
    def proba(self, T):
        pass



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
    def __init__(self, can_same=False):
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
                prob = 1.0/(4 + self.can_same)
            elif (yn == y and xn == x):
                prob = (1.0 + self.can_same)/(4 + self.can_same)
            else:
                prob = 0.0
        elif x == 5 and y > 0 and y < 4:
            if (yn == (y + 1) and xn == x) or \
               (yn == (y - 1) and xn == x) or \
               (yn == y and xn == (x - 1)):
                prob = 1.0/(4 + self.can_same)
            elif (yn == y and xn == x):
                prob = (1.0 + self.can_same)/(4 + self.can_same)
            else:
                prob = 0.0
        elif y == 0 and x > 0 and x < 5:
            if (yn == (y + 1) and xn == x) or \
               (yn == y and xn == (x - 1)) or \
               (yn == y and xn == (x + 1)):
                prob = 1.0/(4 + self.can_same)
            elif (yn == y and xn == x):
                prob = (1.0 + self.can_same)/(4 + self.can_same)
            else:
                prob = 0.0
        elif y == 4 and x > 0 and x < 5:
            if (yn == (y - 1) and xn == x) or \
               (yn == y and xn == (x - 1)) or \
               (yn == y and xn == (x + 1)):
                prob = 1.0/(4 + self.can_same)
            elif (yn == y and xn == x):
                prob = (1.0 + self.can_same)/(4 + self.can_same)
            else:
                prob = 0.0
        # CORNER POSITIONS
        elif y==0 and x==0:
            if (yn == (y + 1) and xn == x) or \
               (yn == y and xn == (x + 1)):
                prob = 1.0/(4 + self.can_same)
            elif (yn == y and xn == x):
                prob = (2.0 + self.can_same)/(4 + self.can_same)
            else:
                prob = 0.0
        elif y==4 and x==0:
            if (yn == (y - 1) and xn == x) or \
               (yn == y and xn == (x + 1)):
                prob = 1.0/(4 + self.can_same)
            elif (yn == y and xn == x):
                prob = (2.0 + self.can_same)/(4 + self.can_same)
            else:
                prob = 0.0
        elif y==0 and x==5:
            if (yn == (y + 1) and xn == x) or \
               (yn == y and xn == (x - 1)):
                prob = 1.0/(4 + self.can_same)
            elif (yn == y and xn == x):
                prob = (2.0 + self.can_same)/(4 + self.can_same)
            else:
                prob = 0.0
        elif y==4 and x==5:
            if (yn == (y - 1) and xn == x) or \
               (yn == y and xn == (x - 1)):
                prob = 1.0/(4 + self.can_same)
            elif (yn == y and xn == x):
                prob = (2.0 + self.can_same)/(4 + self.can_same)
            else:
                prob = 0.0
        return prob

    def test_transition(self):
        iters = [5, 6, 5, 6]
        ranges = [range(x) for x in iters]
        for y, x, yn, xn in product(*ranges):
            prob = self.transition([y,x],[yn,xn])
            print('p{{({},{})|({},{})}} = {}'.format(yn,xn,y,x,prob))

Gary = Player()
Minotaur = Enemy()
TheGame = Game(Gary, Minotaur)


opt_policies, u2 = TheGame.BW_induction(15)
minautor_pos = [[4,4], [4,3], [3,3], [3,2], [3,1],[3,2], [3,1],[3,2], [3,1],[3,2], [3,1],[3,2], [3,1],[3,2], [3,1],[3,2], [3,1],[3,2], [3,1]]

actions = TheGame.opt_policy(opt_policies, minautor_pos)