import random
import time
import numpy as np
from itertools import product

class PlayerBase():
    """docstring for ClassName"""
    def __init__(self):
        self.actions = ['U','D','L','R','S']
        self.actdict = {'U': (-1,0),
                        'D': (1,0),
                        'L': (0,-1),
                        'R': (0,1),
                        'S': (0,0)}

    def test_transition(self, H, W):
        for y in range(H):
            for x in range(W):
                for action in self.actions:
                    yn, xn = self.transition([y, x], action, H, W)
                    print('[{},{}] > [{},{}] via {}'.format(y,x,yn,xn,action))

    def into_edge(self, pos, action, H, W):
        # returns true if action walks player into edge of map
        y, x = pos[0], pos[1]
        if (y == 0 and action == 'U') or \
           (y == (H - 1) and action == 'D') or \
           (x == 0 and action == 'L') or \
           (x == (W - 1) and action == 'R'):
            return True
        else:
            return False

    def transition(self, pos, action, H, W):
        # ACCEPTS: old position, action
        # RETURNS: new position
        # NOTE: assumes invalid action returns same position!
        if self.into_edge(pos, action, H, W):
            posn_p = pos
        else:
            posn_p = [y + x for y, x in zip(pos, self.actdict[action])]
        return posn_p


class Ex1Player(PlayerBase):
    """docstring for Ex1Player"""
    def __init__(self):
        super(Ex1Player, self).__init__()
        self.pos = [0,0]

    def transition(self, pos, action, H, W):
        y, x = pos[0], pos[1]
        # edge proximity actions
        if self.into_edge(pos, action, H, W):
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


class Ex1Enemy():
    """docstring  for ClassName"""
    def __init__(self, can_same=False):
        self.can_same = can_same

    def test_transition(self, H, W):
        iters = [H, W, H, W]
        ranges = [range(x) for x in iters]
        for y, x, yn, xn in product(*ranges):
            prob = self.transition([y,x], [yn,xn], H, W)
            print('p{{({},{})|({},{})}} = {}'.format(yn,xn,y,x,prob))
        
    def transition(self, oldpos, newpos, H, W):
        # ACCEPTS: old position, new position
        # RETURNS: probability of transition
        # NOTE: assumes invalid action results in same position!
        y, x = oldpos[0], oldpos[1]
        yn, xn = newpos[0], newpos[1]
        # CENTRAL RECTANGLE POSITIONS
        prob = 0.0
        if y > 0 and y < (H-1) and x > 0 and x < (W-1):
            if (yn == (y + 1) and xn == x) or \
               (yn == (y - 1) and xn == x) or \
               (yn == y and xn == (x + 1)) or \
               (yn == y and xn == (x - 1)):
                prob = 1.0/(4 + self.can_same)
            elif (yn == y and xn == x):
                prob = self.can_same/(4.0 + self.can_same)
        # EDGE POSITIONS
        elif x == 0 and y > 0 and y < (H-1):
            if (yn == (y + 1) and xn == x) or \
               (yn == (y - 1) and xn == x) or \
               (yn == y and xn == (x + 1)):
                prob = 1.0/(3 + self.can_same)
            elif (yn == y and xn == x):
                prob = self.can_same/(3.0 + self.can_same)
        elif x == (W-1) and y > 0 and y < (H-1):
            if (yn == (y + 1) and xn == x) or \
               (yn == (y - 1) and xn == x) or \
               (yn == y and xn == (x - 1)):
                prob = 1.0/(3 + self.can_same)
            elif (yn == y and xn == x):
                prob = self.can_same/(3.0 + self.can_same)
        elif y == 0 and x > 0 and x < (W-1):
            if (yn == (y + 1) and xn == x) or \
               (yn == y and xn == (x - 1)) or \
               (yn == y and xn == (x + 1)):
                prob = 1.0/(3 + self.can_same)
            elif (yn == y and xn == x):
                prob = self.can_same/(3.0 + self.can_same)
        elif y == (H-1) and x > 0 and x < (W-1):
            if (yn == (y - 1) and xn == x) or \
               (yn == y and xn == (x - 1)) or \
               (yn == y and xn == (x + 1)):
                prob = 1.0/(3 + self.can_same)
            elif (yn == y and xn == x):
                prob = self.can_same/(3.0 + self.can_same)
        # CORNER POSITIONS
        elif y==0 and x==0:
            if (yn == (y + 1) and xn == x) or \
               (yn == y and xn == (x + 1)):
                prob = 1.0/(2 + self.can_same)
            elif (yn == y and xn == x):
                prob = self.can_same/(2.0 + self.can_same)
        elif y==(H-1) and x==0:
            if (yn == (y - 1) and xn == x) or \
               (yn == y and xn == (x + 1)):
                prob = 1.0/(2 + self.can_same)
            elif (yn == y and xn == x):
                prob = self.can_same/(2.0 + self.can_same)
        elif y==0 and x==(W-1):
            if (yn == (y + 1) and xn == x) or \
               (yn == y and xn == (x - 1)):
                prob = 1.0/(2 + self.can_same)
            elif (yn == y and xn == x):
                prob = self.can_same/(2.0 + self.can_same)
        elif y==(H-1) and x==(W-1):
            if (yn == (y - 1) and xn == x) or \
               (yn == y and xn == (x - 1)):
                prob = 1.0/(2 + self.can_same)
            elif (yn == y and xn == x):
                prob = self.can_same/(2.0 + self.can_same)
        return prob


class Ex2Enemy():
    """docstring  for ClassName"""
    def __init__(self, can_same=False):
        self.pos = [1,2]
        self.can_same = can_same # Can it remain in same place, see transition

    def test_transition(self, H, W):
        iters = [H, W, H, W, H, W]
        ranges = [range(x) for x in iters]
        for y_e, x_e, yn_e, xn_e, y_p, x_p in product(*ranges):
            prob = self.transition([y_e, x_e], [yn_e, xn_e], [y_p, x_p], H, W)
            print('[y_e,x_e] = [{},{}], [yn_e,xn_e] = [{},{}], '
                  '[y_p,x_p] = [{},{}], prob = {}'
                  .format(y_e, x_e, yn_e, xn_e, y_p, x_p, prob))
        
    def transition(self, pos_e, posn_e, pos_p, H, W):
        y_e, x_e = pos_e[0], pos_e[1]
        yn_e, xn_e = posn_e[0], posn_e[1]
        y_p, x_p = pos_e[0], pos_e[1]

        prob = 0.0 # Default probablity
        # PLAYER UL, UR, DL, DR of POLICE
        if y_p < y_e and x_p < x_e:
            if [yn_e, xn_e] in [[y_e-1,x_e],[y_e,x_e-1]]:
                prob = 0.5
        elif y_p < y_e and x_p > x_e:
            if [yn_e, xn_e] in [[y_e-1,x_e],[y_e,x_e+1]]:
                prob = 0.5
        elif y_p > y_e and x_p < x_e:
            if [yn_e, xn_e] in [[y_e+1,x_e],[y_e,x_e-1]]:
                prob = 0.5
        elif y_p > y_e and x_p > x_e:
            if [yn_e, xn_e] in [[y_e+1,x_e],[y_e,x_e+1]]:
                prob = 0.5
        # PLAYER L, R, U, D of POLICE
        elif y_p == y_e and x_p < x_e:
            if y_e == 0:
                if [yn_e, xn_e] in [[y_e+1,x_e],[y_e,x_e-1]]:
                    prob = 0.5
            elif y_e == (H-1):
                if [yn_e, xn_e] in [[y_e-1,x_e],[y_e,x_e-1]]:
                    prob = 0.5
            else:
                if [yn_e, xn_e] in [[y_e-1,x_e],[y_e+1,x_e],[y_e,x_e-1]]:
                    prob = 1.0/3
        elif y_p == y_e and x_p > x_e:
            if y_e == 0:
                if [yn_e, xn_e] in [[y_e+1,x_e],[y_e,x_e+1]]:
                    prob = 0.5
            elif y_e == (H-1):
                if [yn_e, xn_e] in [[y_e-1,x_e],[y_e,x_e+1]]:
                    prob = 0.5
            else:
                if [yn_e, xn_e] in [[y_e-1,x_e],[y_e+1,x_e],[y_e,x_e+1]]:
                    prob = 1.0/3
        elif y_p < y_e and x_p == x_e:
            if x_e == 0:
                if [yn_e, xn_e] in [[y_e-1,x_e],[y_e,x_e+1]]:
                    prob = 0.5
            elif x_e == (W-1):
                if [yn_e, xn_e] in [[y_e-1,x_e],[y_e,x_e-1]]:
                    prob = 0.5
            else:
                if [yn_e, xn_e] in [[y_e-1,x_e],[y_e,x_e-1],[y_e,x_e+1]]:
                    prob = 1.0/3
        elif y_p > y_e and x_p == x_e:
            if x_e == 0:
                if [yn_e, xn_e] in [[y_e+1,x_e],[y_e,x_e+1]]:
                    prob = 0.5
            elif x_e == (W-1):
                if [yn_e, xn_e] in [[y_e+1,x_e],[y_e,x_e-1]]:
                    prob = 0.5
            else:
                if [yn_e, xn_e] in [[y_e-1,x_e],[y_e,x_e+1],[y_e,x_e-1]]:
                    prob = 1.0/3
        # REMAIN SAME IF PLAYER CAUGHT
        elif pos_e == pos_p and pos_e == posn_e:
            prob = 1.0
        return prob


class GameBase():
    """docstring for ClassName"""
    def __init__(self, can_same=False):
        self.can_same = can_same

    def init_2(self):
        self.S_dim = (self.W*self.H)**2
        self.calc_pij()
        self.calc_rewards()

    def tostate(self, pos_p, pos_e):
        return self.H*self.W*(pos_p[0]*self.W + pos_p[1]) + \
               pos_e[0]*self.W + pos_e[1]

    def fromstate(self, s):
        pos_p = s // (self.H*self.W)
        pos_e = s % (self.H*self.W)
        y_p = pos_p // self.W
        x_p = pos_p % self.W
        y_e = pos_e // self.W
        x_e = pos_e % self.W
        return [y_p, x_p], [y_e, x_e]

    def test_pij(self):
        # some sanity checks
        self.print_stats()
        is_rowsum_one = 0
        for action in range(len(self.player.actions)):
            rowsum = np.sum(self.pij[:,:,action],1)
            non_zero_rows = rowsum!=1.0
            state_list = np.arange(self.S_dim)
            non_zero_states = state_list[non_zero_rows]
            for S in non_zero_states:
                self.print_error_S(action, S)
                non_zero_states_new = state_list[self.pij[S,:,action]>0.0]
                if len(non_zero_states_new) == 0:
                    print('prob = 0.0 for all S\'')
                for Sn in non_zero_states_new:
                    self.print_error_Sn(S, Sn, action)
                print('')
            if sum(rowsum) == self.pij.shape[1]:
                is_rowsum_one += 1
        if is_rowsum_one == len(self.player.actions):
            print('all rows sum to 1.0 for each action :)')


    def test_rewards(self):
        iters = [self.H, self.W, self.H, self.W]
        ranges = [range(x) for x in iters]
        for y_p, x_p, y_e, x_e in product(*ranges):
            for idx, action in enumerate(self.player.actions):
                S = self.tostate([y_p,x_p],[y_e,x_e])
                print('player = [{},{}], enemy = [{},{}], '
                      'action = {}, reward = {}'
                      .format(y_p, x_p, y_e, x_e, action, self.rewards[S,idx]))

    def print_error_S(self, action, S):
        [y_p, x_p], [y_e, x_e] = self.fromstate(S)
        print('the following state/action row does not sum to 1.0:')
        print('[y_p,x_p] = [{},{}], [y_e,x_e] = [{},{}], A = {}'
              .format(y_p, x_p, y_e, x_e, self.player.actions[action]))
        print('details:')

    def print_error_Sn(self, S, Sn, action):
        [yn_p, xn_p], [yn_e, xn_e] = self.fromstate(Sn)
        print('[yn_p,xn_p] = [{},{}], [yn_e,xn_e] = [{},{}], prob = {}'
              .format(tn, yn_p, xn_p, yn_e, xn_e, self.pij[S,Sn,action]))

    def print_stats(self):
        non_zero = np.sum(self.pij[self.pij>0])
        non_zero_frac = 1.0*non_zero/self.pij.size
        no_terminal = np.sum(self.pij==1.0)
        no_terminal_frac = 1.0*no_terminal/self.pij.size
        other = non_zero - no_terminal
        other_frac = 1.0*other/self.pij.size
        print('no of states = {:.0f}'.format(self.S_dim))
        print('no of actions = {:.0f}'.format(len(self.player.actions)))
        print('no of elements = {:.0f}'.format(self.pij.size))
        print('no of terminal elements = {:.0f}'.format(no_terminal))
        print('fraction of terminal elements = {:.5f}'
              .format(no_terminal_frac))
        print('no of elements x where 0 < x < 1 = {:.0f}'.format(other))
        print('fraction of elements x where 0 < x < 1 = {:.5f}'
              .format(other_frac))
        print('no of non-zero elements = {:.0f}'.format(non_zero))
        print('fraction of non zero elements = {:.5f}\n\n'
              .format(non_zero_frac))

    def one_step(self, S, A):
        # probability of each Sn given S,A
        prob = self.pij[S,:,A]
        prob = prob/np.sum(prob) # in case of sum(prob) slightly > 1
        S = np.random.choice(np.arange(self.S_dim), p=prob)
        return S


class Ex1Game(GameBase):
    """docstring for ClassName"""
    def __init__(self, *args, **kwargs):
        super(Ex1Game, self).__init__(*args, **kwargs)
        self.player = Ex1Player()
        self.enemy = Ex1Enemy(self.can_same)
        self.enemy.pos = [4,5]
        self.H, self.W = 5, 6
        self.exit_pos = [4,4]
        self.r_not_escaped = -1.0
        self.r_eaten = -1.0
        self.r_escaped = 0.0
        self.init_2()

    def is_terminal(self, pos_p, pos_e):
        return pos_p == pos_e or pos_p == self.exit_pos
    
    def calc_pij(self):
        pij = np.zeros((self.S_dim,self.S_dim,len(self.player.actions)),
                       dtype=np.float32)
        ''' Iterates through player_pos, enemy_pos and new_enemy_pos, because we know new_player_pos from player.transition function.'''
        iters = [self.H, self.W, self.H, self.W, self.H, self.W]
        ranges = [range(x) for x in iters]
        for y_p, x_p, y_e, x_e, yn_e, xn_e in product(*ranges):
            for idx, action in enumerate(self.player.actions):
                S = self.tostate([y_p,x_p],[y_e,x_e])
                if self.is_terminal([y_p,x_p],[y_e,x_e]):
                    pij[S,S,idx] = 1.0
                else:
                    posn_p = self.player.transition([y_p,x_p], action, 
                                                    self.H, self.W)
                    prob = self.enemy.transition([y_e,x_e],[yn_e,xn_e],
                                                 self.H, self.W)
                    Sn = self.tostate(posn_p, [yn_e,xn_e])
                    pij[S,Sn,idx] = prob
                ''' terminal states are recursive: minotaur kills player, player escapes maze '''
        self.pij = pij

    def calc_rewards(self):
        rewards = np.ones((self.S_dim,len(self.player.actions))) * \
                  self.r_not_escaped
        iters = [self.H, self.W, self.H, self.W]
        ranges = [range(x) for x in iters]
        for y_p, x_p, y_e, x_e in product(*ranges):
            for idx, action in enumerate(self.player.actions):
                S = self.tostate([y_p,x_p],[y_e,x_e])
                posn_p = self.player.transition([y_p,x_p], action, 
                                                    self.H, self.W)
                if posn_p == self.exit_pos:
                    rewards[S, idx] = self.r_escaped
                if posn_p == [y_e, x_e]:
                    rewards[S, idx] = self.r_eaten
        self.rewards = rewards

    def get_optimal(self, time):
        """ Back ward induction algorithm for a finite horizon MDP problem """
        u2 = np.zeros((self.S_dim, time+1))
        """ Setting of the last colomn (for final time) of the value function  """
        u2[:,time] = np.ones(self.S_dim)*self.r_not_escaped
        policy = np.zeros((self.S_dim, time),dtype=np.int32)
        """ Iterates through time, state and action """
        for t in reversed(range(time)):
            """ Starting from T-1 to 0 """
            u1 = np.zeros(self.S_dim)
            for s1 in range(self.S_dim):
                u_temp = np.zeros(len(self.player.actions))
                for a in range(len(self.player.actions)):
                    u_temp[a] = np.sum(self.pij[s1,:,a] * u2[:,t+1]) + self.rewards[s1,a]
                u1[s1] = max(u_temp) #value function for s1 at time-1-t
                policy[s1,t] = np.argmax(u_temp) #optimal policy for s1 at time-1-t
            u2[:,t] = np.copy(u1)
        self.v_opt = u2
        self.p_opt = policy

    def get_optimal2(self, T):
        p_opt = np.zeros((self.S_dim, T),dtype=np.int32) # optimal policy
        v_opt = np.zeros((self.S_dim, T+1)) # optimal value
        v_opt[:,T] = np.ones(self.S_dim)*self.r_not_escaped
        for t in reversed(range(T)):
            max_val = np.zeros((self.S_dim, len(self.player.actions)))
            for action in range(len(self.player.actions)):
                # Pij*S for all S and Action=a
                v_opt_temp = v_opt[:,t]
                v_opt_temp.shape = (self.S_dim,1)
                mult = v_opt_temp*self.pij[:,:,action]
                temp = np.sum(mult,1)
                max_val[:,action] = temp
            v_optn_a = max_val + self.rewards
            v_opt[:,t] = np.max(v_optn_a,1)
            p_opt[:,t] = np.argmax(v_optn_a,1)
        print(v_opt)
        self.v_opt = v_opt
        self.p_opt = p_opt

    def simulate(self, T, verbose=True):
        S = self.tostate(self.player.pos, self.enemy.pos) # initial state
        if verbose:
            self.display_board()
        for t in range(T):
            A = self.p_opt[S, t]
            S = self.one_step(S,A)
            pos_p, pos_e = self.fromstate(S) # get new positions
            self.player.pos, self.enemy.pos = pos_p, pos_e
            if verbose:
                self.display_board()
            else:
                if t==T-1 or pos_p == pos_e:
                    return 0.0
                elif pos_p == self.exit_pos:
                    return 1.0

    def display_board(self):
        vis_board = np.empty((self.H,self.W), dtype='str')
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
        print('|{}{}\u203e{}\u0305{}\u0305|\u0305{}\u0305{}\u0305|'
              .format(*vis_board[4,:]))
        print(' ' + '\u203e'*8 + ' ')
        time.sleep(0.3)


class Ex2Game(GameBase):
    """docstring for Ex2Game"""
    def __init__(self, *args, **kwargs):
        super(Ex2Game, self).__init__(*args, **kwargs)
        self.player = PlayerBase()
        self.enemy = Ex2Enemy()
        self.H, self.W = 3, 6
        self.bank_pos = [[0,0],[2,0],[0,5],[2,5]]
        self.r_bank = 10.0
        self.r_caught = -50.0
        self.init_2()

    def is_bank(self, pos_p, pos_e):
        if pos_p in self.bank_pos and pos_p != pos_e:
            return True
        else:
            return False

    def is_caught(self, pos_p, pos_e):
        return True if pos_p == pos_e else False

    def calc_pij(self):
        pij = np.zeros((self.S_dim,self.S_dim,len(self.player.actions)),
                       dtype=np.float32)
        iters = [self.H, self.W, self.H, self.W, self.H, self.W]
        ranges = [range(x) for x in iters]
        for y_p, x_p, y_e, x_e, yn_e, xn_e in product(*ranges):
            for idx, action in enumerate(self.player.actions):
                S = self.tostate([y_p,x_p],[y_e,x_e])
                if self.is_caught([y_p,x_p],[y_e,x_e]):
                    pij[S,S,idx] = 1.0
                else:
                    posn_p = self.player.transition([y_p,x_p], action, 
                                                    self.H, self.W)
                    prob = self.enemy.transition([y_e,x_e], [yn_e,xn_e], 
                                                 [y_p,x_p], self.H, self.W)
                    Sn = self.tostate(posn_p, [yn_e,xn_e])
                    pij[S,Sn,idx] = prob
        self.pij = pij

    def calc_rewards(self):
        rewards = np.zeros((self.S_dim,len(self.player.actions)))
        iters = [self.H, self.W, self.H, self.W]
        ranges = [range(x) for x in iters]
        for y_p, x_p, y_e, x_e in product(*ranges):
            for idx, action in enumerate(self.player.actions):
                S = self.tostate([y_p,x_p],[y_e,x_e])
                posn_p = self.player.transition([y_p,x_p], action, 
                                                    self.H, self.W)
                if self.is_bank(posn_p,[y_e,x_e]):
                    rewards[S, idx] = self.r_bank
                elif self.is_caught(posn_p,[y_e,x_e]):
                    rewards[S, idx] = self.r_caught
        self.rewards = rewards

    def display_board(self):
        vis_board = np.empty((self.H,self.W), dtype='str')
        vis_board[:] = ' '
        for bank in self.bank_pos:
            vis_board[tuple(bank)] = '$'
        if self.player.pos == self.enemy.pos:
            vis_board[tuple(self.player.pos)] = 'X' # caught
        else:
            vis_board[tuple(self.player.pos)] = 'R' # Robber
            vis_board[tuple(self.enemy.pos)] = 'P' # Police
        print(' ' + '_'*6 + ' ')
        print('|{}{}{}{}{}{}|'.format(*vis_board[0,:]))
        print('|{}{}{}{}{}{}|'.format(*vis_board[1,:]))
        print('|{}{}{}{}{}{}|'.format(*vis_board[2,:]))
        print(' ' + '\u203e'*6 + ' ')
        time.sleep(0.3)
        

class Ex3Game(GameBase):
    """docstring for Ex3Game"""
    def __init__(self, *args, **kwargs):
        super(Ex3Game, self).__init__(*args, **kwargs)
        self.player = PlayerBase()
        self.enemy = Ex1Enemy()
        self.enemy.pos = [3,3]
        self.H, self.W = 4, 4
        self.bank_pos = [1,1]
        self.r_bank = 1.0
        self.r_caught = -10.0
        self.lamb = 0.8
        self.init_2()

    def is_bank(self, pos_p, pos_e):
        if pos_p == self.bank_pos and pos_p != pos_e:
            return True
        else:
            return False

    def is_caught(self, pos_p, pos_e):
        return True if pos_p == pos_e else False

    def calc_pij(self):
        pij = np.zeros((self.S_dim,self.S_dim,len(self.player.actions)),
                       dtype=np.float32)
        iters = [self.H, self.W, self.H, self.W, self.H, self.W]
        ranges = [range(x) for x in iters]
        for y_p, x_p, y_e, x_e, yn_e, xn_e in product(*ranges):
            for idx, action in enumerate(self.player.actions):
                S = self.tostate([y_p,x_p],[y_e,x_e])
                if self.is_caught([y_p,x_p],[y_e,x_e]):
                    pij[S,S,idx] = 1.0
                else:
                    posn_p = self.player.transition([y_p,x_p], action, 
                                                    self.H, self.W)
                    prob = self.enemy.transition([y_e,x_e],[yn_e,xn_e],
                                                 self.H, self.W)
                    Sn = self.tostate(posn_p, [yn_e,xn_e])
                    pij[S,Sn,idx] = prob
        self.pij = pij

    def calc_rewards(self):
        rewards = np.zeros(self.S_dim)
        iters = [self.H, self.W, self.H, self.W]
        ranges = [range(x) for x in iters]
        for y_p, x_p, y_e, x_e in product(*ranges):
            S = self.tostate([y_p,x_p],[y_e,x_e])
            if self.is_bank([y_p,x_p],[y_e,x_e]):
                rewards[S] = self.r_bank
            elif self.is_caught(posn_p,[y_e,x_e]):
                rewards[S] = self.r_caught
        self.rewards = rewards

    def get_optimal(self, iters):
        # Q LEARNING ALGORITHM / SARSA
        n_actions = len(self.player.actions)
        Q = np.zeros(S_dim, n_actions) # init Q values
        S = self.tostate(self.player.pos, self.enemy.pos)
        for i in range(iters):
            # Set learning rate alpha
            alpha = 0.1 # !!!TODO!!!!
            # Select an action with equal probability
            A = np.choice(range(n_actions),p=[1.0/n_actions]*n_actions)
            # Perform 1 step of algorithm
            Sn = self.one_step(S,A)
            # Update Q value table
            Q[S,A] = alpha*(R[Sn] + self.lamb*np.max(Q[Sn,:]) - Q[S,A])
            # If bank robber is caught, reinitialise game
            posn_p, posn_e = self.from_state(Sn)
            if self.is_caught(posn_p, posn_e):
                self.player.pos = [0,0]
                self.enemy.pos = [3,3]
                S = self.tostate(self.player.pos, self.enemy.pos)
            else:
                S = Sn
        self.p_opt = np.max(Q,1)

    def simulate(self, verbose=False):
        pass


    def display_board(self):
        vis_board = np.empty((self.H,self.W), dtype='str')
        vis_board[:] = ' '
        vis_board[tuple(self.bank_pos)] = '$'
        if self.player.pos == self.enemy.pos:
            vis_board[tuple(self.player.pos)] = 'X' # caught
        else:
            vis_board[tuple(self.player.pos)] = 'R' # Robber
            vis_board[tuple(self.enemy.pos)] = 'P' # Police
        print(' ' + '_'*4 + ' ')
        print('|{}{}{}{}|'.format(*vis_board[0,:]))
        print('|{}{}{}{}|'.format(*vis_board[1,:]))
        print('|{}{}{}{}|'.format(*vis_board[2,:]))
        print('|{}{}{}{}|'.format(*vis_board[3,:]))
        print(' ' + '\u203e'*4 + ' ')
        time.sleep(0.3)

def Ex1():
    maxtime = 30 # highest time we wish to find policy for
    no_sims = 100
    for can_same in [False, True]:
        for time in range(10, maxtime): # impossible to win for t < 10
            MazeEscape = Ex1Game(time, can_same)
            MazeEscape.get_optimal(time)
            escaped = 0.0
            for sim in range(no_sims):
                MazeEscape.player.pos = [0,0]
                MazeEscape.enemy.pos = [4,5]
                escaped += MazeEscape.simulate(time,False)
            print('can_same = {}, time = {}, success = {}'
                  .format(can_same, time, escaped/no_sims))