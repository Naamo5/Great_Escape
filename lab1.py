import random
import time
import numpy as np

class Game():
    """docstring for ClassName"""
    def __init__(self, player, enemy):
        self.player = player
        self.enemy = enemy

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
        time.sleep(0.3)

class Player():
    """docstring for ClassName"""
    def __init__(self):
        self.pos = [0,0]
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

Gary = Player()
Minotaur = Enemy()


# TEST TO SEE RESULT OF ACTIONS
for y in range(5):
    for x in range(6):
        for action in Gary.actdict.keys():
            yn, xn = Gary.transition([y, x], action)
            print('({},{}) > ({},{}) via {}'.format(y,x,yn,xn,action))

# TEST TO SEE STATE TRANSITIONS
for y in range(5):
    for x in range(6):
        for yn in range(5):
            for xn in range(6):
                prob = Minotaur.transition([y,x],[yn,xn])
                print('p{{({},{})|({},{})}} = {}'.format(yn,xn,y,x,prob))                