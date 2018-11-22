import random
import time
import numpy as np

class Board():
	"""docstring for ClassName"""
	def __init__(self):
		pass
	def display(self, player_pos, minotaur_pos):
		vis_board = np.empty((5,6), dtype='str')
		vis_board[:] = ' '
		if player_pos == minotaur_pos:
			vis_board[tuple(player_pos)] = 'B' # Both in same position
		else:
			vis_board[tuple(player_pos)] = 'P' # Player
			vis_board[tuple(minotaur_pos)] = 'M' # Minotaur
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

class Minotaur():
	"""docstring for ClassName"""
	def __init__(self, can_same=False):
		self.pos = [4,4]
		self.can_same = can_same # Can it stay in the same place
		self.pij = self.get_prob(can_same)

	def get_prob(self, can_same):
		pij = np.zeros((30,30))
		for y in range(5):
			for x in range(6):
				# CENTRE POSITIONS
				if y > 0 and y < 4 and x > 0 and x < 5:
					pij[self.p2s([y,x]),self.p2s([y+1,x])] = 1.0/(4 + can_same)
					pij[self.p2s([y,x]),self.p2s([y-1,x])] = 1.0/(4 + can_same)
					pij[self.p2s([y,x]),self.p2s([y,x+1])] = 1.0/(4 + can_same)
					pij[self.p2s([y,x]),self.p2s([y,x-1])] = 1.0/(4 + can_same)
					pij[self.p2s([y,x]),self.p2s([y,x])] = 1 - (4.0/(4 + can_same))
				# EDGE POSITIONS
				elif x == 0 and y > 0 and y < 4:
					pij[self.p2s([y,x]),self.p2s([y,x])] = (1.0 + can_same)/(4 + can_same)
					pij[self.p2s([y,x]),self.p2s([y+1,x])] = 1.0/(4 + can_same)
					pij[self.p2s([y,x]),self.p2s([y-1,x])] = 1.0/(4 + can_same)
					pij[self.p2s([y,x]),self.p2s([y,x+1])] = 1.0/(4 + can_same)
				elif x == 5 and y > 0 and y < 4:
					pij[self.p2s([y,x]),self.p2s([y,x])] = (1.0 + can_same)/(4 + can_same)
					pij[self.p2s([y,x]),self.p2s([y+1,x])] = 1.0/(4 + can_same)
					pij[self.p2s([y,x]),self.p2s([y-1,x])] = 1.0/(4 + can_same)
					pij[self.p2s([y,x]),self.p2s([y,x-1])] = 1.0/(4 + can_same)
				elif y == 0 and x > 0 and x < 5:
					pij[self.p2s([y,x]),self.p2s([y,x])] = (1.0 + can_same)/(4 + can_same)
					pij[self.p2s([y,x]),self.p2s([y+1,x])] = 1.0/(4 + can_same)
					pij[self.p2s([y,x]),self.p2s([y,x-1])] = 1.0/(4 + can_same)
					pij[self.p2s([y,x]),self.p2s([y,x+1])] = 1.0/(4 + can_same)
				elif y == 4 and x > 0 and x < 5:
					pij[self.p2s([y,x]),self.p2s([y,x])] = (1.0 + can_same)/(4 + can_same)
					pij[self.p2s([y,x]),self.p2s([y-1,x])] = 1.0/(4 + can_same)
					pij[self.p2s([y,x]),self.p2s([y,x-1])] = 1.0/(4 + can_same)
					pij[self.p2s([y,x]),self.p2s([y,x+1])] = 1.0/(4 + can_same)
				# CORNER POSITIONS
				elif y==0 and x==0:
					pij[self.p2s([y,x]),self.p2s([y,x])] = (2.0 + can_same)/(4 + can_same)
					pij[self.p2s([y,x]),self.p2s([y+1,x])] = 1.0/(4 + can_same)
					pij[self.p2s([y,x]),self.p2s([y,x+1])] = 1.0/(4 + can_same)
				elif y==4 and x==0:
					pij[self.p2s([y,x]),self.p2s([y,x])] = (2.0 + can_same)/(4 + can_same)
					pij[self.p2s([y,x]),self.p2s([y-1,x])] = 1.0/(4 + can_same)
					pij[self.p2s([y,x]),self.p2s([y,x+1])] = 1.0/(4 + can_same)
				elif y==0 and x==5:
					pij[self.p2s([y,x]),self.p2s([y,x])] = (2.0 + can_same)/(4 + can_same)
					pij[self.p2s([y,x]),self.p2s([y+1,x])] = 1.0/(4 + can_same)
					pij[self.p2s([y,x]),self.p2s([y,x-1])] = 1.0/(4 + can_same)
				elif y==4 and x==5:
					pij[self.p2s([y,x]),self.p2s([y,x])] = (2.0 + can_same)/(4 + can_same)
					pij[self.p2s([y,x]),self.p2s([y-1,x])] = 1.0/(4 + can_same)
					pij[self.p2s([y,x]),self.p2s([y,x-1])] = 1.0/(4 + can_same)
		return(pij)

	def p2s(self, pos): # transform position coordinate ([0,4],[0,5]) to state [0,29]
		return(6*pos[0] + pos[1])

	def s2p(self, state):
		return([state//6,state%6])

	def move(self):
		state = np.zeros(30)
		state[self.p2s(self.pos)] = 1
		prob = np.matmul(state,self.pij)
		self.pos = self.s2p(np.random.choice(list(range(30)),p=prob))

Boardy = Board()
Playa = Player()
Minny = Minotaur(True)
Boardy.display(Playa.pos, Minny.pos)
for i in range(100):
	Minny.move()
	Boardy.display(Playa.pos, Minny.pos)