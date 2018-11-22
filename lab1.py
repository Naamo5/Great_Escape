import random
import time
import numpy as np

class Board():
	"""docstring for ClassName"""
	def __init__(self):
		pass
	def display(self, PlayerPos, MinotaurPos):
		VisBoard = np.empty((5,6), dtype='str')
		VisBoard[:] = ' '
		if PlayerPos == MinotaurPos:
			VisBoard[tuple(PlayerPos)] = 'B' # Both in same position
		else:
			VisBoard[tuple(PlayerPos)] = 'P' # Player
			VisBoard[tuple(MinotaurPos)] = 'M' # Minotaur
		print(' ' + '_'*8 + ' ')
		print('|{}{}|{}{} {}{}|'.format(*VisBoard[0,:]))
		print('|{}{}|{}{}|{}{}|'.format(*VisBoard[1,:]))
		print('|{}{}|{}{}|{}\u0305{}\u0305|'.format(*VisBoard[2,:]))
		print('|{}{} {}{} {}{}|'.format(*VisBoard[3,:]))
		print('|{}{}\u203e{}\u0305{}\u0305|\u0305{}\u0305{}\u0305|'.format(
			*VisBoard[4,:]))
		print(' ' + '\u203e'*8 + ' ')
		time.sleep(0.3)

class Player():
	"""docstring for ClassName"""
	def __init__(self):
		self.Pos = [0,0]

class Minotaur():
	"""docstring for ClassName"""
	def __init__(self, CanSame=False):
		self.Pos = [4,4]
		self.CanSame = CanSame # Can it stay in the same place
		self.Actions = ['U','D','L','R'] # Available actions
		self.Action = '' # Current action
		if self.CanSame:
			self.Actions.append('S')

	def move(self):
		self.Action = self.Actions[random.randint(0,len(self.Actions)-1)]
		if self.Pos[1] == 0 and self.Action == 'L':
			pass
		elif self.Pos[1] == 5 and self.Action == 'R':
			pass
		elif self.Pos[0] == 0 and self.Action == 'U':
			pass
		elif self.Pos[0] == 4 and self.Action == 'D':
			pass
		else:
			if self.Action == 'L':
				self.Pos[1] = self.Pos[1] - 1
			elif self.Action == 'R':
				self.Pos[1] = self.Pos[1] + 1
			elif self.Action == 'U':
				self.Pos[0] = self.Pos[0] - 1
			elif self.Action == 'D':
				self.Pos[0] = self.Pos[0] + 1
			else:
				pass

Boardy = Board()
Playa = Player()
Minny = Minotaur(True)
Boardy.display(Playa.Pos, Minny.Pos)
for i in range(100):
	Minny.move()
	print(Minny.Action)
	Boardy.display(Playa.Pos, Minny.Pos)