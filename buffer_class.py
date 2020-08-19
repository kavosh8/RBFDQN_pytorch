import numpy
from collections import deque
import random


class buffer_class:
	def __init__(self, max_length, seed_number):
		self.storage = deque(maxlen=max_length)

	def append(self, s, a, r, done, sp):
		dic = {}
		dic['s'] = s
		dic['a'] = a
		dic['r'] = r
		if done == True:
			dic['done'] = 1
		else:
			dic['done'] = 0
		dic['sp'] = sp
		self.storage.append(dic)

	def sample(self, batch_size):
		batch = random.sample(self.storage, batch_size)
		s_li = [b['s'] for b in batch]
		sp_li = [b['sp'] for b in batch]
		r_li = [b['r'] for b in batch]
		done_li = [b['done'] for b in batch]
		a_li = [b['a'] for b in batch]
		s_matrix = numpy.array(s_li).reshape(batch_size, -1)
		a_matrix = numpy.array(a_li).reshape(batch_size, -1)
		r_matrix = numpy.array(r_li).reshape(batch_size, 1)
		sp_matrix = numpy.array(sp_li).reshape(batch_size, -1)
		done_matrix = numpy.array(done_li).reshape(batch_size, 1)
		return s_matrix, a_matrix, r_matrix, done_matrix, sp_matrix

	def __len__(self):
		return len(self.storage)
