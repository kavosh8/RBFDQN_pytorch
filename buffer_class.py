import numpy
from collections import deque

class buffer_class:
	def __init__(self,max_length):
		self.storage=deque(maxlen=max_length)

	def append(self,s,a,r,done,sp):
		dic={}
		dic['s']=s
		dic['a']=a
		dic['r']=r
		if done==True:
			dic['done']=1
		else:
			dic['done']=0
		dic['sp']=sp
		self.storage.append(dic)
