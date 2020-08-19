import gym
import sys
import time
import numpy
import random
import utils_for_q_learning, buffer_class

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
import pickle
from collections import deque


LOGGING_LEVEL = 1

def _print(logging_level, stuff):
	if LOGGING_LEVEL >= logging_level:
		print(stuff)

def rbf_function(centroid_locations, action_set, beta):
	'''
	centroid_locations: Tensor [batch x num_centroids (N) x a_dim (action_size)]
	action_set: Tensor [batch x num_act x a_dim (action_size)]
		- Note: pass in num_act = 1 if you want a single action evaluated
	beta: float
		- Parameter for RBF function

	Description: Computes the RBF function given centroid_locations and some actions
	'''
	assert len(centroid_locations.shape) == 3, "Must pass tensor with shape: [batch x N x a_dim]"
	assert len(action_set.shape) == 3, "Must pass tensor with shape: [batch x num_act x a_dim]"

	diff_norm = torch.cdist(centroid_locations, action_set, p=2) # batch x N x num_act
	diff_norm = diff_norm * beta * -1
	weights   = F.softmax(diff_norm, dim=2) # batch x N x num_act
	return weights

class Net(nn.Module):
	def __init__(self, params, env, state_size, action_size, device):
		super(Net, self).__init__()
		
		self.env = env
		self.device = device
		self.params = params
		self.N = self.params['num_points']
		self.max_a = self.env.action_space.high[0]
		self.beta = self.params['temperature']

		self.buffer_object = buffer_class.buffer_class(
				max_length = self.params['max_buffer_size'],
				seed_number = self.params['seed_number']
			)

		self.state_size, self.action_size = state_size, action_size

		self.value_module = nn.Sequential(
			nn.Linear(self.state_size, self.params['layer_size']),
			nn.ReLU(),
			nn.Linear(self.params['layer_size'], self.params['layer_size']),
			nn.ReLU(),
			nn.Linear(self.params['layer_size'], self.params['layer_size']),
			nn.ReLU(),
			nn.Linear(self.params['layer_size'], self.N),
		)

		self.location_module = nn.Sequential(
			nn.Linear(self.state_size, self.params['layer_size']),
			nn.Dropout(p=self.params['dropout_rate']),
			nn.ReLU(),
			nn.Linear(self.params['layer_size'], self.action_size*self.N),
			utils_for_q_learning.Reshape(-1, self.N, self.action_size),
			nn.Tanh(),
		)

		torch.nn.init.xavier_uniform_(self.location_module[0].weight)
		torch.nn.init.zeros_(self.location_module[0].bias)

		self.location_module[3].weight.data.uniform_(-.1, .1)
		self.location_module[3].bias.data.uniform_(-1., 1.)

		self.criterion = nn.MSELoss()

		self.params_dic = [
			{'params': self.value_module.parameters(), 'lr': self.params['learning_rate']},
			{'params': self.location_module.parameters(), 'lr': self.params['learning_rate_location_side']}
		]

		self.optimizer = optim.RMSprop(self.params_dic)

		self.to(self.device)

	def get_centroid_values(self, s):
		'''
			given a batch of s, get V(s)_i for i in 1 through N
		'''
		centroid_values = self.value_module(s)
		_print(2, "centroid values shape: {}".format(centroid_values.shape))
		return centroid_values

	def new_get_all_centroids_batch_mode(self, s):
		centroid_locations = self.max_a * self.location_module(s)
		return centroid_locations
	
	def new_get_best_centroid_batch(self, s):
		"""Doing a new one because I want to be able to directly compare the results."""
		'''
			given a batch of states s
			determine max_{a} Q(s,a)
		'''
		all_centroids = self.new_get_all_centroids_batch_mode(s)
		values = self.get_centroid_values(s)
		weights = rbf_function(all_centroids, all_centroids, self.beta) # [batch x N x N]
		allq = torch.bmm(weights, values.unsqueeze(2)).squeeze(2) # bs x num_centroids
		# a -> all_centroids[idx] such that idx is max(dim=1) in allq
		# a = torch.gather(all_centroids, dim=1, index=indices)
                # (dim: bs x 1, dim: bs x action_dim)
		best, indices = allq.max(dim=1)
		if s.shape[0] == 1:
			index_star = indices.item()
			a = all_centroids[0, index_star]
			return best, a
		else:
			return best, None 

	def forward(self, s, a):
		'''
			given a batch of s,a compute Q(s,a)
		'''
		centroid_values = self.get_centroid_values(s) # [batch_dim x N]
		centroid_locations = self.new_get_all_centroids_batch_mode(s)
		centroid_weights = rbf_function(centroid_locations, a.unsqueeze(dim=1), self.beta) # [batch x N x 1]
		output = torch.mul(centroid_weights.squeeze(-1), centroid_values) # [batch x N]
		output = output.sum(1,keepdim=True) # [batch x 1]
		_print(2, "Forward output shape: {}".format(output.shape))
		return output

	def e_greedy_policy(self,s,episode,train_or_test):
		epsilon=1.0/numpy.power(episode, 1.0/self.params['policy_parameter'])

		if train_or_test=='train' and random.random() < epsilon:
			a = self.env.action_space.sample()
			return a.tolist()
		else:
			self.eval()
			s_matrix = numpy.array(s).reshape(1,self.state_size)
			with torch.no_grad():
				_, a = self.new_get_best_centroid_batch(torch.FloatTensor(s_matrix).to(self.device))
				a = a.cpu().numpy()
			self.train()
			return a

	def update(self, target_Q, count):

		if len(self.buffer_object.storage) < self.params['batch_size']:
			return 0
		else:
			pass
		s_matrix, a_matrix, r_matrix, done_matrix, sp_matrix = self.buffer_object.sample(self.params['batch_size'])
		r_matrix=numpy.clip(r_matrix, a_min = -self.params['reward_clip'], a_max = self.params['reward_clip'])

		s_matrix = torch.FloatTensor(s_matrix).to(self.device)
		a_matrix = torch.FloatTensor(a_matrix).to(self.device)
		r_matrix = torch.FloatTensor(r_matrix).to(self.device)
		done_matrix = torch.FloatTensor(done_matrix).to(self.device)
		sp_matrix = torch.FloatTensor(sp_matrix).to(self.device)

		Q_star, _ = target_Q.new_get_best_centroid_batch(sp_matrix)
		Q_star = Q_star.reshape((self.params['batch_size'],-1))
		with torch.no_grad():
			y = r_matrix + self.params['gamma'] * (1 - done_matrix) * Q_star
		y_hat = self.forward(s_matrix,a_matrix)
		loss = self.criterion(y_hat, y)
		self.zero_grad()
		loss.backward()
		self.optimizer.step()
		self.zero_grad()
		utils_for_q_learning.sync_networks(target = target_Q,
				online = self,
				alpha = self.params['target_network_learning_rate'],
				copy = False)
		return loss.cpu().data.numpy()



if __name__=='__main__':
	if torch.cuda.is_available():
		device = torch.device("cuda")
		print("Running on the GPU")
	else:
		device = torch.device("cpu")
		print("Running on the CPU")
	hyper_parameter_name = sys.argv[1]
	alg='rbf'
	params=utils_for_q_learning.get_hyper_parameters(hyper_parameter_name,alg)
	params['hyper_parameters_name']=hyper_parameter_name
	env=gym.make(params['env_name'])
	#env = gym.wrappers.Monitor(env, 'videos/'+params['env_name']+"/", video_callable=lambda episode_id: episode_id%10==0,force = True)
	params['env']=env
	params['seed_number']=int(sys.argv[2])
	utils_for_q_learning.set_random_seed(params)
	s0 = env.reset()
	utils_for_q_learning.action_checker(env)
	Q_object = Net(params, env, state_size=len(s0), action_size=len(env.action_space.low), device=device)
	Q_object_target = Net(params, env, state_size=len(s0), action_size=len(env.action_space.low), device=device)
	Q_object_target.eval()

	utils_for_q_learning.sync_networks(target = Q_object_target, online = Q_object, alpha = params['target_network_learning_rate'], copy = True)

	import time

	G_li=[]
	loss_li = []
	all_times_per_steps = []
	all_times_per_updates = []
	for episode in range(params['max_episode']):

		Q_this_episode = Net(params,env,state_size=len(s0),action_size=len(env.action_space.low), device=device)
		utils_for_q_learning.sync_networks(target = Q_this_episode, online = Q_object, alpha = params['target_network_learning_rate'], copy = True)
		Q_this_episode.eval()

		s,done,t=env.reset(),False,0
		start = time.time()
		num_steps = 0
		while num_steps < 1000:
			num_steps += 1
			a=Q_object.e_greedy_policy(s,episode+1,'train')
			sp,r,done,_=env.step(numpy.array(a))
			t=t+1
			done_p = False if t == env._max_episode_steps else done
			Q_object.buffer_object.append(s,a,r,done_p,sp)
			s=sp
		end = time.time()
		time_per_step = (end - start) / num_steps
		all_times_per_steps.append(time_per_step)
		# print('num_steps:', num_steps)
		if num_steps != 0:
			_print(1, "e-greedy: {} per step".format(time_per_step))
			_print(1, "Average e-greedy after {} episodes: {}".format(episode, sum(all_times_per_steps) / len(all_times_per_steps)))
		else:
			_print(1, 'no steps...')
		#now update the Q network
		start = time.time()
		loss = []
		for count in range(params['updates_per_episode']):
			temp = Q_object.update(Q_object_target, count)
			loss.append(temp)
		end = time.time()
		time_per_update = (end - start) / params['updates_per_episode']
		all_times_per_updates.append(time_per_update)
		_print(1, "per update: {}".format(time_per_update))
		_print(1, "Average time-per-update after {} episodes: {}".format(episode, sum(all_times_per_updates) / len(all_times_per_updates)))

		loss_li.append(numpy.mean(loss))


		if False and (episode % 10 == 0) or (episode == params['max_episode'] - 1):
			temp = []
			for _ in range(10):
				s,G,done,t=env.reset(),0,False,0
				while done==False:
					a=Q_object.e_greedy_policy(s,episode+1,'test')
					sp,r,done,_=env.step(numpy.array(a))
					s,G,t=sp,G+r,t+1
				temp.append(G)
			print("after {} episodes, learned policy collects {} average returns".format(episode,numpy.mean(temp)))
			G_li.append(numpy.mean(temp))	
			utils_for_q_learning.save(G_li,loss_li,params,alg)
