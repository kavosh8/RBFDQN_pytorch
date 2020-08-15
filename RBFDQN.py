import gym, sys
import numpy, random
import utils_for_q_learning, buffer_class

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
import pickle
from collections import deque

def rbf_function_single_batch_mode(centroid_locations, beta, N, norm_smoothing):
	'''
		no batch
		given N centroids * size of each centroid
		determine weight of each centroid at each other centroid
	'''
	centroid_locations = [c.unsqueeze(1) for c in centroid_locations]
	centroid_locations_cat = torch.cat(centroid_locations, dim=1)
	centroid_locations_cat = centroid_locations_cat.unsqueeze(2)
	centroid_locations_cat = torch.cat([centroid_locations_cat for _ in range(N)],dim=2)
	centroid_locations_cat_transpose = centroid_locations_cat.permute(0,2,1,3)
	diff      = centroid_locations_cat - centroid_locations_cat_transpose
	diff_norm = diff**2
	diff_norm = torch.sum(diff_norm, dim=3)
	diff_norm = diff_norm + norm_smoothing
	diff_norm = torch.sqrt(diff_norm)
	diff_norm = diff_norm * beta * -1
	weights   = F.softmax(diff_norm, dim=2)
	return weights

def rbf_function(centroid_locations, action, beta, N, norm_smoothing):
	'''
		given batch size * N centroids * size of each centroid
		and batch size * size of each action, determine the weight of
		each centroid for the action 
	'''
	centroid_locations_squeezed = [l.unsqueeze(1) for l in centroid_locations]
	centroid_locations_cat = torch.cat(centroid_locations_squeezed, dim=1)
	action_unsqueezed = action.unsqueeze(1)
	action_cat = torch.cat([action_unsqueezed for _ in range(N)], dim=1)
	diff = centroid_locations_cat - action_cat
	diff_norm = diff**2
	diff_norm = torch.sum(diff_norm, dim=2)
	diff_norm = diff_norm + norm_smoothing
	diff_norm = torch.sqrt(diff_norm)
	diff_norm = diff_norm * beta * -1
	output    = F.softmax(diff_norm, dim=1)
	return output 

class Net(nn.Module):
	def __init__(self, params, env, state_size, action_size):
		super(Net, self).__init__()
		
		self.env = env
		self.params = params
		self.N = self.params['num_points']
		self.max_a = self.env.action_space.high[0]
		self.beta = self.params['temperature']

		self.buffer_object = buffer_class.buffer_class(max_length = self.params['max_buffer_size'],
													   seed_number = self.params['seed_number'])

		self.state_size, self.action_size = state_size, action_size

		self.value_side1 = nn.Linear(self.state_size, self.params['layer_size'])
		self.value_side1_parameters = self.value_side1.parameters()

		self.value_side2 = nn.Linear(self.params['layer_size'], self.params['layer_size'])
		self.value_side2_parameters = self.value_side2.parameters()

		self.value_side3 = nn.Linear(self.params['layer_size'], self.params['layer_size'])
		self.value_side3_parameters = self.value_side3.parameters()

		self.value_side4 = nn.Linear(self.params['layer_size'], self.N)
		self.value_side4_parameters = self.value_side4.parameters()

		self.drop = nn.Dropout(p=self.params['dropout_rate'])

		self.location_side1 = nn.Linear(self.state_size, self.params['layer_size'])
		torch.nn.init.xavier_uniform_(self.location_side1.weight)
		torch.nn.init.zeros_(self.location_side1.bias)

		self.location_side2 = []
		for _ in range(self.N):
			temp = nn.Linear(self.params['layer_size'], self.action_size)
			temp.weight.data.uniform_(-.1, .1)
			temp.bias.data.uniform_(-1, +1)
			#nn.init.uniform_(temp.bias,a = -2.0, b = +2.0)
			self.location_side2.append(temp)
		self.location_side2 = torch.nn.ModuleList(self.location_side2)
		self.criterion = nn.MSELoss()


		self.params_dic=[]
		self.params_dic.append({'params': self.value_side1_parameters, 'lr': self.params['learning_rate']})
		self.params_dic.append({'params': self.value_side2_parameters, 'lr': self.params['learning_rate']})
		
		self.params_dic.append({'params': self.value_side3_parameters, 'lr': self.params['learning_rate']})
		self.params_dic.append({'params': self.value_side4_parameters, 'lr': self.params['learning_rate']})
		self.params_dic.append({'params': self.location_side1.parameters(), 'lr': self.params['learning_rate_location_side']})

		for i in range(self.N):
		    self.params_dic.append({'params': self.location_side2[i].parameters(), 'lr': self.params['learning_rate_location_side']}) 
		self.optimizer = optim.RMSprop(self.params_dic)

	def forward(self, s, a):
		'''
			given a batch of s,a compute Q(s,a)
		'''
		centroid_values = self.get_centroid_values(s)
		centroid_locations = self.get_all_centroids_batch_mode(s)
		centroid_weights = rbf_function(centroid_locations, 
										a, 
										self.beta, 
										self.N, 
										self.params['norm_smoothing'])
		output = torch.mul(centroid_weights,centroid_values)
		output = output.sum(1,keepdim=True)
		return output

	def get_centroid_values(self, s):
		'''
			given a batch of s, get V(s)_i for i in 1 through N
		'''
		temp = F.relu(self.value_side1(s))
		temp = F.relu(self.value_side2(temp))
		temp = F.relu(self.value_side3(temp))
		centroid_values = self.value_side4(temp)
		return centroid_values

	def get_all_centroids_batch_mode(self, s):
		temp = F.relu(self.location_side1(s))
		temp = self.drop(temp)
		temp = [self.location_side2[i](temp).unsqueeze(0) for i in range(self.N)]
		temp = torch.cat(temp,dim=0)
		temp = self.max_a*torch.tanh(temp)
		centroid_locations = list(torch.split(temp, split_size_or_sections=1, dim=0))
		centroid_locations = [c.squeeze(0) for c in centroid_locations]
		return centroid_locations

	def get_best_centroid_batch(self, s):
		'''
			given a batch of states s
			determine max_{a} Q(s,a)
		'''
		all_centroids = self.get_all_centroids_batch_mode(s)
		values = self.get_centroid_values(s).unsqueeze(2)
		weights = rbf_function_single_batch_mode(all_centroids, 
												 self.beta, 
												 self.N, 
												 self.params['norm_smoothing'])
		allq = torch.bmm(weights, values).squeeze(2)
		best,indices = allq.max(1)
		if s.shape[0] == 1: #the function is called for a single state s
			index_star = indices.data.numpy()[0]
			a = list(all_centroids[index_star].data.numpy()[0])
			return best.data.numpy(), a
		else: #batch mode, for update
			return best.data.numpy()


	def e_greedy_policy(self,s,episode,train_or_test):
		epsilon=1./numpy.power(episode,1./self.params['policy_parameter'])

		if train_or_test=='train' and random.random() < epsilon:
			a = self.env.action_space.sample()
			return a.tolist()
		else:
			self.eval()
			s_matrix = numpy.array(s).reshape(1,self.state_size)
			q,a = self.get_best_centroid_batch( torch.FloatTensor(s_matrix))
			self.train()
			return a	

	def update(self, target_Q, count):

		if len(self.buffer_object.storage)<self.params['batch_size']:
			return 0
		else:
			pass
		s_matrix, a_matrix, r_matrix, done_matrix, sp_matrix = self.buffer_object.sample(self.params['batch_size'])
		r_matrix=numpy.clip(r_matrix,a_min=-self.params['reward_clip'],a_max=self.params['reward_clip'])

		Q_star = target_Q.get_best_centroid_batch(torch.FloatTensor(sp_matrix))
		Q_star = Q_star.reshape((self.params['batch_size'],-1))
		y=r_matrix+self.params['gamma']*(1-done_matrix)*Q_star
		y_hat = self.forward(torch.FloatTensor(s_matrix),torch.FloatTensor(a_matrix))
		loss = self.criterion(y_hat,torch.FloatTensor(y).detach())
		self.zero_grad()
		loss.backward()
		self.optimizer.step()
		self.zero_grad()
		utils_for_q_learning.sync_networks(target = target_Q,
										   online = self, 
										   alpha = self.params['target_network_learning_rate'], 
										   copy = False)
		return loss.data.numpy()

if __name__=='__main__':
	hyper_parameter_name=sys.argv[1]
	alg='rbf'
	params=utils_for_q_learning.get_hyper_parameters(hyper_parameter_name,alg)
	params['hyper_parameters_name']=hyper_parameter_name
	env=gym.make(params['env_name'])
	#env = gym.wrappers.Monitor(env, 'videos/'+params['env_name']+"/", video_callable=lambda episode_id: episode_id%10==0,force = True)
	params['env']=env
	params['seed_number']=int(sys.argv[2])
	utils_for_q_learning.set_random_seed(params)
	s0=env.reset()
	utils_for_q_learning.action_checker(env)
	Q_object = Net(params,env,state_size=len(s0),action_size=len(env.action_space.low))
	Q_object_target = Net(params,env,state_size=len(s0),action_size=len(env.action_space.low))
	Q_object_target.eval()

	utils_for_q_learning.sync_networks(target = Q_object_target, online = Q_object, alpha = params['target_network_learning_rate'], copy = True)

	G_li=[]
	loss_li = []
	for episode in range(params['max_episode']):

		Q_this_episode = Net(params,env,state_size=len(s0),action_size=len(env.action_space.low))
		utils_for_q_learning.sync_networks(target = Q_this_episode, online = Q_object, alpha = params['target_network_learning_rate'], copy = True)
		Q_this_episode.eval()

		s,done,t=env.reset(),False,0
		while done==False:
			a=Q_object.e_greedy_policy(s,episode+1,'train')
			sp,r,done,_=env.step(numpy.array(a))
			t=t+1
			done_p = False if t == env._max_episode_steps else done
			Q_object.buffer_object.append(s,a,r,done_p,sp)
			s=sp
		#now update the Q network
		loss = []
		for count in range(params['updates_per_episode']):
			temp = Q_object.update(Q_object_target, count)
			loss.append(temp)
		loss_li.append(numpy.mean(loss))


		if (episode % 10 == 0) or (episode == params['max_episode'] - 1):
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
