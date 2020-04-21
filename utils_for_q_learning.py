import numpy
import os

def action_checker(env):
	print("action range:",env.action_space.low,env.action_space.high)
	for l,h in zip(env.action_space.low,env.action_space.high):
		if l!=-h:
			print("asymetric action space")
			print("don't know how to deal with it")
			assert False
	if numpy.max(env.action_space.low)!=numpy.min(env.action_space.low):
		print("different action range per dimension")
		assert False
	if numpy.max(env.action_space.high)!=numpy.min(env.action_space.high):
		print("different action range per dimension")
		assert False


def get_hyper_parameters(name,alg):
	meta_params={}
	with open(alg+"_hyper_parameters/"+name+".hyper") as f:
		lines = [line.rstrip('\n') for line in f]
		for l in lines:
			parameter_name,parameter_value,parameter_type=(l.split(','))
			if parameter_type=='string':
				meta_params[parameter_name]=str(parameter_value)
			elif parameter_type=='integer':
				meta_params[parameter_name]=int(parameter_value)
			elif parameter_type=='float':
				meta_params[parameter_name]=float(parameter_value)
			else:
				print("unknown parameter type ... aborting")
				print(l)
				sys.exit(1)
	return meta_params

def sync_networks(target, online, alpha, copy = False):
	if copy == True:
		for online_param, target_param in zip(online.parameters(),
		                                      target.parameters()):
		    target_param.data.copy_(online_param.data)
	elif copy == False:
		for online_param, target_param in zip(online.parameters(),
		                                      target.parameters()):
		    target_param.data.copy_(alpha * online_param.data +
		                            (1 - alpha) * target_param.data)

def save(li_returns,params,alg):
	directory=alg+"_results/"+params['hyper_parameters_name']+'/'
	if not os.path.exists(directory):
	    os.makedirs(directory)
	numpy.savetxt(directory+str(params['seed_number'])+".txt",li_returns)

def set_random_seed(meta_params):
	seed_number=meta_params['seed_number']
	import numpy
	numpy.random.seed(seed_number)
	import random
	random.seed(seed_number)
	import torch
	torch.manual_seed(seed_number)
	meta_params['env'].seed(seed_number)