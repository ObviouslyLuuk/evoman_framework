# the demo_controller file contains standard controller structures for the agents.
# you can overwrite the method 'control' in your own instance of the environment
# and then use a different type of controller if you wish.
# note that the param 'controller' received by 'control' is provided through environment.play(pcont=x)
# 'controller' could contain either weights to be used in the standard controller (or other controller implemented),
# or even a full network structure (ex.: from NEAT).
from evoman.controller import Controller
import numpy as np
import torch


def sigmoid_activation(x):
	return 1./(1.+np.exp(-x))


class player_controller(Controller):
	def __init__(self, _n_hidden, net):
		self.env = None
		self.n_hidden = [_n_hidden]
		self.states = []
		self.actions = []
		self.logprobs = []
		self.rewards = []
		self.deterministic = False

		self.net = net
	
	def get_reward(self):
		# return self.env.player.life*0.1 - self.env.enemy.life*0.9
		return self.env.player.life - self.env.enemy.life
	
	def get_probs(self, inputs):
		inputs = np.array(inputs)
		mini = np.min(inputs, axis=-1, keepdims=True)
		rang = np.max(inputs, axis=-1, keepdims=True) - mini
		inputs = (inputs-mini)/rang # shape (20,) or (N, 20)
		probs = self.net(torch.from_numpy(inputs).float()) # shape (5,) or (N, 5)
		return probs
	
	def get_logprobs(self, output, probs):
		return torch.where(output == 1, torch.log(probs), torch.log(1-probs))
	
	# def get_action(self, inputs, deterministic=False):
	#   # Pick any number of actions
	# 	probs = self.get_probs(inputs).detach()
	# 	if deterministic:
	# 		output = torch.where(probs > 0.5, 1, 0)
	# 	else:
	# 		randoms = torch.rand_like(probs)
	# 		output = torch.where(randoms < probs, 1, 0)
	# 	logprobs = self.get_logprobs(output, probs)
	# 	return output, logprobs
	
	def get_action(self, inputs, deterministic=False):
		# Only pick one action
		probs = self.get_probs(inputs).detach()
		dist = torch.distributions.Categorical(probs)
		if deterministic:
			output = torch.zeros_like(probs)
			output[dist.probs.argmax()] = 1
		else:
			output = torch.zeros_like(probs)
			output[dist.sample()] = 1
		logprobs = self.get_logprobs(output, probs)
		return output, logprobs

	def control(self, inputs, controller):
		self.net = controller
		output, logprobs = self.get_action(inputs, self.deterministic)

		self.states.append(inputs)
		self.actions.append(output)
		# self.logprobs.append(logprobs)
		self.logprobs.append(logprobs[torch.argmax(output)])
		self.rewards.append(self.get_reward())
		return output
	
	def reset(self):
		self.states = []
		self.actions = []
		self.logprobs = []
		self.rewards = []



# implements controller structure for enemy
class enemy_controller(Controller):
	def __init__(self, _n_hidden):
		# Number of hidden neurons
		self.n_hidden = [_n_hidden]

	def control(self, inputs,controller):
		# Normalises the input using min-max scaling
		inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))

		if self.n_hidden[0]>0:
			# Preparing the weights and biases from the controller of layer 1

			# Biases for the n hidden neurons
			bias1 = controller[:self.n_hidden[0]].reshape(1,self.n_hidden[0])
			# Weights for the connections from the inputs to the hidden nodes
			weights1_slice = len(inputs)*self.n_hidden[0] + self.n_hidden[0]
			weights1 = controller[self.n_hidden[0]:weights1_slice].reshape((len(inputs),self.n_hidden[0]))

			# Outputs activation first layer.
			output1 = sigmoid_activation(inputs.dot(weights1) + bias1)

			# Preparing the weights and biases from the controller of layer 2
			bias2 = controller[weights1_slice:weights1_slice + 5].reshape(1,5)
			weights2 = controller[weights1_slice + 5:].reshape((self.n_hidden[0],5))

			# Outputting activated second layer. Each entry in the output is an action
			output = sigmoid_activation(output1.dot(weights2)+ bias2)[0]
		else:
			bias = controller[:5].reshape(1, 5)
			weights = controller[5:].reshape((len(inputs), 5))

			output = sigmoid_activation(inputs.dot(weights) + bias)[0]

		# takes decisions about sprite actions
		if output[0] > 0.5:
			attack1 = 1
		else:
			attack1 = 0

		if output[1] > 0.5:
			attack2 = 1
		else:
			attack2 = 0

		if output[2] > 0.5:
			attack3 = 1
		else:
			attack3 = 0

		if output[3] > 0.5:
			attack4 = 1
		else:
			attack4 = 0

		return [attack1, attack2, attack3, attack4]
