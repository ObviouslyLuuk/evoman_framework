# the demo_controller file contains standard controller structures for the agents.
# you can overwrite the method 'control' in your own instance of the environment
# and then use a different type of controller if you wish.
# note that the param 'controller' received by 'control' is provided through environment.play(pcont=x)
# 'controller' could contain either weights to be used in the standard controller (or other controller implemented),
# or even a full network structure (ex.: from NEAT).
from evoman.controller import Controller
import numpy as np


def sigmoid_activation(x):
	return 1./(1.+np.exp(-x))

def sort_inputs_by_dist(inputs, projectile_x_dist_idx, projectile_y_dist_idx):
	"""
	Sorts projectiles by euclidean distance to player.
	"""
	euclid_dists = np.sqrt(inputs[projectile_x_dist_idx]**2 + inputs[projectile_y_dist_idx]**2)
	euclid_dists[euclid_dists==0] = 1 # ignore non-existing projectiles

	# Sort projectile x and y indices by euclidean distance
	sorted_idx = np.argsort(euclid_dists)
	sorted_x_idx = np.array(projectile_x_dist_idx)[sorted_idx]
	sorted_y_idx = np.array(projectile_y_dist_idx)[sorted_idx]

	# Interleave x and y indices
	sorted_xy_idx = np.empty((sorted_x_idx.size + sorted_y_idx.size,), dtype=sorted_x_idx.dtype)
	sorted_xy_idx[0::2] = sorted_x_idx
	sorted_xy_idx[1::2] = sorted_y_idx

	inputs[4:] = inputs[sorted_xy_idx]

	# Print rounded euclidean distances
	euclid_dists = np.sqrt(inputs[projectile_x_dist_idx]**2 + inputs[projectile_y_dist_idx]**2)
	euclid_dists[euclid_dists==0] = 1 # ignore non-existing projectiles
	# print(np.round(euclid_dists, 2))

	# inputs[3] = 0 # ignore enemy direction
	# inputs[12:] = 0 # ignore 4 furthest projectiles

	return inputs


def normalize_inputs(inputs, method="default"):
	"""
	default normalizes inputs between 0 and 1.
	around_0 normalizes inputs between -1 and 1.
	custom normalizes inputs between -1 and 1 according to the following:
	The inputs are:
        - enemy x distance (-736 to 736)
        - enemy y distance (-512 to 512)
        - player direction (-1 to 1)
		- enemy direction  (-1 to 1)
		16 times:
            - projectile x distance (-736 to 736)
            - projectile y distance (-512 to 512)
	"""
	if method == "default": # default normalization between 0 and 1
		return (inputs-min(inputs))/float((max(inputs)-min(inputs)))
	elif method == "around_0":
		return inputs/float(max(abs(inputs)))
	elif method == "domain_specific":
		inputs = np.array(inputs)
		e_x_dist_idx = [0]
		e_y_dist_idx = [1]
		projectile_x_dist_idx = [4, 6, 8, 10, 12, 14, 16, 18]
		projectile_y_dist_idx = [5, 7, 9, 11, 13, 15, 17, 19]
		x_dist_idx = np.concatenate((e_x_dist_idx, projectile_x_dist_idx))
		y_dist_idx = np.concatenate((e_y_dist_idx, projectile_y_dist_idx))
		inputs[x_dist_idx] = inputs[x_dist_idx]/736
		inputs[y_dist_idx] = inputs[y_dist_idx]/512
		inputs[[2, 3]] = inputs[[2, 3]]*np.abs(inputs).mean() # put less weight on player and enemy direction

		# inputs = sort_inputs_by_dist(inputs, projectile_x_dist_idx, projectile_y_dist_idx)

		return inputs

# implements controller structure for player
class player_controller(Controller):
	def __init__(self, _n_hidden, normalization_method="default"):
		self.n_hidden = [_n_hidden]
		self.normalization_method = normalization_method

	def set(self,controller, n_inputs):
		# Number of hidden neurons

		if self.n_hidden[0] > 0:
			# Preparing the weights and biases from the controller of layer 1

			# Biases for the n hidden neurons
			self.bias1 = controller[:self.n_hidden[0]].reshape(1, self.n_hidden[0])
			# Weights for the connections from the inputs to the hidden nodes
			weights1_slice = n_inputs * self.n_hidden[0] + self.n_hidden[0]
			self.weights1 = controller[self.n_hidden[0]:weights1_slice].reshape((n_inputs, self.n_hidden[0]))

			# Outputs activation first layer.


			# Preparing the weights and biases from the controller of layer 2
			self.bias2 = controller[weights1_slice:weights1_slice + 5].reshape(1, 5)
			self.weights2 = controller[weights1_slice + 5:].reshape((self.n_hidden[0], 5))
		else:
			self.bias1 = controller[:5].reshape(1, 5)
			self.weights1 = controller[5:].reshape((n_inputs, 5))

	def control(self, inputs, controller):
		inputs = normalize_inputs(inputs, method=self.normalization_method)

		if self.n_hidden[0]>0:
			# Preparing the weights and biases from the controller of layer 1

			# Outputs activation first layer.
			output1 = sigmoid_activation(inputs.dot(self.weights1) + self.bias1)

			# Outputting activated second layer. Each entry in the output is an action
			output = sigmoid_activation(output1.dot(self.weights2)+ self.bias2)[0]
		else:
			bias = controller[:5].reshape(1, 5)
			weights = controller[5:].reshape((len(inputs), 5))

			output = sigmoid_activation(inputs.dot(weights) + bias)[0]

		# takes decisions about sprite actions
		if output[0] > 0.5:
			left = 1
		else:
			left = 0

		if output[1] > 0.5:
			right = 1
		else:
			right = 0

		if output[2] > 0.5:
			jump = 1
		else:
			jump = 0

		if output[3] > 0.5:
			shoot = 1
		else:
			shoot = 0

		if output[4] > 0.5:
			release = 1
		else:
			release = 0

		return [left, right, jump, shoot, release]


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
