###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import pandas as pd
import time
import json
import torch
from matplotlib import pyplot as plt

import gym
from helpers import find_folder, get_random_str

RESULTS_DIR = 'results_cartpole'

# imports other libs
import numpy as np
import os


class EVO:
	"""
		This is the Evolution class we will use as our model in main.py
	"""
	def __init__(self, policy_class, env, hidden_dims=(10,), logger=None, **hyperparameters):
		"""
			Initializes the Evolution model, including hyperparameters.

			Parameters:
				policy_class - the policy class to use for our actor networks.
				env - the environment to train on.
				hyperparameters - all extra arguments passed into the evolution that should be hyperparameters.

			Returns:
				None
		"""
		# Make sure the environment is compatible with our code
		# assert isinstance(env.observation_space, gym.spaces.Box)
		# assert isinstance(env.action_space, gym.spaces.Box)

		# Initialize hyperparameters for training with PPO
		self._init_hyperparameters(hyperparameters)

		# Extract environment information
		self.env = env
		self.obs_dim = env.observation_space.shape[0]
		self.act_dim = env.action_space.n
		self.hidden_dims = hidden_dims

		# Initialize actors
		self.policy_class = policy_class
		self.population = [policy_class(self.obs_dim, self.act_dim, hidden_dims, end_activation_fn=torch.nn.Softmax) for _ in range(self.pop_size)]
		self.pop_fitness, self.pop_lens = self.evaluate(self.population)

		self.logger = logger

	def evolve(self, generations):
		for gen in range(generations):
			# Select parents
			parent_indices = self.select_fitness_proportionate(self.pop_fitness)
			
			# Copy children from selected parents
			children = [self.policy_class(self.obs_dim, self.act_dim, hidden_dims=self.hidden_dims, end_activation_fn=torch.nn.Softmax) for _ in range(self.pop_size)]
			for i, child in enumerate(children):
				child.load_state_dict(self.population[parent_indices[i]].state_dict())

			# Mutate children
			children = self.mutate(children)

			# Evaluate children
			child_fitness, child_lens = self.evaluate(children)

			# Select survivors
			combined_pop = self.population + children
			combined_fitness = self.pop_fitness + child_fitness
			combines_lens = self.pop_lens + child_lens
			survivor_indices = self.select_fitness_proportionate(combined_fitness)
			self.population = [combined_pop[i] for i in survivor_indices]
			self.pop_fitness = [combined_fitness[i] for i in survivor_indices]
			self.pop_lens = [combines_lens[i] for i in survivor_indices]
			
			# Save our model
			torch.save(self.population[0].state_dict(), './actor.pth')

			if self.logger:
				results_dict = {
					"gen": float(gen),
					"best": float(self.pop_lens[0]),
					"mean": float(np.mean(self.pop_lens)),
					"std": float(np.std(self.pop_lens)),
				}
				self.logger.save(results_dict)

	def evaluate(self, population):
		# Evaluate population
		pop_fitness = []
		pop_lens = []
		for individual in population:
			fitness, length = self.eval_individual(individual)
			pop_fitness.append(fitness)
			pop_lens.append(length)
		return pop_fitness, pop_lens

	def eval_individual(self, individual):
		# Run an episode for a maximum of max_timesteps_per_episode timesteps
		ep_rews = []
		obs = self.env.reset()[0]
		for ep_t in range(self.max_timesteps_per_episode):
			# Calculate action and make a step in the env.
			# Note that rew is short for reward.
			action = self.get_action(individual, torch.from_numpy(obs).float())
			obs, rew, done, _, _ = self.env.step(action)

			# Track recent reward, action
			ep_rews.append(rew)

			# If the environment tells us the episode is terminated, break
			if done:
				break

		# Track episodic lengths and rewards
		ep_rtgs = self.compute_rtgs(ep_rews)
		fitness = ep_rtgs[0]
		return fitness, len(ep_rtgs)
	
	def select_fitness_proportionate(self, pop_fitness):
		# Normalize fitness
		fitness_squared = np.square(pop_fitness)
		fitness_sum = np.sum(fitness_squared)
		fitness_norm = fitness_squared / fitness_sum

		# Make distribution
		dist = torch.distributions.Categorical(torch.from_numpy(fitness_norm).float())

		indices = dist.sample((self.pop_size,))
		
		# Elitism
		best = np.argmax(pop_fitness)
		indices[0] = best

		return indices
	
	def mutate(self, population):
		for individual in population:
			state_dict = individual.state_dict()
			mutated_state_dict = {}
			for key in state_dict.keys():
				params = state_dict[key]

				# Mutate
				mask = torch.rand_like(params) < self.mutation_rate
				mutated_params = params + mask * torch.randn_like(params) * self.mutation_std

				mutated_state_dict[key] = mutated_params
			individual.load_state_dict(mutated_state_dict)
		return population

	def compute_rtgs(self, ep_rews):
		"""
			Compute the Reward-To-Go of each timestep in a episode given the rewards.

			Parameters:
				ep_rews - the rewards in a episode, Shape: (number of timesteps in episode)

			Return:
				ep_rtgs - the rewards to go, Shape: (number of timesteps in episode)
		"""
		ep_rtgs = []
		discounted_reward = 0 # The discounted reward so far

		# Iterate through all rewards in the episode. We go backwards for smoother calculation of each
		# discounted return (think about why it would be harder starting from the beginning)
		for rew in reversed(ep_rews):
			discounted_reward = rew + discounted_reward * self.gamma
			ep_rtgs.insert(0, discounted_reward)
		return ep_rtgs

	def get_action(self, individual, obs):
		"""
			Queries an action from the actor network

			Parameters:
				individual - the actor network to query
				obs - the observation at the current timestep

			Return:
				action - the action to take, as a numpy array
		"""
		# Get probabilities for each action
		probs = individual(obs).detach()

		# Sample an action from the distribution
		action = torch.distributions.Categorical(probs).sample()

		# Return the sampled action and the log probability of that action in our distribution
		return action.detach().numpy()

	def _init_hyperparameters(self, hyperparameters):
		"""
			Initialize default and custom values for hyperparameters

			Parameters:
				hyperparameters - the extra arguments included when creating the model, should only include
									hyperparameters defined below with custom values.

			Return:
				None
		"""
		# Initialize default values for hyperparameters
		# Algorithm hyperparameters
		self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
		self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the params
		self.pop_size = 100                             # Population size, must be >= 1
		self.max_timesteps_per_episode = 1000           # Max number of timesteps per episode
		self.mutation_rate = 0.2                        # Mutation rate for the population
		self.mutation_std = 0.1                         # Mutation standard deviation

		# Miscellaneous parameters
		self.seed = None                                # Sets the seed of our program, used for reproducibility of results

		# Change any default values to custom values for specified hyperparameters
		for param, val in hyperparameters.items():
			exec('self.' + param + ' = ' + str(val))

		# Sets the seed if specified
		if self.seed != None:
			# Check if our seed is valid first
			assert(type(self.seed) == int)

			# Set the seed 
			torch.manual_seed(self.seed)
			print(f"Successfully set seed to {self.seed}")

		
class MLP(torch.nn.Module):
	"""
		A simple multilayer perceptron (feedforward neural network) policy.
	"""
	def __init__(self, input_dim, output_dim, hidden_dims=(10,), activation_fn=torch.nn.Sigmoid, end_activation_fn=torch.nn.Identity):
		"""
			Initializes the policy network.

			Parameters:
				input_dim - the dimension of the inputs
				output_dim - the number of possible actions to output
				hidden_dims - the dimensions of the hidden layers
				activation_fc - the activation function to use in between layers
		"""
		super(MLP, self).__init__()

		# Hidden layers
		self.hidden_layers = torch.nn.ModuleList()
		prev_dim = input_dim
		for h_dim in hidden_dims:
			self.hidden_layers.append(torch.nn.Linear(prev_dim, h_dim))
			self.hidden_layers.append(activation_fn())
			prev_dim = h_dim

		# Output layer
		self.output_layer = torch.nn.Sequential(
			torch.nn.Linear(prev_dim, output_dim),
			end_activation_fn(dim=-1)
		)

	def forward(self, x):
		"""
			Performs a forward pass through the network.

			Parameters:
				x - the input to the network

			Return:
				The action output by the network
		"""
		# Pass the input through the hidden layers
		for layer in self.hidden_layers:
			x = layer(x)

		# Pass the output through the output layer and return
		return self.output_layer(x)


def ppo_main(
		n_hidden_neurons = 10,
		start_new = True,
		experiment_name = 'evo',
		gens = 100,
		clip = 0.2,
		pop_size = 100,
		mutation_rate = 0.2,
		mutation_std = 0.1,
):
	kwarg_dict = locals()

	results_dir = RESULTS_DIR + "_evo"

	# Make results dir if it doesn't exist
	os.makedirs(results_dir, exist_ok=True)

	# Find folder
	if start_new:
		use_folder = None
		start_epoch = 0
	else:
		use_folder, start_epoch = find_folder(kwarg_dict)

	if not use_folder:
		milliseconds = int(round(time.time() * 1000))
		use_folder = f'{milliseconds}_{experiment_name}'
		if start_new:
			# In case we're doing parallel runs, we don't want to overwrite the folder
			# Add random hash to folder name
			use_folder += f'_{get_random_str()}'

		os.makedirs(f'{results_dir}/{use_folder}')

	# env = gym.make('LunarLander-v2')
	# env = gym.make('CartPole-v1')
	env = gym.make('Acrobot-v1')
	# env = gym.make('MountainCar-v0')

	evo = EVO(MLP, env, hidden_dims=(n_hidden_neurons,), logger=Logger(results_dir, use_folder, kwarg_dict),
			  **{
				  "pop_size": pop_size,
				  "mutation_rate": mutation_rate,
				  "mutation_std": mutation_std,
				'clip': clip,
	})
	
	evo.evolve(gens)


class Logger:
	def __init__(self, results_dir, use_folder, kwarg_dict):
		self.results_dir = results_dir
		self.use_folder = use_folder
		self.kwarg_dict = kwarg_dict

	def save(self, results_dict):
		save_results(self.use_folder, results_dict, self.kwarg_dict, self.results_dir)



def save_results(use_folder, results_dict, kwarg_dict={}, results_dir=RESULTS_DIR+'_evo'):
	"""Save results to csv and print them."""
	print(f'\n Generation {results_dict["gen"]}')
	print(f'  fitness:  best: {round(results_dict["best"],6)} mean: {round(results_dict["mean"],6)} std: {round(results_dict["std"],6)}')

	# Save results using pandas
	# Load csv if it exists
	if os.path.exists(f'{results_dir}/{use_folder}/results.csv'):
		df = pd.read_csv(f'{results_dir}/{use_folder}/results.csv')
	else:
		# Make dirs
		os.makedirs(f'{results_dir}/{use_folder}', exist_ok=True)
		df = pd.DataFrame(columns=results_dict.keys())

	# Concat new row
	new_row = pd.DataFrame([results_dict.values()], columns=results_dict.keys())
	df = pd.concat([df, new_row], ignore_index=True)
	df['gen'] = df['gen'].astype(int)

	# Save to csv
	df.to_csv(f'{results_dir}/{use_folder}/results.csv', index=False)

	# Save plot of results
	plt.figure(figsize=(10, 5))
	# Mean in dashed
	plt.plot(df['gen'], df['mean'], label='mean', linestyle='--')
	plt.fill_between(df['gen'], df['mean']-df['std'], df['mean']+df['std'], alpha=0.2)
	plt.plot(df['gen'], df['best'], label='best')
	plt.xlabel('Generation')
	plt.ylabel('Fitness')
	plt.savefig(f'{results_dir}/{use_folder}/results.png')
	plt.close()

	# Save json with all kwargs plus gen, best, mean, std
	kwarg_dict.update(results_dict)
	with open(f'{results_dir}/{use_folder}/config.json', 'w') as f:
		json.dump(kwarg_dict, f, indent=4)



if __name__ == '__main__':
	config = {
		"n_hidden_neurons":     10,
		"start_new":            True,
		"experiment_name":      'evo',
		"gens":                 100,
		"clip":                 0.2,
		"pop_size":             100,
		"mutation_rate":        0.2,
		"mutation_std":         0.1,
	}

	config["experiment_name"] = f'{config["gens"]}_evo'

	# Track time
	start_time = time.time()
	ppo_main(**config)
	# Print time in minutes and seconds
	print(f'\nTotal runtime: {round((time.time() - start_time) / 60, 2)} minutes')
	print(f'Total runtime: {round((time.time() - start_time), 2)} seconds')
