# Works but not as well as without the ppo addons

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


class PPO:
	"""
		This is the PPO class we will use as our model in main.py
	"""
	def __init__(self, policy_class, env, hidden_dims=(10,), opt_direction=1, logger=None, **hyperparameters):
		"""
			Initializes the PPO model, including hyperparameters.

			Parameters:
				policy_class - the policy class to use for our actor/critic networks.
				env - the environment to train on.
				hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

			Returns:
				None
		"""
		# Make sure the environment is compatible with our code
		assert isinstance(env.observation_space, gym.spaces.Box)
		# assert isinstance(env.action_space, gym.spaces.Box)

		# Initialize hyperparameters for training with PPO
		self._init_hyperparameters(hyperparameters)

		# Extract environment information
		self.env = env
		self.opt_direction = opt_direction
		self.obs_dim = env.observation_space.shape[0]
		self.act_dim = env.action_space.n

		 # Initialize actor and critic networks
		self.actor = policy_class(self.obs_dim, self.act_dim, hidden_dims, end_activation_fn=torch.nn.Softmax)                                                   # ALG STEP 1
		self.critic = policy_class(self.obs_dim, 1, hidden_dims)

		# Initialize optimizers for actor and critic
		self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

		self.logger = logger

	def learn(self, total_timesteps):
		"""
			Train the actor and critic networks. Here is where the main PPO algorithm resides.

			Parameters:
				total_timesteps - the total number of timesteps to train for

			Return:
				None
		"""
		print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
		print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
		t_so_far = 0 # Timesteps simulated so far
		i_so_far = 0 # Iterations ran so far
		while t_so_far < total_timesteps:                                                                       # ALG STEP 2
			# Autobots, roll out (just kidding, we're collecting our batch simulations here)
			batch_obs, batch_acts, batch_log_probs, batch_rtgs, A_k, batch_lens = self.rollout()                     # ALG STEP 3

			# Calculate how many timesteps we collected this batch
			t_so_far += np.sum(batch_lens)

			# Increment the number of iterations
			i_so_far += 1

			# One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
			# isn't theoretically necessary, but in practice it decreases the variance of 
			# our advantages and makes convergence much more stable and faster. I added this because
			# solving some environments was too unstable without it.
			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

			# This is the loop where we update our network for some n epochs
			for i in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
				# Calculate V_phi and pi_theta(a_t | s_t)
				V = self.critic(batch_obs).squeeze()
				probs = self.actor(batch_obs)
				dist = torch.distributions.Categorical(probs)
				curr_log_probs = dist.log_prob(batch_acts) # curr_log_probs will be the same as batch_log_probs for the first time

				# NOTE: Kinda crazy but because of a bug using softmax for V it was always 1.0, so basically no critic, but it still worked (for cartpole)

				# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
				# NOTE: we just subtract the logs, which is the same as
				# dividing the values and then canceling the log with e^log.
				# For why we use log probabilities instead of actual probabilities,
				# here's a great explanation: 
				# https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
				# TL;DR makes gradient ascent easier behind the scenes.
				ratios = torch.exp(curr_log_probs - batch_log_probs)

				# Calculate surrogate losses.
				surr1 = ratios * A_k
				surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

				# Calculate actor and critic losses.
				# NOTE: we take the negative min of the surrogate losses because we're trying to maximize
				# the performance function, but Adam minimizes the loss. So minimizing the negative
				# performance function maximizes it.
				entropy_loss = dist.entropy().mean()
				critic_loss = torch.nn.MSELoss()(V, batch_rtgs)
				actor_loss = (-torch.min(surr1, surr2)).mean() - self.entropy_weight * entropy_loss

				# Calculate gradients and perform backward propagation for actor network
				self.actor_optim.zero_grad()
				actor_loss.backward(retain_graph=True)
				torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
				self.actor_optim.step()

				# Calculate gradients and perform backward propagation for critic network
				self.critic_optim.zero_grad()
				critic_loss.backward()
				torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
				self.critic_optim.step()

				print(f"Actor Loss: {actor_loss.item()}\tCritic Loss: {critic_loss.item()}")

				# Approximate the KL divergence between the old policy and new policy.
				# Using ((ratios-1) - torch.log(ratios)).mean()
				approx_kl = ((ratios-1) - torch.log(ratios)).mean().detach().numpy()
				if approx_kl > self.target_kl:
					print(f"Early stopping at step {i} due to reaching max KL divergence: {approx_kl}")
					break
			
			# Save our model if it's time
			if i_so_far % self.save_freq == 0:
				torch.save(self.actor.state_dict(), './ppo_actor.pth')
				torch.save(self.critic.state_dict(), './ppo_critic.pth')

			best = np.max(batch_lens)
			if self.opt_direction == -1:
				best = np.min(batch_lens)

			if self.logger:
				results_dict = {
					"epoch": float(i_so_far),
					"timesteps": float(t_so_far),
					"best": float(best),
					"mean": float(np.mean(batch_lens)),
					"std": float(np.std(batch_lens)),
				}
				self.logger.save(results_dict)

	def rollout(self):
		"""
			Too many transformers references, I'm sorry. This is where we collect the batch of data
			from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
			of data each time we iterate the actor/critic networks.

			Parameters:
				None

			Return:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
		"""
		# Batch data. For more details, check function header.
		batch_obs = []
		batch_acts = []
		batch_log_probs = []
		batch_rtgs = []
		batch_adv = []
		batch_lens = []

		t = 0 # Keeps track of how many timesteps we've run so far this batch

		# Keep simulating until we've run more than or equal to specified timesteps per batch
		while t < self.timesteps_per_batch:
			ep_rews = [] # rewards collected per episode

			# Reset the environment. sNote that obs is short for observation. 
			obs = self.env.reset()[0]
			done = False

			# Run an episode for a maximum of max_timesteps_per_episode timesteps
			for ep_t in range(self.max_timesteps_per_episode):
				t += 1 # Increment timesteps ran this batch so far

				# Track observations in this batch
				batch_obs.append(obs)

				# Calculate action and make a step in the env. 
				# Note that rew is short for reward.
				action, log_prob = self.get_action(torch.from_numpy(obs).float())
				obs, rew, done, _, _ = self.env.step(action)

				# Track recent reward, action, and action log probability
				ep_rews.append(rew)
				batch_acts.append(action.item())
				batch_log_probs.append(log_prob)

				# If the environment tells us the episode is terminated, break
				if done:
					break

			# Track episodic lengths and rewards
			batch_lens.append(ep_t + 1)
			batch_rtgs.extend(self.compute_rtgs(ep_rews))
			ep_vals = self.critic(torch.tensor(np.array(batch_obs), dtype=torch.float)).detach().numpy().squeeze()
			batch_adv.extend(self.compute_gae(ep_rews, ep_vals))

		# Reshape data as tensors in the shape specified in function description, before returning
		batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
		batch_acts = torch.tensor(batch_acts, dtype=torch.float)
		batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
		# ALG STEP 4
		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
		batch_adv = torch.tensor(batch_adv, dtype=torch.float)

		return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_adv, batch_lens
	
	def compute_gae(self, ep_rews, ep_vals):
		"""
			Compute the Generalized Advantage Estimation of the rewards.

			Parameters:
				ep_rews - the rewards in a episode, Shape: (number of timesteps in episode)
				ep_vals - the value function predictions for each obs in episode, Shape: (number of timesteps in episode)

			Return:
				ep_adv - the advantage estimation for each timestep in the episode, Shape: (number of timesteps in episode)
		"""
		ep_adv = []
		last_adv = 0

		# Iterate backwards through episode backwards
		for i in reversed(range(len(ep_rews))):
			if i+1 == len(ep_rews):
				# If we're at the last timestep in the episode, there is no bootstrap
				# value, so use the true reward
				delta = ep_rews[i] - ep_vals[i]
			else:
				# Otherwise, bootstrap the value of the current timestep plus the
				# discounted value of the next timestep
				delta = ep_rews[i] + self.gamma * ep_vals[i+1] - ep_vals[i]

			# Calculate the advantage
			adv = delta + self.gamma * self.lam * last_adv

			# Prepend the advantage to the list of advantages (we need it in backwards order)
			ep_adv.insert(0, adv)

			# Update the last_advantage
			last_adv = adv

		return ep_adv

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

	def get_action(self, obs):
		"""
			Queries an action from the actor network, should be called from rollout.

			Parameters:
				obs - the observation at the current timestep

			Return:
				action - the action to take, as a numpy array
				log_prob - the log probability of the selected action in the distribution
		"""
		# Get probabilities for each action
		probs = self.actor(obs).detach()

		# Create a distribution with the probabilities
		dist = torch.distributions.Categorical(probs)

		# Sample an action from the distribution
		action = dist.sample()

		# Calculate the log probability for that action
		log_prob = dist.log_prob(action)

		# Return the sampled action and the log probability of that action in our distribution
		return action.detach().numpy(), log_prob.detach()

	def _init_hyperparameters(self, hyperparameters):
		"""
			Initialize default and custom values for hyperparameters

			Parameters:
				hyperparameters - the extra arguments included when creating the PPO model, should only include
									hyperparameters defined below with custom values.

			Return:
				None
		"""
		# Initialize default values for hyperparameters
		# Algorithm hyperparameters
		self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
		self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
		self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
		self.lr = 0.005                                 # Learning rate of actor optimizer
		self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
		self.lam = 0.95                                 # Discount factor to be applied when calculating Generalized Advantage Estimation
		self.entropy_weight = 0.1                       # Weight of the entropy bonus
		self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA
		self.target_kl = 0.01                           # KL divergence limit for stopping criterion

		self.max_grad_norm = 0.5                        # Max gradient norm to be applied to update networks

		# Miscellaneous parameters
		self.render = True                              # If we should render during rollout
		self.render_every_i = 10                        # Only render every n iterations
		self.save_freq = 10                             # How often we save in number of iterations
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
		experiment_name = 'ppo',
		epochs = 100,
		learning_rate = 0.005,
		gamma = 0.95,
		lam = 0.95,
		entropy_weight = 0.1,
		clip = 0.2,
		target_kl = 0.01,
		max_grad_norm = 0.5,
		updates_per_epoch = 5,
		timesteps_per_batch = 4800,
):
	"""
	PPO main function."""
	kwarg_dict = locals()

	results_dir = RESULTS_DIR + "_ppo"

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
	env, opt_direction = gym.make('CartPole-v1'), 1
	env, opt_direction = gym.make('Acrobot-v1'), -1
	env, opt_direction = gym.make('MountainCar-v0'), -1

	ppo = PPO(MLP, env, hidden_dims=(n_hidden_neurons,), opt_direction=opt_direction, logger=Logger(results_dir, use_folder, kwarg_dict),
			  **{
		'timesteps_per_batch': timesteps_per_batch,
		'max_timesteps_per_episode': 2500,
		'n_updates_per_iteration': updates_per_epoch,
		'lr': learning_rate,
		'gamma': gamma,
		'lam': lam,
		'entropy_weight': entropy_weight,
		'clip': clip,
		'target_kl': target_kl,
		'max_grad_norm': max_grad_norm,
	})
	
	ppo.learn(total_timesteps=epochs * timesteps_per_batch)


class Logger:
	def __init__(self, results_dir, use_folder, kwarg_dict):
		self.results_dir = results_dir
		self.use_folder = use_folder
		self.kwarg_dict = kwarg_dict

	def save(self, results_dict):
		save_results(self.use_folder, results_dict, self.kwarg_dict, self.results_dir)



def save_results(use_folder, results_dict, kwarg_dict={}, results_dir=RESULTS_DIR+'_ppo'):
	"""Save results to csv and print them."""
	print(f'\n EPOCH {results_dict["epoch"]}')
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
	df['epoch'] = df['epoch'].astype(int)

	# Save to csv
	df.to_csv(f'{results_dir}/{use_folder}/results.csv', index=False)

	# Save plot of results
	plt.figure(figsize=(10, 5))
	# Mean in dashed
	plt.plot(df['epoch'], df['mean'], label='mean', linestyle='--')
	plt.fill_between(df['epoch'], df['mean']-df['std'], df['mean']+df['std'], alpha=0.2)
	plt.plot(df['epoch'], df['best'], label='best')
	plt.xlabel('Epoch')
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
		"epochs":               100,
		"learning_rate":        0.005,
		"gamma":                0.95,
		"lam":                  0.99,
		"entropy_weight":       0.1,
		"clip":                 0.2,
		"target_kl":            0.001,
		"max_grad_norm":        0.5,
		"updates_per_epoch":    50,
		"timesteps_per_batch":  4800,
	}

	config["experiment_name"] = f'{config["epochs"]}_ppo'

	# Track time
	start_time = time.time()
	ppo_main(**config)
	# Print time in minutes and seconds
	print(f'\nTotal runtime: {round((time.time() - start_time) / 60, 2)} minutes')
	print(f'Total runtime: {round((time.time() - start_time), 2)} seconds')
