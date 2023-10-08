# Does not work

# imports framework
import pandas as pd
import time
import json
import torch

from evoman.environment import Environment
from rl_controller import player_controller
from helpers import find_folder, get_random_str, RESULTS_DIR

# imports other libs
import numpy as np
import os


def visualize_test(enemies, pcont):
	"""Visualize a test run."""
	multi = "no"
	if len(enemies) > 1:
		multi = "yes"
	
	# Remove SDL_VIDEODRIVER from environment variables to show visuals
	if "SDL_VIDEODRIVER" in os.environ:
		os.environ.pop("SDL_VIDEODRIVER")

	# Make temp directory
	directory = f'{RESULTS_DIR}_ppo/temp'
	os.makedirs(directory, exist_ok=True)

	# initializes simulation in individual evolution mode, for single static enemy.
	env = Environment(experiment_name=directory,
					multiplemode=multi,
					enemies=enemies,
					playermode="ai",
					player_controller=player_controller(10, pcont), # you  can insert your own controller here
					enemymode="static",
					level=2,
					speed="fastest",
					visuals=True)
	env.player_controller.env = env
	env.player_controller.deterministic = True
	f,p,e,t = env.play(pcont=pcont)

	# print(pcont.state_dict())

	# Delete temp directory and contents
	os.system(f'rm -r {directory}')

	os.environ["SDL_VIDEODRIVER"] = "dummy"
	
	return f,p,e,t


class PPO:
	"""
		This is the PPO class we will use as our model in main.py
	"""
	def __init__(self, actor, critic, env, logger=None, **hyperparameters):
		"""
			Initializes the PPO model, including hyperparameters.

			Parameters:
				policy_class - the policy class to use for our actor/critic networks.
				env - the environment to train on.
				hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

			Returns:
				None
		"""
		# Initialize hyperparameters for training with PPO
		self._init_hyperparameters(hyperparameters)

		# Extract environment information
		self.env = env
		self.actor = actor
		self.critic = critic

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
			# Collecting our batch simulations
			batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, fitnesses = self.rollout()                     # ALG STEP 3

			# Calculate how many timesteps we collected this batch
			t_so_far += np.sum(batch_lens)

			# Increment the number of iterations
			i_so_far += 1

			# Calculate advantage at k-th iteration
			V = self.critic(batch_obs).squeeze()
			A_k = batch_rtgs - V.detach()                                                                       # ALG STEP 5

			# Normalizing advantages decreases the variance of our advantages and makes 
			# convergence much more stable and faster
			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
			A_k = A_k.unsqueeze(-1)
			# print(f'advantage: {A_k}')

			# This is the loop where we update our network for some n epochs
			for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
				# Calculate V_phi and pi_theta(a_t | s_t)
				# Query the critic network for values of each state
				V = self.critic(batch_obs).squeeze()
				# Calculate the log probabilities of batch actions using most recent actor network.
				curr_log_probs = self.env.player_controller.get_logprobs(batch_acts, self.env.player_controller.get_probs(batch_obs))
				# Index the log probs with the actions taken to get the log probs of taken actions. batch_acts is one-hot.
				curr_log_probs = torch.sum(curr_log_probs * batch_acts, dim=1)

				# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
				ratios = torch.exp(curr_log_probs - batch_log_probs)

				# Calculate surrogate losses.
				surr1 = ratios * A_k
				surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

				# Calculate actor and critic losses. (Adam minimizes the loss but we're working with reward, hence minus)
				actor_loss = (-torch.min(surr1, surr2)).mean()
				critic_loss = torch.nn.MSELoss()(V, batch_rtgs)

				# Calculate gradients and perform backward propagation for actor network
				self.actor_optim.zero_grad()
				actor_loss.backward(retain_graph=True)
				self.actor_optim.step()

				# Calculate gradients and perform backward propagation for critic network
				self.critic_optim.zero_grad()
				critic_loss.backward()
				self.critic_optim.step()

				print(f'actor loss: {actor_loss}, critic loss: {critic_loss}')

			f,p,e,t = visualize_test(self.env.enemies, self.actor)
			self.env.player_controller.reset()

			print(f'fitness: {f}')
			
			# Save our model if it's time
			if i_so_far % self.save_freq == 0:
				torch.save(self.actor.state_dict(), './ppo_actor.pth')
				torch.save(self.critic.state_dict(), './ppo_critic.pth')

			if self.logger:
				results_dict = {
					"epoch": float(i_so_far),
					"timesteps": float(t_so_far),
					"best": float(np.max(fitnesses)),
					"mean": float(np.mean(fitnesses)),
					"std": float(np.std(fitnesses)),
					"fitness": f,
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
		batch_lens = []
		fitnesses = []

		cumulative_t = 0 # Keeps track of how many timesteps we've run so far this batch

		# Keep simulating until we've run more than or equal to specified timesteps per batch
		while cumulative_t < self.timesteps_per_batch:
			f,p,e,t = self.env.play(pcont=self.actor) # Pointless to run this more than once, since the policy is deterministic
			batch_obs.extend(self.env.player_controller.states)
			batch_acts.extend(self.env.player_controller.actions)
			batch_log_probs.extend(self.env.player_controller.logprobs)
			fitnesses.append(f)
			cumulative_rewards = [*self.env.player_controller.rewards, self.env.player_controller.get_reward()]

			# Get rewards by subtracting the previous cumulative reward from the current cumulative reward
			ep_rews = []
			for i in range(len(cumulative_rewards)-1):
				ep_rews.append(cumulative_rewards[i+1] - cumulative_rewards[i])

			# Track episodic lengths and rewards
			batch_lens.append(len(ep_rews))
			ep_rtgs = self.compute_rtgs(ep_rews)
			batch_rtgs.extend(ep_rtgs)

			cumulative_t += t
			self.env.player_controller.reset()

		# Reshape data as tensors in the shape specified in function description, before returning
		batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
		batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
		batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float)
		batch_rtgs = torch.tensor(np.array(batch_rtgs), dtype=torch.float)
		fitnesses = np.array(fitnesses)

		return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, fitnesses

	def compute_rtgs(self, ep_rews):
		"""
			Compute the Reward-To-Go of each timestep in an episode given the rewards.

			Parameters:
				ep_rews - the rewards in an episode, Shape: (number of timesteps in episode)

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
		self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA

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
		enemies = [2],
		n_hidden_neurons = 10,
		start_new = True,
		experiment_name = 'ppo',
		epochs = 100,
		learning_rate = 0.005,
		gamma = 0.95,
		clip = 0.2,
		updates_per_epoch = 5,
		timesteps_per_batch = 4800,
		headless = True,
):
	"""
	PPO main function."""
	kwarg_dict = locals()

	results_dir = RESULTS_DIR + "_ppo"
	
	# choose this for not using visuals and thus making experiments faster
	visuals = True
	if headless:
		os.environ["SDL_VIDEODRIVER"] = "dummy"
		visuals = False

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

	multi = "no"
	if len(enemies) > 1:
		multi = "yes"

	obs_dim = 20
	act_dim = 5

	# Initialize actor and critic networks
	actor = MLP(obs_dim, act_dim, (n_hidden_neurons,), end_activation_fn=torch.nn.Softmax)                                                   # ALG STEP 1
	critic = MLP(obs_dim, 1, (n_hidden_neurons,))

	# initializes simulation in individual evolution mode, for single static enemy.
	env = Environment(experiment_name=f'{results_dir}/{use_folder}',
					multiplemode=multi,
					enemies=enemies,
					playermode="ai",
					player_controller=player_controller(n_hidden_neurons, actor), # you  can insert your own controller here
					enemymode="static",
					level=2,
					speed="fastest",
					visuals=visuals,
					randomini="no")
	env.player_controller.env = env

	ppo = PPO(actor, critic, env, logger=Logger(results_dir, use_folder, kwarg_dict),
			  hyperparameters={
		'timesteps_per_batch': timesteps_per_batch,
		'max_timesteps_per_episode': 2500,
		'n_updates_per_iteration': updates_per_epoch,
		'lr': learning_rate,
		'gamma': gamma,
		'clip': clip,
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

	# Save json with all kwargs plus gen, best, mean, std
	kwarg_dict.update(results_dict)
	with open(f'{results_dir}/{use_folder}/config.json', 'w') as f:
		json.dump(kwarg_dict, f, indent=4)



if __name__ == '__main__':
	# Visualize test
	visualize_test([1], MLP(20, 5, (10,), end_activation_fn=torch.nn.Softmax))


	config = {
		"enemies":              [1],	
		"n_hidden_neurons":     10,
		"epochs":               100,
		"learning_rate":        0.5,
		"gamma":                0.99,
	}

	config["experiment_name"] = f'{config["epochs"]}_ppo'

	# Track time
	start_time = time.time()
	ppo_main(**config)
	# Print time in minutes and seconds
	print(f'\nTotal runtime: {round((time.time() - start_time) / 60, 2)} minutes')
	print(f'Total runtime: {round((time.time() - start_time), 2)} seconds')
