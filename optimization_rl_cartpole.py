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

import gym
from helpers import find_folder, get_random_str

RESULTS_DIR = 'results_cartpole'

# imports other libs
import numpy as np
import os


class MLP(torch.nn.Module):
    """
        A simple multilayer perceptron (feedforward neural network) policy.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=(10,), activation_fc=torch.nn.Sigmoid):
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
            self.hidden_layers.append(activation_fc())
            prev_dim = h_dim

        # Output layer
        self.output_layer = torch.nn.Sequential(
			torch.nn.Linear(prev_dim, output_dim),
			torch.nn.Softmax(dim=-1)
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


def run_test(actor, results_dir, use_folder):
    """Run test and save video"""
    # Make video dir
    os.makedirs(f'{results_dir}/{use_folder}/videos', exist_ok=True)
    env = gym.make('CartPole-v1', render_mode='rgb_array')

    state = env.reset()[0]
    done = False
    length = 0
    while not done:
        # Predict action probabilities
        probs_ = actor(torch.from_numpy(state).float()).detach().numpy()

        # Sample action
        action = np.random.choice(np.arange(len(probs_)), p=probs_)

        # Take action
        next_state, reward, done, _, info = env.step(action)

        # Update state
        state = next_state
        length += 1

        # Save video

    env.close()

    return length
    

def ppo_main(
        n_hidden_neurons = 10,
        start_new = True,
        experiment_name = 'ppo',
        epochs = 100,
        learning_rate = 0.005,
        gamma = 0.95,
        clip = 0.2,
        updates_per_epoch = 5,
        timesteps_per_batch = 4800,
):
    """
    PPO main function.
    Step 1: Collecting Data and Computing Rewards 
    The first step in PPO is to collect data by running the current policy in the environment. This data consists of state-action pairs and the corresponding rewards obtained from the environment. By interacting with the environment, the agent explores different actions and observes the resulting states and rewards.

    Step 2: Computing Advantage Estimates
    Once the data is collected, the next step involves estimating the advantages for each state-action pair. Advantages represent the expected improvement of taking a specific action in a given state compared to the average action value. It quantifies how much better an action is than the average action in a particular state.

    Step 3: Updating the Policy
    PPO aims to improve the policy by updating it in a way that maximizes the expected rewards. The policy update is performed by maximizing an objective function that balances exploration and exploitation. PPO achieves this by using a surrogate objective function that approximates the ratio of the new policy to the old policy.

    Step 4: Clipping the Surrogate Objective
    To ensure stable and reliable updates, PPO introduces a clipping mechanism. The surrogate objective is clipped to a specified range, preventing drastic policy changes that could lead to instability. Clipping bounds the policy update, restraining it within an acceptable range.

    Step 5: Iterative Optimization
    PPO performs multiple iterations of steps 1 to 4 to progressively improve the policy. By collecting new data, computing advantages, updating the policy, and applying the clipping mechanism, the algorithm iteratively converges towards an optimal policy
    """
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


    # initializes simulation in individual evolution mode, for single static enemy.
    env = gym.make('CartPole-v1')

    n_inputs = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Initialize policy parameters
    actor = MLP(n_inputs, n_actions, hidden_dims=(n_hidden_neurons,))
    critic = MLP(n_inputs, 1, hidden_dims=(n_hidden_neurons,))

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate)

    # For each epoch
    for epoch in range(start_epoch, epochs):
        # Collect data
        states, actions, log_probs, discounted_rewards, f = collect_data(env, actor, n_steps=timesteps_per_batch, gamma=gamma)

        # Predict state values
        V = critic(states.float()).squeeze()

        # Compute advantages
        advantages = discounted_rewards - V.detach()
        advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-10)
        advantages = advantages[:, None]

        for _ in range(updates_per_epoch):
            # Predict action probabilities
            V = critic(states.float()).squeeze()
            new_probs = actor(states.float())
            dist = torch.distributions.Categorical(probs=new_probs)
            new_log_probs = dist.log_prob(actions)

            ratios = torch.exp(new_log_probs - log_probs) # shape (n_states, 5)

            # Compute surrogate objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-clip, 1+clip) * advantages

            # Compute loss
            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = torch.nn.MSELoss()(V.squeeze(), discounted_rewards.float())

            # Backpropagation
            actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            actor_optimizer.step()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

        # Save results
        results_dict = {
            'epoch': epoch,
            'best': f.max().item(),
            'mean': f.float().mean().item(),
            'std': f.float().std().item(),
        }
        save_results(use_folder, results_dict, kwarg_dict, results_dir=results_dir)

        # Save best solution
        torch.save(actor.state_dict(), f'{results_dir}/{use_folder}/best_actor.pt')


def collect_data(env, actor, n_steps=4800, gamma=0.95):
    """Collects data by running the current policy in the environment. This data consists of state-action pairs and the corresponding rewards obtained from the environment. By interacting with the environment, the agent explores different actions and observes the resulting states and rewards."""
    states = []
    actions = []
    logprobs = []
    discounted_rewards = []
    fitnesses = []

    cumulative_steps = 0
    while cumulative_steps < n_steps:
        state = env.reset()[0] # reset returns tuple (state, ?)
        done = False
        rewards = []
        length = 0
        while not done:
            # Predict action probabilities
            probs_ = actor(torch.from_numpy(state).float()).detach()

            # Make probability distribution
            dist = torch.distributions.Categorical(probs=probs_)

            # Sample action
            action = dist.sample()
            log_prob = dist.log_prob(action).item()
            action = action.item()

            # Take action
            next_state, reward, done, _, info = env.step(action)

            # Save data
            states.append(state)
            actions.append(action)
            logprobs.append(log_prob)
            rewards.append(reward)

            # Update state
            state = next_state

            length += 1

        fitnesses.append(length)
        cumulative_steps += length

        # Calculate discounted rewards
        discounted_rewards.extend(calc_discounted_rewards(torch.tensor(rewards), gamma=gamma))

    return torch.tensor(np.array(states)), torch.tensor(actions), torch.tensor(logprobs), torch.tensor(discounted_rewards), torch.tensor(fitnesses)

# def calc_discounted_rewards(rewards, gamma=0.95):
#     """Calculates the discounted rewards for each state-action pair. The discount factor gamma determines the importance of future rewards. 
#     A high gamma value indicates that the agent is more concerned about the distant future, whereas a low gamma value indicates that the agent is more concerned about the immediate reward."""
#     discounted_rewards = torch.zeros_like(rewards)
#     for i in reversed(range(len(rewards))):
#         if i == len(rewards)-1: # Last reward
#             discounted_rewards[i] = rewards[i]
#         else:
#             discounted_rewards[i] = rewards[i] + gamma * discounted_rewards[i+1]

#     return discounted_rewards

def calc_discounted_rewards(ep_rews, gamma=0.95):
    # The shape will be (num timesteps per episode)
    batch_rtgs = []

    discounted_reward = 0 # The discounted reward so far

    # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
    # discounted return (think about why it would be harder starting from the beginning)
    for rew in reversed(ep_rews):
        discounted_reward = rew + discounted_reward * gamma
        batch_rtgs.insert(0, discounted_reward)

    # Convert the rewards-to-go into a tensor
    batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

    return batch_rtgs



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
    config = {
        "n_hidden_neurons":     10,
        "epochs":               100,
    }

    config["experiment_name"] = f'{config["epochs"]}_ppo'

    # Track time
    start_time = time.time()
    ppo_main(**config)
    # Print time in minutes and seconds
    print(f'\nTotal runtime: {round((time.time() - start_time) / 60, 2)} minutes')
    print(f'Total runtime: {round((time.time() - start_time), 2)} seconds')
