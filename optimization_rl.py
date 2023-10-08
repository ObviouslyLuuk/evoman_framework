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

from evoman.environment import Environment
from rl_controller import player_controller
from helpers import find_folder, RESULTS_DIR, get_best, get_random_str

# imports other libs
import numpy as np
import os

def fitness_balanced(player_life, enemy_life, time):
    """Returns a balanced fitness, based on the player life, enemy life and time"""
    return .5*(100-enemy_life) + .5*player_life - np.log(time+1)

def run_test(config, based_on_eval_best=None, deterministic=True, results_dir=RESULTS_DIR+'_ppo'):
    """Run the best solution for the given config"""
    enemies = config['enemies']

    folder = get_best(config, based_on_eval_best=based_on_eval_best, results_dir=results_dir, use_key='best')
    with open(f'{results_dir}/{folder}/config.json', 'r') as f:
        config = json.load(f)
    n_hidden_neurons = config['n_hidden_neurons']

    print(config)

    best_solution = np.loadtxt(f'{results_dir}/{folder}/best.txt')

    print(f'\nRunning best solution for enemy {enemies}')
    print(f'Best folder: {folder}')
    print(f'Best fitness: {config["best"]}')

    env = Environment(experiment_name=f'{results_dir}/{folder}',
                    enemies=[enemies[0]],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="normal",
                    visuals=True,
                    randomini="no")
    env.player_controller.env = env
    env.player_controller.deterministic = deterministic

    def print_result(f, p, e, t):
        win_condition = e <= 0
        win_str = 'WON\n' if win_condition else 'LOST\n'
        print(f'Fitness: {f}, player life: {p}, enemy life: {e}, time: {t}')
        print(f' balanced fitness: {fitness_balanced(p, e, t)}')
        print(win_str)

    total_gain = 0
    for enemy in enemies:
        env.enemies = [enemy]
        print(f'Running enemy {enemy}')
        f,p,e,t = env.play(pcont=best_solution)
        print_result(f, p, e, t)
        total_gain += p-e
    print(f'Total gain: {total_gain}')


def ppo_main(
        enemies = [2],
        n_hidden_neurons = 10,
        headless = True,
        start_new = True,
        experiment_name = 'ppo',
        epochs = 100,
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

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=f'{results_dir}/{use_folder}',
                    multiplemode=multi,
                    enemies=enemies,
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=visuals,
                    randomini="no")
    env.player_controller.env = env

    # number of weights for multilayer with 10 hidden neurons
    if n_hidden_neurons > 0:
        n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
    else:
        n_vars = (env.get_num_sensors()+1)*5

    # Initialize policy parameters
    actor = np.random.uniform(-1, 1, size=(n_vars,))
    critic = torch.nn.Sequential(
        torch.nn.Linear(20, n_hidden_neurons),
        torch.nn.Sigmoid(),
        torch.nn.Linear(n_hidden_neurons, 1),
        torch.nn.Sigmoid()
    )

    actor_optimizer = torch.optim.Adam(env.player_controller.net.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    # For each epoch
    for epoch in range(start_epoch, epochs):
        # Collect data
        states, actions, probs, discounted_rewards, f = collect_data(env, actor)
        log_probs = torch.where(actions == 1, torch.log(probs), torch.log(1-probs))

        # Predict state values
        V = critic(states.float()).squeeze()

        # Compute advantages
        advantages = discounted_rewards - V.detach()
        advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-10)
        advantages = advantages[:, None]

        for _ in range(5):
            env.player_controller.set(actor, states.shape[-1])
            new_probs = env.player_controller.net(states.float()) # shape (n_states, 5)
            new_log_probs = torch.where(actions == 1, torch.log(new_probs), torch.log(1-new_probs)) # shape (n_states, 5)

            ratios = torch.exp(new_log_probs - log_probs) # shape (n_states, 5)

            # Compute surrogate objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-0.2, 1+0.2) * advantages

            # Compute loss
            actor_loss = -torch.mean(torch.min(surr1, surr2))
            critic_loss = torch.nn.MSELoss()(V, discounted_rewards.float())

            # Backpropagation
            actor_loss.backward()
            actor_optimizer.step()
            actor = env.player_controller.get()
            critic_loss.backward()
            critic_optimizer.step()

            V = critic(states.float())

        env.player_controller.deteministic = True
        f,p,e,t = env.play(pcont=actor)
        env.player_controller.deteministic = False

        # Save results
        results_dict = {
            'epoch': epoch,
            'best': f,
            'mean': f,
            'std': f,
        }
        save_results(use_folder, results_dict, kwarg_dict, results_dir=results_dir)

        # Save best solution
        np.savetxt(f'{results_dir}/{use_folder}/best.txt', actor)

        # Save environment
        env.update_solutions([actor, f.mean()])
        env.save_state()

    env.state_to_log() # checks environment state


def collect_data(env, policy_parameters, n_episodes=10):
    """Collects data by running the current policy in the environment. This data consists of state-action pairs and the corresponding rewards obtained from the environment. By interacting with the environment, the agent explores different actions and observes the resulting states and rewards."""
    states = []
    actions = []
    probs = []
    discounted_rewards = []
    fitnesses = []

    for _ in range(n_episodes):
        f,p,e,t = env.play(pcont=policy_parameters) # Pointless to run this more than once, since the policy is deterministic
        fitnesses.append(f)
        states.extend(env.player_controller.states)
        actions.extend(env.player_controller.actions)
        probs.extend(env.player_controller.probs)
        cumulative_rewards = [*env.player_controller.rewards, env.player_controller.get_reward()]
        env.player_controller.reset()

        # Get rewards by subtracting the previous cumulative reward from the current cumulative reward
        rewards = torch.zeros(len(cumulative_rewards)-1)
        for i in range(len(rewards)):
            rewards[i] = cumulative_rewards[i+1] - cumulative_rewards[i]

        discounted_rewards.extend(calc_discounted_rewards(rewards))

    return torch.tensor(states), torch.tensor(actions), torch.tensor(probs), torch.tensor(discounted_rewards), np.array(fitnesses)

def calc_discounted_rewards(rewards, gamma=0.95):
    """Calculates the discounted rewards for each state-action pair. The discount factor gamma determines the importance of future rewards. 
    A high gamma value indicates that the agent is more concerned about the distant future, whereas a low gamma value indicates that the agent is more concerned about the immediate reward."""
    discounted_rewards = torch.zeros_like(rewards)
    for i in reversed(range(len(rewards))):
        if i == len(rewards)-1: # Last reward
            discounted_rewards[i] = rewards[i]
        else:
            discounted_rewards[i] = rewards[i] + gamma * discounted_rewards[i+1]

    return discounted_rewards



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
        "enemies":              [2],                # [1, 2, 3, 4, 5, 6, 7, 8]
        "n_hidden_neurons":     10,
        "epochs":               500,
    }

    config["experiment_name"] = f'{config["enemies"]}_ppo'

    RUN_PPO = False

    # Track time
    if RUN_PPO:
        start_time = time.time()
        ppo_main(**config)
        # Print time in minutes and seconds
        print(f'\nTotal runtime: {round((time.time() - start_time) / 60, 2)} minutes')
        print(f'Total runtime: {round((time.time() - start_time), 2)} seconds')
    else:
        run_test(config, deterministic=False)
