"""
This file takes the current best individuals as its population and runs evolution on them, keeping them as separate species.
"""

import os
import json
import time
import numpy as np
from tqdm import tqdm
import pandas as pd

# Library for playing a noise when the evolution is done
from beepy import beep

from evoman.environment import Environment
from demo_controller import player_controller

from optimization_dummy import mutate
from helpers import RESULTS_DIR, get_random_str, find_folders, load_population, get_best

BOOST_RESULTS_DIR = 'boost_results'

def fitness_function_min(gains):
    return np.mean(gains, axis=-1) + np.min(gains, axis=-1)

def fitness_function(gains):
    return np.sum(gains, axis=-1)

fitness_fns = {
    'sum': fitness_function,
    'mean+min': fitness_function_min,
}


def save_pop_info(dir_path, pop, gains):
    # Save best population to file
    np.save(os.path.join(dir_path, 'best_pop.npy'), pop)

    gain_per_enemy = np.array(gains)
    gain = np.sum(gain_per_enemy, axis=1)
    enemies_beaten = np.sum(gain_per_enemy > 0, axis=1)

    # Save best individual to txt file
    best_index = np.argmax(gain)
    np.savetxt(os.path.join(dir_path, 'best.txt'), pop[best_index])

    # Save best gain per enemy, best_gain and enemies_beaten to one json
    list_of_dicts = []
    for i in range(pop.shape[0]):
        list_of_dicts.append({
            'enemies_beaten': int(enemies_beaten[i]),
            'gain': float(gain[i]),
            'gain_per_enemy': gain_per_enemy[i].tolist(),
        })
    with open(os.path.join(dir_path, 'best_pop_info.json'), 'w') as f:
        json.dump(list_of_dicts, f, indent=4)

    results_dict = {}
    for i, (eb, g) in enumerate(zip(enemies_beaten, gain)):
        results_dict[f'beaten_{i}'] = eb
        results_dict[f'gain_{i}'] = g
    
    # Save results using pandas
    # Load csv if it exists
    if os.path.exists(f'{dir_path}/results.csv'):
        df = pd.read_csv(f'{dir_path}/results.csv')
    else:
        df = pd.DataFrame([0 for _ in range(len(results_dict.keys()))], index=results_dict.keys()).transpose()

    # Concat new row
    new_row = pd.DataFrame([results_dict.values()], columns=results_dict.keys())
    df = pd.concat([df, new_row], ignore_index=True)

    # Replace row
    df = new_row

    # Save to csv
    df.to_csv(f'{dir_path}/results.csv', index=False)


def aggregate_best(results_dir=BOOST_RESULTS_DIR, length=1):
    boost_folders = [f for f in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, f))]
    pop_stack = []
    gains_stack = []
    for folder in boost_folders:
        if not (os.path.exists(os.path.join(results_dir, folder, 'best_pop.npy')) and os.path.exists(os.path.join(results_dir, folder, 'best_pop_info.json'))):
            continue
        best_pop, gains = load_best_population(os.path.join(results_dir, folder))
        if len(best_pop) != length:
            continue
        pop_stack.append(best_pop)
        gains_stack.append(gains)
    pop_stack = np.stack(pop_stack)                 # Shape (n_folders, n_individuals, n_weights)
    gains_stack = np.stack(gains_stack)             # Shape (n_folders, n_individuals, n_enemies)
    beaten_stack = np.sum(gains_stack > 0, axis=-1) # Shape (n_folders, n_individuals)
    gain_stack = np.sum(gains_stack, axis=-1)       # Shape (n_folders, n_individuals)
    max_beaten = np.max(beaten_stack, axis=0, keepdims=True)       # Shape (1, n_individuals)
    # If beaten is smaller than max_beaten, set gain to 0
    gain_stack[beaten_stack < max_beaten] = 0
    # Get best set of weights for each individual
    best_pop = pop_stack[np.argmax(gain_stack, axis=0), np.arange(pop_stack.shape[1])]
    gains = gains_stack[np.argmax(gain_stack, axis=0), np.arange(pop_stack.shape[1])]    

    return best_pop, gains


def load_best_population(
        dir_path,
        min_beaten=8,
        get_results_dir=RESULTS_DIR,
    ):
    # Check for best population in dir_path
    if os.path.exists(os.path.join(dir_path, 'best_pop.npy')) and os.path.exists(os.path.join(dir_path, 'best_pop_info.json')):
        best_pop = np.load(os.path.join(dir_path, 'best_pop.npy'))
        
        with open(os.path.join(dir_path, 'best_pop_info.json'), 'r') as f:
            saved = json.load(f)
        gains = []
        for result in saved:
            gains.append(result['gain_per_enemy'])
        gains = np.array(gains)

        return best_pop, gains

    folders = [f for f in os.listdir(get_results_dir) if os.path.isdir(os.path.join(get_results_dir, f))]

    # These folders each contain one best individual, saved in best.txt
    # The results for this best individual are saved in eval_best_all-enemies.json
    # This json file is structured as follows:
    # {
    #     "results": [
    #         {
    #             "gains": [gain1, gain2, ..., gain8],
    #         }
    #     ]
    # }
    # We'll use this to construct the gain_per_enemy array
    best_pop = []
    best_results = []
    folders_without_eval = 0
    for folder in tqdm(folders):
        if not os.path.exists(os.path.join(get_results_dir, folder, 'eval_best_all-enemies.json')):
            folders_without_eval += 1
            continue
        with open(os.path.join(get_results_dir, folder, 'eval_best_all-enemies.json'), 'r') as f:
            saved = json.load(f)
        best_results.append(saved['results'][0]['gains'])
        best_pop.append(np.loadtxt(os.path.join(get_results_dir, folder, 'best.txt')))
    best_pop = np.array(best_pop)
    best_gain_per_enemy = np.array(best_results)
    best_gain = np.sum(best_gain_per_enemy, axis=1)

    # Sort the gain array and get the indices
    indices = np.argsort(best_gain)[::-1]
    best_pop = best_pop[indices]
    best_gain_per_enemy = best_gain_per_enemy[indices]
    enemies_beaten = np.sum(best_gain_per_enemy > 0, axis=1)

    # Only select individuals that beat at least min_beaten enemies
    best_pop = best_pop[enemies_beaten >= min_beaten]
    gains = best_gain_per_enemy[enemies_beaten >= min_beaten]

    print(f'Folders without eval: {folders_without_eval}')
    print('Best Population size:', best_pop.shape)

    return best_pop, gains


def evaluate(pop, env, selection, fitness_method="mean+min", enemies=[1, 2, 3, 4, 5, 6, 7, 8]):
    # Evaluate population
    gains_new = np.zeros((len(pop), len(enemies)))
    for i, individual in enumerate(pop):
        if i not in selection:
            gains_new[i] = np.zeros((len(enemies))) -1e6
            continue
        for enemy in enemies:
            env.enemies = [enemy]
            f,p,e,t = env.play(pcont=individual)
            gains_new[i][enemy-1] = p-e
    return gains_new


def evolution_step_lookahead(pop, gains, env, max_so_far, fitness_method="sum", mutation_rate=0.1, mut_std=0.1, enemies=[1, 2, 3, 4, 5, 6, 7, 8], eval_select=None, lookahead_width=5, n_lookahead=1):
    """
    Run one step of evolution on the given population.
    """
    # Copy parents
    # Create stack of lookahead_width children for each parent
    children = np.stack([pop.copy() for _ in range(lookahead_width)], axis=0) # Shape (lookahead_width, pop_size, n_weights)
    children = mutate(children, mutation_rate=mutation_rate, std=mut_std)

    # Only evaluate a selection of the population
    selection = range(len(pop))
    if eval_select is not None:
        selection = eval_select

    if n_lookahead == 0:
        lookahead_pfit_new = []
        for i in range(lookahead_width):
            # Evaluate children
            gains_new = evaluate(children[i], env, selection, fitness_method=fitness_method, enemies=enemies)
            pfit_new = fitness_fns[fitness_method](gains_new)
            lookahead_pfit_new.append(pfit_new)
        avg_pfit_new = np.mean(np.stack(lookahead_pfit_new), axis=0) # Shape (pop_size)
        return avg_pfit_new
    else:
        lookahead_pfit_new = []
        for i in range(lookahead_width):
            avg_pfit_new = evolution_step_lookahead(children[i], gains, env, fitness_method=fitness_method, mut_std=mut_std, mutation_rate=mutation_rate, eval_select=eval_select, lookahead_width=lookahead_width, 
                                             n_lookahead=n_lookahead-1, enemies=enemies, max_so_far=max_so_far)
            lookahead_pfit_new.append(avg_pfit_new)
        lookahead_pfit_new = np.stack(lookahead_pfit_new) # Shape (lookahead_width, pop_size)

    pfit = fitness_fns[fitness_method](gains)
    fitness_diff = np.round(np.max(lookahead_pfit_new, axis=0) - pfit, 2)[selection].tolist()
    print(f'Diff in fitness: {fitness_diff}')

    # Get indices of highest fitness children, so one for each axis 1
    indices = np.argmax(lookahead_pfit_new, axis=0)

    # Get the best individual from each axis 1
    children = children[indices, np.arange(pop.shape[0])]
    gains_new = evaluate(pop, env, selection, fitness_method=fitness_method, enemies=enemies)

    pfit_new = fitness_fns[fitness_method](gains_new) + 1e3 # Add large bias for new individuals

    enemies_beaten = np.sum(gains > 0, axis=1)
    enemies_beaten_new = np.sum(gains_new > 0, axis=1)
    # If there is a difference in the number of enemies beaten, set the other fitness to something very low
    pfit[enemies_beaten < enemies_beaten_new] = -1e6
    pfit_new[enemies_beaten_new < enemies_beaten] = -1e6

    # Set parents and children next to each other
    combined = np.stack((pfit, pfit_new)) # Shape (2, pop_size)

    # Get indices of highest fitness, so one for each axis 1
    indices = np.argmax(combined, axis=0)

    # Get the best individual from each axis 1
    pop = np.stack((pop, children))[indices, np.arange(pop.shape[0])]
    gains = np.stack((gains, gains_new))[indices, np.arange(pop.shape[0])]

    if np.sum(indices) > 0:
        print(f'Replacing {np.sum(indices)} parents with children')
        new_gain = np.sum(gains, axis=1)
        print(f'New Gain: {np.round(new_gain)[selection]}')
        if (new_gain > max_so_far).any():
            beep(sound=1)
            max_so_far = new_gain

    return pop, gains, max_so_far


def evolution_step(pop, gains, env, fitness_method="sum", mutation_rate=0.1, mut_std=0.1, enemies=[1, 2, 3, 4, 5, 6, 7, 8], eval_select=None):
    """
    Run one step of evolution on the given population.
    """
    # Copy parents
    children = pop.copy()
    children = mutate(children, mutation_rate=mutation_rate, std=mut_std)

    # Only evaluate a selection of the population
    selection = range(len(pop))
    if eval_select is not None:
        selection = eval_select

    # Evaluate children
    gains_new = evaluate(children, env, selection, fitness_method=fitness_method, enemies=enemies)

    pfit = fitness_fns[fitness_method](gains)
    pfit_new = fitness_fns[fitness_method](gains_new)
    fitness_diff = np.round(pfit_new - pfit, 2)[selection].tolist()
    print(f'Diff in fitness: {fitness_diff}')
    if np.max(fitness_diff) > 0:
        beep(sound=1)
    pfit_new += 1e-6 # Add small bias for new individuals

    enemies_beaten = np.sum(gains > 0, axis=1)
    enemies_beaten_new = np.sum(gains_new > 0, axis=1)
    # If there is a difference in the number of enemies beaten, set the other fitness to something very low
    pfit[enemies_beaten < enemies_beaten_new] = -1e6
    pfit_new[enemies_beaten_new < enemies_beaten] = -1e6

    # Set parents and children next to each other
    combined = np.stack((pfit, pfit_new)) # Shape (2, pop_size)

    # Get indices of highest fitness, so one for each axis 1
    indices = np.argmax(combined, axis=0)

    if np.sum(indices) > 0:
        print(f'Replacing {np.sum(indices)} parents with children')

    # Get the best individual from each axis 1
    pop = np.stack((pop, children))[indices, np.arange(pop.shape[0])]
    gains = np.stack((gains, gains_new))[indices, np.arange(pop.shape[0])]

    return pop, gains


def main(
        experiment_name='run',
        gens=300,
        fitness_method="sum",
        mutation_rate=0.1,
        mut_std=0.1,
        eval_select=None,
        results_dir=BOOST_RESULTS_DIR,
    ):
    kwarg_dict = locals()
    kwarg_dict.pop('results_dir')
    kwarg_dict.pop('experiment_name')
    kwarg_dict.pop('gens')

    milliseconds = int(round(time.time() * 1000))
    experiment_name = f"{milliseconds}_{experiment_name}_{get_random_str()}"

    # Make results dir if it doesn't exist
    dir_path = f'{results_dir}/{experiment_name}'
    os.makedirs(dir_path, exist_ok=True)

    # Save config
    with open(f'{dir_path}/config.json', 'w') as f:
        json.dump(kwarg_dict, f, indent=4)
    
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    n_hidden_neurons = 10
    enemies = [1, 2, 3, 4, 5, 6, 7, 8]

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=dir_path,
                    multiplemode="no",
                    enemies=[enemies[0]],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False,
                    randomini="no")

    # Load best population
    # pop, gains = load_best_population(dir_path, min_beaten=8)
    pop, gains = aggregate_best(results_dir=results_dir, length=1)

    save_pop_info(dir_path, pop, gains)

    gain = np.sum(gains, axis=1)
    max_so_far = gain[np.argmax(gain)]

    for i in range(gens):
        print(f'Generation {i}/{gens}')

        pop, gains = evolution_step(pop, gains, env, fitness_method=fitness_method, mut_std=mut_std, mutation_rate=mutation_rate, eval_select=eval_select)
        # pop, gains, max_so_far = evolution_step_lookahead(pop, gains, env, max_so_far, fitness_method=fitness_method, mut_std=mut_std, mutation_rate=mutation_rate, eval_select=eval_select,
        #                                                    lookahead_width=5, n_lookahead=1)

        save_pop_info(dir_path, pop, gains)


if __name__ == "__main__":
    main(
        experiment_name='boost',
        gens=500000,
        fitness_method="sum", # "sum", "mean+min"
        mutation_rate=0.05,
        mut_std=0.001,
        # eval_select=[1],
        eval_select=None,
        results_dir=BOOST_RESULTS_DIR,
    )
