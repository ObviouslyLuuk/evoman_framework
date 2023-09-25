###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys
import pandas as pd
import time
import json

from evoman.environment import Environment
from custom_controller import player_controller
from helpers import save_results, load_population, find_folder, find_folders, RESULTS_DIR, get_best, ENEMY_POSITIONS

# imports other libs
import numpy as np
import os

def fitness_balanced(player_life, enemy_life, time):
    """Returns a balanced fitness, based on the player life, enemy life and time"""
    return .5*(100-enemy_life) + .5*player_life - np.log(time+1)

def simulation(env, x, fitness_method, multi_ini=False, enemies=None):
    """Returns fitness for individual x, where x is a vector of weights and biases"""
    if not multi_ini:
        f,p,e,t = env.play(pcont=x)
        use_f = f
        if fitness_method == 'balanced':
            use_f = fitness_balanced(p, e, t)
    else:
        fitnesses = np.zeros(len(enemies))
        use_fitnesses = np.zeros(len(enemies))
        for i, enemy in enumerate(enemies):
            env.enemies = [enemy]
            positions = ENEMY_POSITIONS[enemy]
            fitnesses_ = np.zeros(len(positions))
            use_fitnesses_ = np.zeros(len(positions))
            for j, position in enumerate(positions):
                env.player_controller.x_dist = position
                f,p,e,t = env.play(pcont=x)
                fitnesses_[j] = f
                use_fitnesses_[j] = f
                if fitness_method == 'balanced':
                    use_fitnesses_[j] = fitness_balanced(p, e, t)
            fitnesses[i] = np.mean(fitnesses_)
            use_fitnesses[i] = np.mean(use_fitnesses_)
        f = np.mean(fitnesses) - np.std(fitnesses)
        use_f = np.mean(use_fitnesses) - np.std(use_fitnesses)
    return f, use_f

def evaluate(env, x, fitness_method, multi_ini=False, enemies=None):
    """Returns tuple of two arrays of fitnesses for population x, the first is the default f, the other is the one used.
    x is a numpy array of individuals"""
    both_f = [simulation(env, individual, fitness_method, multi_ini=multi_ini, enemies=enemies) for individual in x]
    log_f = np.array([f for f, use_f in both_f])
    use_f = np.array([use_f for f, use_f in both_f])
    return log_f, use_f

def normalize_pop_fitness(pfit):
    """Normalize fitnesses to values between 0 and 1.
    pfit is a numpy array of fitnesses"""
    # Check if max - min is 0
    if np.max(pfit) - np.min(pfit) == 0:
        # Set all fitnesses to 1
        return np.ones_like(pfit)
    
    # Normalize
    return (pfit - np.min(pfit)) / (np.max(pfit) - np.min(pfit))

def pick_parent(pop, pfit, method):
    """Return a parent from the population, based on a tournament, or multinomial sampling.
    pop is a numpy array of individuals, where each individual is a numpy array of weights and biases.
    pfit is a numpy array of fitnesses."""
    if method == 'tournament':
        # Pick 2 random parents
        p1 = np.random.randint(0, len(pop))
        p2 = np.random.randint(0, len(pop))
        
        # Return the best of the two
        if pfit[p1] > pfit[p2]:
            return pop[p1]
        else:
            return pop[p2]
    elif method == 'multinomial':
        pfit = normalize_pop_fitness(pfit)
        pfit = pfit**2 # Square fitnesses to increase probability of picking best
        pfit_distribution = pfit / np.sum(pfit)
        return pop[np.random.choice(len(pop), p=pfit_distribution)]
    

def mutate(child, mutation_rate):
    """Mutate child by adding random noise to selection of weights and biases.
    child is a numpy array of weights and biases.
    mutation_rate is the mutation rate."""
    # Create mask of random booleans
    mask = np.random.rand(*child.shape) < mutation_rate

    # Add random noise to weights and biases where mask is True
    child[mask] += np.random.normal(0, 0.1, size=child.shape)[mask]
    
    return child
    

def select_survivors(pfit, survivor_method):
    """Select survivors from population, return indices of selected.
    pfit is a numpy array of fitnesses."""
    if survivor_method == 'greedy':
        idx = np.argsort(pfit)[::-1][:int(len(pfit)/2)]
    elif survivor_method == 'multinomial':
        pfit_norm = normalize_pop_fitness(pfit)
        probs = pfit_norm / np.sum(pfit_norm)
        idx = np.random.choice(len(pfit), size=int(len(pfit)/2), p=probs, replace=False)
        idx[0] = np.argmax(pfit) # Keep best
    return idx


def evolution_step(env, pop, pfit, log_pfit, mutation_rate, fitness_method, pick_parent_method, survivor_method, multi_ini=False, enemies=None):
    """Perform one step of evolution.
    env is the environment.
    pop is a numpy array of individuals, where each individual is a numpy array of weights and biases.
    pfit is a numpy array of fitnesses.
    mutation_rate is the mutation rate."""
    # Normalize fitnesses
    pfit_norm = normalize_pop_fitness(pfit)

    # Print amount of duplicates
    duplicates = len(pfit) - len(np.unique(pfit))
    print(f'Amount of duplicate fitnesses: {duplicates}')
    # mutation_rate += duplicates / len(pop) * 0.5 # Increase mutation rate with more duplicates
    
    # Create new population
    pop_new = np.zeros_like(pop)
    
    # Add random individuals
    add_amount = int(len(pop) / 10)
    pop_new[-add_amount:] = np.random.uniform(-1, 1, size=(add_amount, pop.shape[1]))
    
    if pick_parent_method != 'greedy':
        # For each individual in the population
        for i in range(len(pop)-add_amount):
            # Copy parent
            child = pick_parent(pop, pfit_norm, method=pick_parent_method).copy()

            # Mutate
            child = mutate(child, mutation_rate)
            
            # Add to new population
            pop_new[i] = child
    else:
        # Pick 10 best parents
        best_parents = np.argsort(pfit_norm)[::-1][:10]

        # Copy and repeat parents
        pop_new[:-add_amount] = np.repeat(pop[best_parents], int((len(pop)-add_amount)/10), axis=0)

        # Mutate
        pop_new = mutate(pop_new, mutation_rate)

    # Evaluate new population
    log_pfit_new, pfit_new = evaluate(env, pop_new, fitness_method=fitness_method, multi_ini=multi_ini, enemies=enemies)
    
    # Combine old and new population
    pop_combined = np.vstack((pop, pop_new))
    pfit_combined = np.append(pfit, pfit_new)
    log_pfit_combined = np.append(log_pfit, log_pfit_new)

    # Select survivors
    idx = select_survivors(pfit_combined, survivor_method)
    pop_new = pop_combined[idx]
    pfit_new = pfit_combined[idx]
    log_pfit_new = log_pfit_combined[idx]
    
    # Return new population and fitnesses
    return pop_new, pfit_new, log_pfit_new


def main(
        experiment_name = 'optimization_test',
        enemies = [2],
        n_hidden_neurons = 10,
        domain_upper = 1,
        domain_lower = -1,
        pop_size = 100,
        gens = 100,
        mutation_rate = 0.2,
        normalization_method = "domain_specific",
        fitness_method = "balanced",
        pick_parent_method = "multinomial",
        survivor_method = "greedy",
        randomini = "no",
        headless = True,
        multi_ini = True,
):
    kwarg_dict = locals()

    # choose this for not using visuals and thus making experiments faster
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # Find folder
    use_folder, start_gen = find_folder(kwarg_dict)

    if not use_folder:
        milliseconds = int(round(time.time() * 1000))
        use_folder = f'{milliseconds}_{experiment_name}'
        os.makedirs(f'{RESULTS_DIR}/{use_folder}')

    multi = "no"
    if len(enemies) > 1 and randomini == "no": # if randomini is yes, then we manually do multi
        multi = "yes"
    elif randomini == "yes":
        actual_enemies = enemies
        enemies = [enemies[0]]

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=f'{RESULTS_DIR}/{use_folder}',
                    multiplemode=multi,
                    enemies=enemies,
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons, normalization_method), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False,
                    randomini=randomini)
    env.player_controller.env = env

    # number of weights for multilayer with 10 hidden neurons
    if n_hidden_neurons > 0:
        n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
    else:
        n_vars = (env.get_num_sensors()+1)*5

    # Load population
    pop, pfit, log_pfit, best_idx, mean, std = load_population(domain_lower, domain_upper, pop_size, n_vars, env, evaluate, fitness_method, use_folder, continue_evo=start_gen>0)

    # For each generation
    for gen in range(start_gen, gens):
        # Perform one step of evolution
        pop, pfit, log_pfit = evolution_step(env, pop, pfit, log_pfit, mutation_rate, fitness_method=fitness_method, 
                                   pick_parent_method=pick_parent_method, survivor_method=survivor_method,
                                   multi_ini=multi_ini, enemies=actual_enemies)
        
        # Get stats
        best_idx    = np.argmax(pfit)

        results_dict = {
            'gen': gen,
            'best': pfit[best_idx],
            'mean': np.mean(pfit),
            'std': np.std(pfit),
            'best_log': log_pfit[best_idx],
            'mean_log': np.mean(log_pfit),
            'std_log': np.std(log_pfit),
        }
        
        # Save results
        save_results(use_folder, results_dict, kwarg_dict)
    
        # Save best individual
        np.savetxt(f'{RESULTS_DIR}/{use_folder}/best.txt', pop[best_idx])
        
        # Save environment
        env.update_solutions([pop, pfit])
        env.save_state()

    env.state_to_log() # checks environment state


def run_test(config, randomini_test="no"):
    """Run the best solution for the given config"""
    enemies = config['enemies']
    n_hidden_neurons = config['n_hidden_neurons']
    normalization_method = config['normalization_method']
    fitness_method = config['fitness_method']

    folder = get_best(config)
    with open(f'{RESULTS_DIR}/{folder}/config.json', 'r') as f:
        config = json.load(f)

    best_solution = np.loadtxt(f'{RESULTS_DIR}/{folder}/best.txt')

    print(f'\nRunning best solution for enemy {enemies}')
    print(f'Best folder: {folder}')
    print(f'Best fitness: {config["best"]}')

    env = Environment(experiment_name=f'{RESULTS_DIR}/{folder}',
                    enemies=[enemies[0]],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons, normalization_method), # you can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="normal",
                    visuals=True,
                    randomini=randomini_test)
    env.player_controller.env = env

    for enemy in enemies:
        env.enemies = [enemy]
        print(f'Running enemy {enemy}')
        if randomini_test == "yes":
            enemy_positions = ENEMY_POSITIONS[enemy]
        else:
            enemy_positions = [None]

        for enemy_position in enemy_positions:
            env.player_controller.x_dist = enemy_position
            
            f,p,e,t = env.play(pcont=best_solution)
            win_condition = e <= 0
            win_str = 'WON\n' if win_condition else 'LOST\n'
            print(f'Fitness: {f}, player life: {p}, enemy life: {e}, time: {t}')
            if fitness_method == 'balanced':
                print(f'custom fitness: {fitness_balanced(p, e, t)}')
            print(win_str)


if __name__ == '__main__':
    config = {
        # "experiment_name":      'optimization_test',
        "enemies":              [3],                # [1, 2, 3, 4, 5, 6, 7, 8]
        "randomini":            "yes",               # "yes", "no"
        "multi_ini":            True,               # True, False
        "normalization_method": "default",  # "default", "domain_specific", "around_0"
        "fitness_method":       "balanced",         # "default", "balanced"
        "pick_parent_method":   "multinomial", # "tournament", "multinomial"
        "survivor_method":      "greedy", # "greedy", "multinomial"
        "gens":                 100,
        "n_hidden_neurons":     10,
        "pop_size":             100,
    }

    config["experiment_name"] = f'{config["enemies"]}_{config["n_hidden_neurons"]}_inp-norm-{config["normalization_method"]}_f-{config["fitness_method"]}'

    RUN_EVOLUTION = False
    RANDOMINI_TEST = "yes"

    # Track time
    if RUN_EVOLUTION:
        start_time = time.time()
        main(**config)
        # Print time in minutes and seconds
        print(f'\nTotal runtime: {round((time.time() - start_time) / 60, 2)} minutes')
        print(f'Total runtime: {round((time.time() - start_time), 2)} seconds')
    else:
        run_test(config, randomini_test=RANDOMINI_TEST)
