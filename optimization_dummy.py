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
from numpy import sqrt, exp
from scipy.stats import rankdata
import os

def fitness_balanced(player_life, enemy_life, time):
    """Returns a balanced fitness, based on the player life, enemy life and time"""
    return .5*(100-enemy_life) + .5*player_life - np.log(time+1)

def simulation(env, x, multi_ini=False, enemies=None):
    """Returns fitness for individual x, where x is a vector of weights and biases"""
    if not multi_ini:
        f,p,e,t = env.play(pcont=x)
        f = {
            "default": f,
            "balanced": fitness_balanced(p, e, t),
        }
    else:
        fitnesses = {
            "default": np.zeros(len(enemies)),
            "balanced": np.zeros(len(enemies)),
        }
        for i, enemy in enumerate(enemies):
            env.enemies = [enemy]
            positions = ENEMY_POSITIONS[enemy]
            fitnesses_ = {
                "default": np.zeros(len(positions)),
                "balanced": np.zeros(len(positions)),
            }
            for j, position in enumerate(positions):
                env.player_controller.x_dist = position
                f,p,e,t = env.play(pcont=x)
                fitnesses_["default"][j] = f
                fitnesses_["balanced"][j] = fitness_balanced(p, e, t)
            fitnesses["default"][i] = np.mean(fitnesses_["default"])
            fitnesses["balanced"][i] = np.mean(fitnesses_["balanced"])
        f = {
            "default": np.mean(fitnesses["default"]) - np.std(fitnesses["default"]),
            "balanced": np.mean(fitnesses["balanced"]) - np.std(fitnesses["balanced"]),
        }
    return f

def evaluate(env, x, multi_ini=False, enemies=None):
    """Returns tuple of two arrays of fitnesses for population x, the first is the default f, the other is the one used.
    x is a numpy array of individuals"""
    both_f = [simulation(env, individual, multi_ini=multi_ini, enemies=enemies) for individual in x]
    f = {
        "default": np.array([f["default"] for f in both_f]),
        "balanced": np.array([f["balanced"] for f in both_f]),
    }
    return f

def calculate_percentile_ranks_prob(array):
    # Get the ranks of elements
    ranks = rankdata(array)
    # Calculate the total number of elements
    N = len(array)
    # Convert ranks to percentile ranks
    percentile_ranks = (ranks - 1) / (N - 1)
    return percentile_ranks / np.sum(percentile_ranks)

def normalize_pop_fitness(pfit):
    """Normalize fitnesses to values between 0 and 1.
    pfit is a numpy array of fitnesses"""
    # Check if max - min is 0
    if np.max(pfit) - np.min(pfit) == 0:
        # Set all fitnesses to 1
        return np.ones_like(pfit)
    
    # Normalize
    return (pfit - np.min(pfit)) / (np.max(pfit) - np.min(pfit))

def tournament_selection(pfit):
    """Return the index of an individual from the population, based on a tournament.
    pfit is a numpy array of fitnesses."""
    # Pick 2 random individuals
    p1 = np.random.randint(0, len(pfit))
    p2 = np.random.randint(0, len(pfit))

    # Return the best of the two
    if pfit[p1] > pfit[p2]:
        return p1
    else:
        return p2

def pick_parent(pop, pfit, method):
    """Return a parent from the population, based on a tournament, or multinomial sampling.
    pop is a numpy array of individuals, where each individual is a numpy array of weights and biases.
    pfit is a numpy array of fitnesses."""
    if method == 'tournament':
        return pop[tournament_selection(pfit)].copy()
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

def mutate_stochastic_decaying(child, std, mutation_rate):
    """ Add random stochastic noise based on the fitness of the individual
    pfit must be in the range [-50, 50] 
    """
    # Create mask of random booleans
    mask = np.random.rand(*child.shape) < mutation_rate
    # Add random noise to weights and biases where mask is True
    child[mask] += np.random.normal(0, std, size=child.shape)[mask]
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
    elif survivor_method == 'tournament':
        idx = np.zeros(int(len(pfit)/2), dtype=int)
        for i in range(len(idx)):
            idx[i] = tournament_selection(pfit)
    return idx

def crossover(parents, crossover_method):
    """Perform crossover on parents, return children.
    parents is a numpy array of parents, where each parent is a numpy array of weights and biases.
    crossover_method default takes parents two by two and weighs the child's genes by a random factor between 0 and 1.
    Two children are produced per pair of parents.
    crossover_method ensemble takes parents two by two and randomly picks one of the two networks per parent."""
    if crossover_method == 'default':
        children = np.zeros((len(parents), parents.shape[1]))
        for i in range(0, len(parents), 2):
            # weigh parents
            p1 = parents[i]
            p2 = parents[i+1]
            for j in range(2):
                weight = np.random.rand()
                children[i+j] = p1*weight + p2*(1-weight)
        return children
    elif crossover_method == 'none':
        return parents.copy()
    elif crossover_method == 'ensemble':
        children = np.zeros((len(parents), *parents.shape[1:]))
        for i in range(0, len(parents), 2):
            p1 = parents[i]
            p2 = parents[i+1]
            for j in range(2):
                # Child's network 0 is one of parent 1's networks
                children[i+j][0] = p1[np.random.randint(2)]
                # Child's network 1 is one of parent 2's networks
                children[i+j][1] = p2[np.random.randint(2)]
        return children

def evolution_step(env, pop, pfit, mutation_rate, mutation_type, fitness_method, pick_parent_method, survivor_method, crossover_method, dom_upper, dom_lower, randomini="no", multi_ini=False, enemies=None):
    """Perform one step of evolution.
    env is the environment.
    pop is a numpy array of individuals, where each individual is a numpy array of weights and biases.
    pfit is a numpy array of fitnesses.
    mutation_rate is the mutation rate."""
    # Normalize fitnesses
    if fitness_method == "rank":
        pfit_norm = calculate_percentile_ranks_prob(pfit["default"])
    else:
        pfit_norm = normalize_pop_fitness(pfit[fitness_method])

    # Print amount of duplicates
    duplicates = len(pfit) - len(np.unique(pfit["balanced"]))
    print(f'Amount of duplicate fitnesses: {duplicates}')
    # mutation_rate += duplicates / len(pop) * 0.5 # Increase mutation rate with more duplicates
    
    # Create new population
    pop_new = np.zeros_like(pop)
    
    # Add random individuals
    add_amount = int(len(pop) / 10)
    pop_new[-add_amount:] = np.random.uniform(dom_lower, dom_upper, size=(add_amount, *pop.shape[1:]))
    
    parents = np.zeros((len(pop)-add_amount, *pop.shape[1:]))
    if pick_parent_method == "greedy":
        # Pick 10 best parents
        best_parents = np.argsort(pfit_norm)[::-1][:10]

        # Copy and repeat parents
        parents = np.repeat(pop[best_parents].copy(), int((len(pop)-add_amount)/10), axis=0)
    else:
        # Select parents
        for i in range(len(pop)-add_amount):
            parents[i] = pick_parent(pop, pfit_norm, method=pick_parent_method).copy()

    # Crossover
    pop_new[:-add_amount] = crossover(parents, crossover_method)

    # Mutate
    if mutation_type == 'normal':
        pop_new = mutate(pop_new, mutation_rate)
    elif mutation_type == 'stochastic_decaying':
        # Stochastic Noise
        starting_std = 0.9  # Replace with your desired starting value
        ending_std = 0.005  # Replace with your desired ending value
        std_std = 0.5       # standard deviation of the standard deviation of the noise

        std = starting_std * np.exp((np.log(ending_std / starting_std) / 100) * np.mean(pfit["balanced"]) + np.random.normal(0, std_std,1)[0] - .5*std_std**2 )
        print(f'>>Std: {std:.6f}')

        pop_new = mutate_stochastic_decaying(pop_new, std=std, mutation_rate=mutation_rate)

    # Clip to domain
    pop_new = np.clip(pop_new, -1, 1)

    # Evaluate new population
    if randomini == "no":
        pfit_new = evaluate(env, pop_new, multi_ini=multi_ini, enemies=enemies)
        
        # Combine old and new population
        pop_combined = np.vstack((pop, pop_new))
        pfit_combined = {
            "default": np.append(pfit["default"], pfit_new["default"]),
            "balanced": np.append(pfit["balanced"], pfit_new["balanced"]),
        }
    else:
        # Combine old and new population
        pop_combined = np.vstack((pop, pop_new))

        # Run evaluation on new population n times and take avg
        n_times = 10

        pfit_combined = {
            "default": np.zeros((n_times, len(pop_combined))),
            "balanced": np.zeros((n_times, len(pop_combined))),
        }
        for i in range(n_times):
            pfit_i = evaluate(env, pop_combined, fitness_method=fitness_method, multi_ini=False)
            pfit_combined["default"][i] = pfit_i["default"]
            pfit_combined["balanced"][i] = pfit_i["balanced"]
        pfit_combined = {
            "default": np.mean(pfit_combined["default"], axis=0),
            "balanced": np.mean(pfit_combined["balanced"], axis=0),
        }

    if fitness_method == "rank":
        selection_pfit_combined = calculate_percentile_ranks_prob(pfit_combined["default"])
    else:
        selection_pfit_combined = pfit_combined[fitness_method]

    # Select survivors
    idx = select_survivors(selection_pfit_combined, survivor_method)
    pop_new = pop_combined[idx]
    pfit_new = {
        "default": pfit_combined["default"][idx],
        "balanced": pfit_combined["balanced"][idx],
    }
    
    # Return new population and fitnesses
    return pop_new, pfit_new


def main(
        experiment_name = 'a1',
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
        crossover_method = "none",
        mutation_type = 'stochastic_decaying', # 'normal', 'stochastic_decaying'
):
    kwarg_dict = locals()

    # choose this for not using visuals and thus making experiments faster
    visuals = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        visuals = False

    # Make results dir if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Find folder
    use_folder, start_gen = find_folder(kwarg_dict)

    if not use_folder:
        milliseconds = int(round(time.time() * 1000))
        use_folder = f'{milliseconds}_{experiment_name}'
        os.makedirs(f'{RESULTS_DIR}/{use_folder}')

    multi = "no"
    actual_enemies = enemies
    if len(enemies) > 1 and randomini == "no": # if randomini is yes, then we manually do multi
        multi = "yes"
    elif randomini == "yes":
        enemies = [enemies[0]]

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=f'{RESULTS_DIR}/{use_folder}',
                    multiplemode=multi,
                    enemies=enemies,
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons, normalization_method, crossover_method=crossover_method), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=visuals,
                    randomini=randomini)
    env.player_controller.env = env

    # number of weights for multilayer with 10 hidden neurons
    if n_hidden_neurons > 0:
        n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
    else:
        n_vars = (env.get_num_sensors()+1)*5

    # Load population
    pop, pfit = load_population(domain_lower, domain_upper, pop_size, n_vars, env, evaluate, fitness_method, use_folder, continue_evo=start_gen>0, crossover_method=crossover_method)

    # For each generation
    for gen in range(start_gen, gens):
        # Perform one step of evolution
        pop, pfit = evolution_step(env, pop, pfit, mutation_rate, mutation_type=mutation_type, fitness_method=fitness_method, 
                                   pick_parent_method=pick_parent_method, survivor_method=survivor_method, crossover_method=crossover_method, dom_upper=domain_upper, dom_lower=domain_lower,
                                   randomini=randomini, multi_ini=multi_ini, enemies=actual_enemies)
        
        # Get stats
        if fitness_method == "rank":
            best_idx    = np.argmax(pfit["default"])
        else:
            best_idx    = np.argmax(pfit[fitness_method])

        results_dict = {
            'gen': gen,
        }
        for key in pfit:
            results_dict[f'best_{key}'] = pfit[key][best_idx]
            results_dict[f'mean_{key}'] = np.mean(pfit[key])
            results_dict[f'std_{key}'] = np.std(pfit[key])
            results_dict[f'Q5_{key}'] = np.quantile(pfit[key], 0.05)
            results_dict[f'Q95_{key}'] = np.quantile(pfit[key], 0.95)
        
        # Save results
        save_results(use_folder, results_dict, kwarg_dict)
    
        # Save best individual
        if crossover_method == "ensemble":
            np.savetxt(f'{RESULTS_DIR}/{use_folder}/best.txt', np.mean(pop[best_idx], axis=0))
        else:
            np.savetxt(f'{RESULTS_DIR}/{use_folder}/best.txt', pop[best_idx])
        
        # Save environment
        if fitness_method == "rank":
            env.update_solutions([pop, pfit["default"]])
        else:
            env.update_solutions([pop, pfit[fitness_method]])
        env.save_state()

    env.state_to_log() # checks environment state


def run_test(config, randomini_test="no", multi_ini_test=False, based_on_eval_best=None):
    """Run the best solution for the given config"""
    enemies = config['enemies']

    folder = get_best(config, based_on_eval_best=based_on_eval_best)
    with open(f'{RESULTS_DIR}/{folder}/config.json', 'r') as f:
        config = json.load(f)
    n_hidden_neurons = config['n_hidden_neurons']
    normalization_method = config['normalization_method']
    fitness_method = config['fitness_method']

    print(config)

    best_solution = np.loadtxt(f'{RESULTS_DIR}/{folder}/best.txt')

    print(f'\nRunning best solution for enemy {enemies}')
    print(f'Best folder: {folder}')
    print(f'Best fitness: {config["best"]}')
    print(f'Best default fitness: {config["best_log"]}')

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

    def print_result(f, p, e, t):
        win_condition = e <= 0
        win_str = 'WON\n' if win_condition else 'LOST\n'
        print(f'Fitness: {f}, player life: {p}, enemy life: {e}, time: {t}')
        print(f' balanced fitness: {fitness_balanced(p, e, t)}')
        print(win_str)

    if not multi_ini_test:
        f,p,e,t = env.play(pcont=best_solution)
        print_result(f, p, e, t)
    else:
        for enemy in enemies:
            env.enemies = [enemy]
            print(f'Running enemy {enemy}')
            enemy_positions = ENEMY_POSITIONS[enemy]

            for enemy_position in enemy_positions:
                env.player_controller.x_dist = enemy_position
                
                f,p,e,t = env.play(pcont=best_solution)
                print_result(f, p, e, t)


if __name__ == '__main__':
    config = {
        # "experiment_name":      'optimization_test',
        "enemies":              [3],                # [1, 2, 3, 4, 5, 6, 7, 8]
        "randomini":            "no",               # "yes", "no"
        "multi_ini":            False,               # True, False
        "normalization_method": "default",  # "default", "domain_specific", "around_0"
        "fitness_method":       "default",         # "default", "balanced", "rank"
        "pick_parent_method":   "multinomial", # "tournament", "multinomial", "greedy"
        "survivor_method":      "multinomial", # "greedy", "multinomial", "tournament"
        "crossover_method":     "none",     # "none", "default", "ensemble"
        "mutation_type":        "stochastic_decaying",      # "stochastic_decaying", "normal"
        "gens":                 30,
        "n_hidden_neurons":     10,
        "pop_size":             100,
    }

    config["experiment_name"] = f'{config["enemies"]}_{config["n_hidden_neurons"]}_inp-norm-{config["normalization_method"]}_f-{config["fitness_method"]}'

    RUN_EVOLUTION = True
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
