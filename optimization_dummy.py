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

from evoman.environment import Environment
from custom_controller import player_controller
from helpers import save_results, load_population, find_folder, RESULTS_DIR

# imports other libs
import numpy as np
import os

def fitness_balanced(player_life, enemy_life, time):
    """Returns a balanced fitness, based on the player life, enemy life and time"""
    return .5*(100-enemy_life) + .5*player_life - np.log(time+1)

def simulation(env, x, fitness_method):
    """Returns fitness for individual x, where x is a vector of weights and biases"""
    f,p,e,t = env.play(pcont=x)
    if fitness_method == 'balanced':
        f = fitness_balanced(p, e, t)
    return f

def evaluate(env, x, fitness_method):
    """Returns fitnesses for population x in an array.
    x is a numpy array of individuals"""
    return np.array([simulation(env, individual, fitness_method) for individual in x])

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
    

def select_survivors(pop, pfit, pop_size, best_idx, survivor_method):
    """Select survivors from population.
    pop is a numpy array of individuals, where each individual is a numpy array of weights and biases.
    pfit is a numpy array of fitnesses.
    pop_size is the size of the population."""
    if survivor_method == 'greedy':
        idx = np.argsort(pfit)[::-1][:pop_size]
        pop = pop[idx]
        pfit = pfit[idx]
    elif survivor_method == 'multinomial':
        pfit_norm = normalize_pop_fitness(pfit)
        probs = pfit_norm / np.sum(pfit_norm)
        idx = np.random.choice(len(pop), size=pop_size, p=probs, replace=False)
        idx[0] = best_idx # Keep best
        pop = pop[idx]
        pfit = pfit[idx]
    
    return pop, pfit


def evolution_step(env, pop, pfit, mutation_rate, fitness_method, pick_parent_method, survivor_method):
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
    pfit_new = evaluate(env, pop_new, fitness_method=fitness_method)
    
    # Combine old and new population
    pop_combined = np.vstack((pop, pop_new))
    pfit_combined = np.append(pfit, pfit_new)

    # Select survivors
    pop_new, pfit_new = select_survivors(pop_combined, pfit_combined, len(pop), np.argmax(pfit), survivor_method)
    
    # Return new population and fitnesses
    return pop_new, pfit_new


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
):
    kwarg_dict = locals()

    # choose this for not using visuals and thus making experiments faster
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # Find folder
    use_folder, start_gen = find_folder(experiment_name, gens, kwarg_dict)

    if not use_folder:
        milliseconds = int(round(time.time() * 1000))
        use_folder = f'{milliseconds}_{experiment_name}'
        os.makedirs(f'{RESULTS_DIR}/{use_folder}')

    multi = "no"
    if len(enemies) > 1:
        multi = "yes"

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

    # number of weights for multilayer with 10 hidden neurons
    if n_hidden_neurons > 0:
        n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
    else:
        n_vars = (env.get_num_sensors()+1)*5

    # Load population
    pop, pfit, best_idx, mean, std = load_population(domain_lower, domain_upper, pop_size, n_vars, env, evaluate, fitness_method, use_folder, continue_evo=start_gen>0)

    # For each generation
    for gen in range(start_gen, gens):
        # Perform one step of evolution
        pop, pfit = evolution_step(env, pop, pfit, mutation_rate, fitness_method=fitness_method, pick_parent_method=pick_parent_method, survivor_method=survivor_method)
        
        # Get stats
        best_idx = np.argmax(pfit)
        best = pfit[best_idx]
        mean = np.mean(pfit)
        std = np.std(pfit)
        
        # Save results
        save_results(use_folder, gen, best, mean, std, kwarg_dict)
    
        # Save best individual
        np.savetxt(f'{RESULTS_DIR}/{use_folder}/best.txt', pop[best_idx])
        
        # Save environment
        env.update_solutions([pop, pfit])
        env.save_state()

    env.state_to_log() # checks environment state


def run_test(experiment_name, enemies, n_hidden_neurons, normalization_method, fitness_method, randomini):
    # Check overview.csv for best solution
    if not os.path.exists(f'{RESULTS_DIR}/overview.csv'):
        print('overview.csv does not exist. Exiting...')
        return
    df = pd.read_csv(f'{RESULTS_DIR}/overview.csv')
    if experiment_name not in df['experiment_name'].values:
        print(f'Experiment {experiment_name} not in overview.csv. Exiting...')
        return
    folder = df[df['experiment_name'] == experiment_name]['folder'].values[0]

    best_solution = np.loadtxt(f'{RESULTS_DIR}/{folder}/best.txt')

    print(f'\nRunning best solution for enemy {enemies}')

    multi = "no"
    speed = "normal"
    if len(enemies) > 1:
        multi = "yes"
        # speed = "fastest"

    env = Environment(experiment_name=f'{RESULTS_DIR}/{folder}',
                    multiplemode=multi,
                    enemies=enemies,
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons, normalization_method), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed=speed,
                    visuals=True,
                    randomini=randomini)
    
    f,p,e,t = env.play(pcont=best_solution)
    win_condition = e <= 0
    win_str = 'WON' if win_condition else 'LOST'
    print(win_str)
    print(f'Fitness: {f}, player life: {p}, enemy life: {e}, time: {t}')
    if fitness_method == 'balanced':
        print(f'custom fitness: {fitness_balanced(p, e, t)}')


if __name__ == '__main__':
    # Set experiment name, enemies and number of hidden neurons
    # These are used for both the evolution and the test
    enemies = [3] # [1, 2, 3, 4, 5, 6, 7, 8]
    n_hidden_neurons = 10
    normalization_method = "domain_specific" # "default", "domain_specific", "around_0"
    fitness_method = "balanced" # "balanced", "default"
    randomini = "yes" # "yes", "no"
    experiment_name = f'{enemies}_{n_hidden_neurons}_inp-norm-{normalization_method}_f-{fitness_method}'

    RUN_EVOLUTION = True

    # Track time
    if RUN_EVOLUTION:
        start_time = time.time()
        main(
            experiment_name=experiment_name, enemies=enemies, n_hidden_neurons=n_hidden_neurons, normalization_method=normalization_method, fitness_method=fitness_method, randomini=randomini,
            gens=30,
        )
        # Print time in minutes and seconds
        print(f'\nTotal runtime: {round((time.time() - start_time) / 60, 2)} minutes')
        print(f'Total runtime: {round((time.time() - start_time), 2)} seconds')
    else:
        run_test(experiment_name=experiment_name, enemies=enemies, n_hidden_neurons=n_hidden_neurons, normalization_method=normalization_method, fitness_method=fitness_method, randomini=randomini)
