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

def simulation(env,x):
    """Returns fitness for individual x, where x is a vector of weights and biases"""
    f,p,e,t = env.play(pcont=x)
    return f

def evaluate(env, x):
    """Returns fitnesses for population x in an array.
    x is a numpy array of individuals"""
    return np.array(list(map(lambda y: simulation(env,y), x)))

def normalize_pop_fitness(pfit):
    """Normalize fitnesses to values between 0 and 1.
    pfit is a numpy array of fitnesses"""
    # Check if max - min is 0
    if np.max(pfit) - np.min(pfit) == 0:
        # Set all fitnesses to 1
        return np.ones_like(pfit)
    
    # Normalize
    return (pfit - np.min(pfit)) / (np.max(pfit) - np.min(pfit))

# def normalize_pop_fitness(pfit):
#     # make rank array from fitness array pfit
#     rank = np.argsort(pfit)[::-1]
#     rank = np.exp(-rank)
#     return rank / rank.sum()

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
    

def mutate(child, mutation_rate, mutation_sigma):
    """Mutate child by adding random noise to selection of weights and biases.
    child is a numpy array of weights and biases.
    mutation_rate is the mutation rate."""
    # Create mask of random booleans
    mask = np.random.rand(*child.shape) < mutation_rate

    # Add random noise to weights and biases where mask is True
    child[mask] += np.random.normal(0, mutation_sigma, size=child.shape)[mask]
    
    return child
    

def save_results(experiment_name, gen, best, mean, std):
    print(f'\n GENERATION {gen} best: {round(best,6)} mean: {round(mean,6)} std: {round(std,6)}')

    # Save results using pandas
    df = pd.DataFrame({'gen': [gen], 'best': [best], 'mean': [mean], 'std': [std]})
    df.to_csv(experiment_name+'/results.csv', mode='a', header=False, index=False)


def load_population(experiment_name,
                    domain_lower,
                    domain_upper,
                    pop_size,
                    n_vars,
                    env):
    """Load population from file if it exists. Otherwise initialize new population.
    experiment_name is the name of the experiment."""
    # If population file exists
    if os.path.exists(experiment_name+'/population.npy'):
        print('Loading population...')

        # Load population
        env.load_state()
        pop = env.solutions[0]
        pfit = env.solutions[1]

        # Get last gen number from csv
        df = pd.read_csv(experiment_name+'/results.csv')
        gen = df['gen'].iloc[-1]
    else:
        print('Initializing new population...')

        # Initialize population
        pop = np.random.uniform(domain_lower, domain_upper, (pop_size, n_vars))
        
        # Eval
        pfit = evaluate(env, pop)
        gen = 0
        env.update_solutions([pop, pfit])

    best_idx = np.argmax(pfit)
    mean = np.mean(pfit)
    std = np.std(pfit)
    
    return pop, pfit, gen, best_idx, mean, std


def select_survivors(pop, pfit, pop_size, best_idx, probabilitic=False):
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
        child = mutate(child, mutation_rate, mutation_sigma=mutation_sigma)
        
        # Add to new population
        pop_new[i] = child

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
        n_hidden_neurons = 10,
        domain_upper = 2,
        domain_lower = -2,

        # initialization of pupulation and population parameters
        pop_size = 100,
        gens = 30,
        mutation_rate = 0.2,
):
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


    experiment_name = 'optimization_test'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[2],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons, normalization_method), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)


    # number of weights for multilayer with 10 hidden neurons
    if n_hidden_neurons > 0:
        n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
    else:
        n_vars = (env.get_num_sensors()+1)*5

    # Load population
    pop, pfit, start_gen, best_idx, mean, std = load_population(experiment_name, domain_lower, domain_upper, pop_size, n_vars, env)

    # For each generation
    for gen in range(start_gen, gens):
        # Perform one step of evolution
        pop, pfit = evolution_step(env, pop, pfit, mutation_rate)
        
        # Get stats
        best_idx = np.argmax(pfit)
        best = pfit[best_idx]
        mean = np.mean(pfit)
        std = np.std(pfit)
        
        # Save stats
        stats['generation'].append(gen+1)
        stats['max'].append(best)
        stats['mean'].append(mean)
        stats['q5'].append(np.quantile(pfit, .05))
        stats['q95'].append(np.quantile(pfit, .95))
        stats['min'].append(pfit.min())
        for _pfit in pfit:
            raw_pfit['generation'].append(gen+1)
            raw_pfit['pfit'].append(_pfit)
            raw_pfit['run'].append(run)

        # Save results
        save_results(use_folder, gen, best, mean, std, kwarg_dict)
    
        # Save best individual
        np.savetxt(f'{RESULTS_DIR}/{use_folder}/best.txt', pop[best_idx])
        
        # Save environment
        env.update_solutions([pop, pfit])
        env.save_state()

    # env.state_to_log() # checks environment state


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
    start_time = time.time()
    main()

    # Print time in minutes and seconds
    print(f'\nTotal runtime: {round((time.time() - start_time) / 60, 2)} minutes')
    print(f'Total runtime: {round((time.time() - start_time), 2)} seconds')
