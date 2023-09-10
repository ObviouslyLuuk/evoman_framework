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

# imports other libs
import numpy as np
import os

RESULTS_DIR = 'results'

def simulation(env, x, fitness_method='balanced'):
    """Returns fitness for individual x, where x is a vector of weights and biases"""
    f,p,e,t = env.play(pcont=x)
    if fitness_method == 'balanced':
        f = .5*(100-e) + .5*p - np.log(t+1)
    return f

def evaluate(env, x, fitness_method='balanced'):
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

def pick_parent(pop, pfit, method='multinomial'):
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
    

def save_results(experiment_name, gen, best, mean, std):
    """Save results to csv and print them."""
    print(f'\n GENERATION {gen} best: {round(best,6)} mean: {round(mean,6)} std: {round(std,6)}')

    # Save results using pandas
    # Load csv if it exists
    if os.path.exists(f'{RESULTS_DIR}/{experiment_name}/results.csv'):
        df = pd.read_csv(f'{RESULTS_DIR}/{experiment_name}/results.csv')
    else:
        df = pd.DataFrame(columns=['gen', 'best', 'mean', 'std'])

    # Add new row
    df = df.append({'gen': gen, 'best': best, 'mean': mean, 'std': std}, ignore_index=True)
    df['gen'] = df['gen'].astype(int)

    # Save to csv
    df.to_csv(f'{RESULTS_DIR}/{experiment_name}/results.csv', index=False)


def load_population(experiment_name,
                    domain_lower,
                    domain_upper,
                    pop_size,
                    n_vars,
                    env,
                    continue_evo=True):
    """Load population from file if it exists. Otherwise initialize new population.
    experiment_name is the name of the experiment."""
    # If population file exists
    if os.path.exists(f'{RESULTS_DIR}/{experiment_name}/results.csv') and continue_evo:
        print('Loading population...')

        # Load population
        env.load_state()
        pop = env.solutions[0]
        pfit = env.solutions[1]

        # Get last gen number from csv
        df = pd.read_csv(f'{RESULTS_DIR}/{experiment_name}/results.csv')
        gen = df['gen'].iloc[-1]+1
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
    if not probabilitic:
        idx = np.argsort(pfit)[::-1][:pop_size]
        pop = pop[idx]
        pfit = pfit[idx]
    else:
        pfit_norm = normalize_pop_fitness(pfit)
        probs = pfit_norm / np.sum(pfit_norm)
        idx = np.random.choice(len(pop), size=pop_size, p=probs, replace=False)
        idx[0] = best_idx # Keep best
        pop = pop[idx]
        pfit = pfit[idx]
    
    return pop, pfit


def evolution_step(env, pop, pfit, mutation_rate, fitness_method='balanced', pick_parent_method='multinomial'):
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
    pop_new, pfit_new = select_survivors(pop_combined, pfit_combined, len(pop), np.argmax(pfit))
    
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
        headless = True,
):
    # choose this for not using visuals and thus making experiments faster
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    if not os.path.exists(f'{RESULTS_DIR}/{experiment_name}'):
        os.makedirs(f'{RESULTS_DIR}/{experiment_name}')

    multi = "no"
    if len(enemies) > 1:
        multi = "yes"

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=f'{RESULTS_DIR}/{experiment_name}',
                    multiplemode=multi,
                    enemies=enemies,
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
        pop, pfit = evolution_step(env, pop, pfit, mutation_rate, fitness_method=fitness_method)
        
        # Get stats
        best_idx = np.argmax(pfit)
        best = pfit[best_idx]
        mean = np.mean(pfit)
        std = np.std(pfit)
        
        # Save results
        save_results(experiment_name, gen, best, mean, std)
    
        # Save best individual
        np.savetxt(f'{RESULTS_DIR}/{experiment_name}/best.txt', pop[best_idx])
        
        # Save environment
        env.update_solutions([pop, pfit])
        env.save_state()

    env.state_to_log() # checks environment state


def run_test(experiment_name, enemies, n_hidden_neurons, normalization_method, fitness_method):
    best_solution = np.loadtxt(f'{RESULTS_DIR}/{experiment_name}/best.txt')

    print('\nRunning best solution:\n')

    multi = "no"
    speed = "normal"
    if len(enemies) > 1:
        multi = "yes"
        # speed = "fastest"

    env = Environment(experiment_name=f'{RESULTS_DIR}/{experiment_name}',
                    multiplemode=multi,
                    enemies=enemies,
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons, normalization_method), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed=speed,
                    visuals=True)
    
    fitness = evaluate(env, [best_solution], fitness_method=fitness_method)
    print(f'Fitness: {fitness}')


if __name__ == '__main__':
    # Set experiment name, enemies and number of hidden neurons
    # These are used for both the evolution and the test
    enemies = [3] # [1, 2, 3, 4, 5, 6, 7, 8]
    n_hidden_neurons = 10
    normalization_method = "domain_specific" # "default", "domain_specific", "around_0"
    fitness_method = "balanced" # "balanced", "default"
    experiment_name = f'{enemies}_{n_hidden_neurons}_inp-norm-{normalization_method}_f-{fitness_method}'

    RUN_EVOLUTION = True

    # Track time
    if RUN_EVOLUTION:
        start_time = time.time()
        main(
            experiment_name=experiment_name, enemies=enemies, n_hidden_neurons=n_hidden_neurons, normalization_method=normalization_method, fitness_method=fitness_method,
            gens=30,
        )
        # Print time in minutes and seconds
        print(f'\nTotal runtime: {round((time.time() - start_time) / 60, 2)} minutes')
        print(f'Total runtime: {round((time.time() - start_time), 2)} seconds')
    else:
        run_test(experiment_name=experiment_name, enemies=enemies, n_hidden_neurons=n_hidden_neurons, normalization_method=normalization_method, fitness_method=fitness_method)
