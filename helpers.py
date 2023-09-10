
import pandas as pd
import os
import numpy as np
import json
RESULTS_DIR = 'results'

def save_results(experiment_name, gen, best, mean, std, kwarg_dict={}):
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

    # Save json with all kwargs plus gen, best, mean, std
    kwarg_dict['gen'] = gen
    kwarg_dict['best'] = best
    kwarg_dict['mean'] = mean
    kwarg_dict['std'] = std
    with open(f'{RESULTS_DIR}/{experiment_name}/config.json', 'w') as f:
        json.dump(kwarg_dict, f, indent=4)


def load_population(experiment_name,
                    domain_lower,
                    domain_upper,
                    pop_size,
                    n_vars,
                    env,
                    eval_fn,
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
        pfit = eval_fn(env, pop)
        gen = 0
        env.update_solutions([pop, pfit])

    best_idx = np.argmax(pfit)
    mean = np.mean(pfit)
    std = np.std(pfit)
    
    return pop, pfit, gen, best_idx, mean, std
