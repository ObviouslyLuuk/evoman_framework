
import pandas as pd
import os
import numpy as np
import json
RESULTS_DIR = 'results'

def save_results(use_folder, gen, best, mean, std, kwarg_dict={}):
    """Save results to csv and print them."""
    experiment_name = '_'.join(use_folder.split('_')[1:])
    print(f'\n GENERATION {gen} best: {round(best,6)} mean: {round(mean,6)} std: {round(std,6)}')

    # Save results using pandas
    # Load csv if it exists
    if os.path.exists(f'{RESULTS_DIR}/{use_folder}/results.csv'):
        df = pd.read_csv(f'{RESULTS_DIR}/{use_folder}/results.csv')
    else:
        df = pd.DataFrame(columns=['gen', 'best', 'mean', 'std'])

    df = df.append({'gen': gen, 'best': best, 'mean': mean, 'std': std}, ignore_index=True)
    df['gen'] = df['gen'].astype(int)

    # Save to csv
    df.to_csv(f'{RESULTS_DIR}/{use_folder}/results.csv', index=False)

    # Save json with all kwargs plus gen, best, mean, std
    kwarg_dict['gen'] = gen
    kwarg_dict['best'] = best
    kwarg_dict['mean'] = mean
    kwarg_dict['std'] = std
    with open(f'{RESULTS_DIR}/{use_folder}/config.json', 'w') as f:
        json.dump(kwarg_dict, f, indent=4)

    # Update overview.csv with best score
    if not os.path.exists(f'{RESULTS_DIR}/overview.csv'):
        overview_df = pd.DataFrame(columns=['experiment_name', 'best_score', 'folder'])
        overview_df.to_csv(f'{RESULTS_DIR}/overview.csv', index=False)
    df = pd.read_csv(f'{RESULTS_DIR}/overview.csv')
    if experiment_name in df['experiment_name'].values:
        # Overwrite best score if current score is better
        if best > df[df['experiment_name'] == experiment_name]['best_score'].values[0]:
            df.loc[df['experiment_name'] == experiment_name, 'best_score'] = best
            df.loc[df['experiment_name'] == experiment_name, 'folder'] = use_folder
    else:
        # Add new row
        df = df.append({'experiment_name': experiment_name, 'best_score': best, 'folder': use_folder}, ignore_index=True)
    df.to_csv(f'{RESULTS_DIR}/overview.csv', index=False)


def find_folder(experiment_name, gens, kwarg_dict):
    """Find folder with experiment_name in it."""
    use_folder = None
    gen = 0
    existing_folders = os.listdir(RESULTS_DIR)
    # Check if any of the existing folder names are in the experiment name
    existing_folders = [folder for folder in existing_folders if '_'.join(folder.split('_')[1:]) == experiment_name] # remove timestamp from folder name

    if len(existing_folders) > 0:
        # Sort by timestamp
        existing_folders = sorted(existing_folders, key=lambda x: x.split('_')[0], reverse=True)
        for folder in existing_folders:
            # Check if config.json exists
            if not os.path.exists(f'{RESULTS_DIR}/{folder}/config.json'):
                if not os.path.exists(f'{RESULTS_DIR}/{folder}/results.csv'):
                    print(f'No results.csv in {folder}. Deleting folder...')
                    # Delete folder and contents
                    os.system(f'rm -rf {RESULTS_DIR}/{folder}')
                    continue
                print(f'No config.json in {folder}. Skipping...')
                continue
            # Load config.json
            with open(f'{RESULTS_DIR}/{folder}/config.json', 'r') as f:
                config = json.load(f)
            # Check if config matches kwarg_dict
            if not all([config[key] == kwarg_dict[key] for key in kwarg_dict if key != 'gens']): # we might want to continue evolution with more gens
                print(f'Config in {folder} does not match kwarg_dict. Skipping...')
                continue
            # Check if gen already reached gens
            if config['gen'] >= gens-1:
                print(f'Gen in {folder} already reached gens. Skipping...')
                continue
            use_folder = folder
            gen = config['gen']+1
            break

    return use_folder, gen


def load_population(domain_lower,
                    domain_upper,
                    pop_size,
                    n_vars,
                    env,
                    eval_fn,
                    fitness_method,
                    use_folder,
                    continue_evo):
    """Load population from file if it exists. Otherwise initialize new population.
    experiment_name is the name of the experiment."""
    if continue_evo:
        print(f'Loading population from {use_folder}...')
        # Load population
        env.load_state()
        pop = env.solutions[0]
        pfit = env.solutions[1]
    else:
        print('Initializing new population...')
        # Initialize population
        pop = np.random.uniform(domain_lower, domain_upper, (pop_size, n_vars))
        # Eval
        pfit = eval_fn(env, pop, fitness_method)
        env.update_solutions([pop, pfit])

    best_idx = np.argmax(pfit)
    mean = np.mean(pfit)
    std = np.std(pfit)
    return pop, pfit, best_idx, mean, std
