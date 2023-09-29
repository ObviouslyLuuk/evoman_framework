
import pandas as pd
import os
import numpy as np
import json
RESULTS_DIR = 'results'

ENEMY_DEFAULT_POSITIONS = {
    1: 640,
    2: 588,
	3: 588,
	4: 537,
	5: 500,
	6: 588,
	7: 635,
	8: 628,
}

ENEMY_RANDOM_POSITIONS = {
	1: [640,500,400,300],
	2: [630,610,560,530],
	3: [640,500,400,300],
	4: [640,500,400,300],
	5: [640,500,400,300],
	6: [640,500,400,300],
	7: [640,500,400,300],
	8: [640,500,400,300],
}

ENEMY_POSITIONS = {n: sorted(list(set([ENEMY_DEFAULT_POSITIONS[n]] + ENEMY_RANDOM_POSITIONS[n]))) for n in range(1,9)}
# ENEMY_POSITIONS[1] = [300]

def save_results(use_folder, results_dict, kwarg_dict={}):
    """Save results to csv and print them."""
    fitness_method = kwarg_dict['fitness_method']
    print(f'\n GENERATION {results_dict["gen"]}  (using {fitness_method} fitness)')
    for fitness_method in ['default', 'balanced']:
        print(f'  {fitness_method} fitness:  best: {round(results_dict[f"best_{fitness_method}"],6)} mean: {round(results_dict[f"mean_{fitness_method}"],6)} std: {round(results_dict[f"std_{fitness_method}"],6)} Q5: {round(results_dict[f"Q5_{fitness_method}"],6)} Q95: {round(results_dict[f"Q95_{fitness_method}"],6)}')

    # Save results using pandas
    # Load csv if it exists
    if os.path.exists(f'{RESULTS_DIR}/{use_folder}/results.csv'):
        df = pd.read_csv(f'{RESULTS_DIR}/{use_folder}/results.csv')
    else:
        # Make dirs
        os.makedirs(f'{RESULTS_DIR}/{use_folder}', exist_ok=True)
        df = pd.DataFrame(columns=results_dict.keys())

    # Concat new row
    new_row = pd.DataFrame([results_dict.values()], columns=results_dict.keys())
    df = pd.concat([df, new_row], ignore_index=True)
    df['gen'] = df['gen'].astype(int)

    # Save to csv
    df.to_csv(f'{RESULTS_DIR}/{use_folder}/results.csv', index=False)

    # Save json with all kwargs plus gen, best, mean, std
    kwarg_dict.update(results_dict)
    with open(f'{RESULTS_DIR}/{use_folder}/config.json', 'w') as f:
        json.dump(kwarg_dict, f, indent=4)


def find_folder(config):
    """Find folder to continue with that agrees with config."""
    use_folder = None
    gen = 0
    existing_folders = os.listdir(RESULTS_DIR)
    existing_folders = compare_configs(existing_folders, config=config, results_dir=RESULTS_DIR)

    if len(existing_folders) > 0:
        # Sort by timestamp
        existing_folders = sorted(existing_folders, key=lambda x: x.split('_')[0], reverse=True)
        for folder in existing_folders:
            # Check if gen already reached gens
            with open(f'{RESULTS_DIR}/{folder}/config.json', 'r') as f:
                saved_config = json.load(f)
            if saved_config['gen'] >= config['gens']-1:
                print(f'Gen in {folder} already reached gens. Skipping...')
                continue
            use_folder = folder
            gen = saved_config['gen']+1
            break

    return use_folder, gen



def find_folders(config):
    """Return list of folders that agrees with the given config."""
    folders = [f for f in os.listdir(RESULTS_DIR) if os.path.isdir(f"{RESULTS_DIR}/{f}")]

    # Get folders that agree with config
    folders = compare_configs(folders, config=config, results_dir=RESULTS_DIR)

    if len(folders) == 0:
        print('No folders found with given config')
    return folders

def get_best(config, based_on_eval_best=None):
    """
    Returns the folder with the best fitness for the given config.
    based_on_eval_best: If None, use config.json, if '' or '_multi-ini' or '_randomini' use eval_best{}.json
    """
    folders = find_folders(config)
    fitness_by_folder = {}
    for folder in folders:
        if based_on_eval_best is None:
            with open(f'{RESULTS_DIR}/{folder}/config.json', 'r') as f:
                config = json.load(f)
            best_fitness = config['best_balanced']
            fitness_by_folder[folder] = best_fitness
        else:
            with open(f'{RESULTS_DIR}/{folder}/eval_best{based_on_eval_best}.json', 'r') as f:
                saved = json.load(f)
            best_fitness = saved['results'][0]['gain']
            fitness_by_folder[folder] = best_fitness
    best_folder = max(fitness_by_folder, key=fitness_by_folder.get)
    return best_folder


def load_population(domain_lower,
                    domain_upper,
                    pop_size,
                    n_vars,
                    env,
                    eval_fn,
                    fitness_method,
                    use_folder,
                    continue_evo,
                    crossover_method=None):
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
        if crossover_method == "ensemble":
            pop = np.random.uniform(domain_lower, domain_upper, (pop_size, 2, n_vars))
        else:
            pop = np.random.uniform(domain_lower, domain_upper, (pop_size, n_vars))
        # Eval
        pfit = eval_fn(env, pop)
        env.update_solutions([pop, pfit[fitness_method]])
    return pop, pfit



def compare_config(config1, config2):
    """Return True if the two configs are the same, False otherwise. Ignores gen, best, mean and std. 
    Only keys in the first config are checked."""
    ignore = ['gens', 'headless']
    for fitness_method in ['default', 'balanced']:
        ignore.extend([f'best_{fitness_method}', f'mean_{fitness_method}', f'std_{fitness_method}', f'Q5_{fitness_method}', f'Q95_{fitness_method}'])
    for key in config1:
        if key in ignore:
            continue
        if config1[key] != config2[key]:
            return False
    return True

def compare_configs(folders, config=None, results_dir=RESULTS_DIR, printing=False):
    """Return list of folders with the same config. Only keys in the first config are checked."""
    # First remove folders that don't have a config.json
    left_over = []
    for folder in folders:
        # Check if config.json exists
        if not os.path.exists(f'{results_dir}/{folder}/config.json'):
            # Delete folder and contents
            # os.system(f'rm -rf {results_dir}/{folder}')
            continue
        left_over.append(folder)
    folders = left_over

    # Check if config.json is the same for all runs
    if not config:
        with open(f'{results_dir}/{folders[0]}/config.json', 'r') as f:
            config = json.load(f)
    selected_folders = []
    for folder in folders:
        with open(f'{results_dir}/{folder}/config.json', 'r') as f:
            folder_config = json.load(f)
        if not compare_config(config, folder_config):
            if printing:
                print(f'Config is different for folder {folder}, skipping')
            continue
        selected_folders.append(folder)
    return selected_folders

