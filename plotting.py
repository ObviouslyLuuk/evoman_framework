import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from helpers import RESULTS_DIR

def compare_config(config1, config2):
    """Return True if the two configs are the same, False otherwise. Ignores gen, best, mean and std."""
    ignore = ['gen', 'best', 'mean', 'std']
    for key in config1:
        if key in ignore:
            continue
        if config1[key] != config2[key]:
            return False
    return True

def compare_configs(folders):
    """Return list of folders with the same config."""
    # Check if config.json is the same for all runs
    with open(f'{RESULTS_DIR}/{folders[0]}/config.json', 'r') as f:
        config = json.load(f)
    for folder in folders:
        with open(f'{RESULTS_DIR}/{folder}/config.json', 'r') as f:
            if not compare_config(config, json.load(f)):
                print(f'Config is different for folder {folder}, skipping')
                folders.remove(folder)
    return folders

def create_plot(experiment_name, figsize=(10,5), save_png=False):
    """
    Creates a plot of the fitness vs generation for the given folder.
    One line for the average best fitness, one for the average mean fitness and a band for the standard deviation of both.
    Data is aggregated from all the experiment's runs in different folders. Thus the lines are averaged over all runs.
    """
    # Find folders
    experiment_folders = [f for f in os.listdir(RESULTS_DIR) if '_'.join(f.split('_')[1:]) == experiment_name]
    
    # Check if config.json is the same for all runs
    experiment_folders = compare_configs(experiment_folders)

    dfs = []
    for folder in experiment_folders:
        # Read data from csv file
        dfs.append(pd.read_csv(f'{RESULTS_DIR}/{folder}/results.csv'))

    # Get shortest df
    min_len = min([len(df) for df in dfs])
    
    # Aggregate data
    aggr_df = pd.concat(dfs).groupby('gen').agg({'best': ['mean', 'std'], 'mean': ['mean', 'std']})
    aggr_df = aggr_df.iloc[:min_len] # Take shortest df
    
    # Plot gen vs best fitness
    plt.figure(figsize=figsize)
    plt.plot(aggr_df['best','mean'], color='green')
    plt.plot(aggr_df['mean','mean'], color='orange')
    plt.fill_between(range(len(aggr_df['best','mean'])), aggr_df['best','mean'] - aggr_df['best','std'], aggr_df['best','mean'] + aggr_df['best','std'], alpha=0.3, color='green')
    plt.fill_between(range(len(aggr_df['mean','mean'])), aggr_df['mean','mean'] - aggr_df['mean','std'], aggr_df['mean','mean'] + aggr_df['mean','std'], alpha=0.3, color='orange')

    # Add legend
    plt.legend(['Avg Best', 'Avg Mean', 'Best Std', 'Mean Std'])

    plt.title('Fitness vs generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    if save_png:
        if not os.path.exists(f'plots/{experiment_name}'):
            os.makedirs(f'plots/{experiment_name}')
        plt.savefig(f'plots/{experiment_name}/plot.png')
    plt.show()

def create_boxplot(experiment_name, metric="gain", figsize=(10,5), save_png=False):
    """
    Creates a boxplot for one experiment with multiple runs. Each of these runs has 5 final evaluations of the best solution in eval_best.json.
    Each boxplot datapoint represents one run, so it's the mean of that run's 5 evals.
    Can use the gain, default fitness, balanced fitness or number of wins as the metric.
    """
    # Find folders
    experiment_folders = [f for f in os.listdir(RESULTS_DIR) if '_'.join(f.split('_')[1:]) == experiment_name]
    
    # Check if config.json is the same for all runs
    experiment_folders = compare_configs(experiment_folders)

    eval_config = None

    runs = pd.DataFrame(columns=['gain', 'fitness', 'fitness_balanced', 'wins'])
    for folder in experiment_folders:
        # Create empty df with columns for [gain, fitness, fitness_balanced, n_wins]
        df = pd.DataFrame(columns=['gain', 'fitness', 'fitness_balanced', 'wins'])

        # Read results from eval_best.json
        with open(f'{RESULTS_DIR}/{folder}/eval_best.json', 'r') as f:
            saved = json.load(f)
            if not eval_config:
                eval_config = {k: saved[k] for k in saved if k != "results"}
        for result in saved["results"]:
            df = df.append(result, ignore_index=True)
        # Turn wins list into number of wins
        df['wins'] = df['wins'].apply(lambda x: sum(x))

        # Average over the 5 evals
        df = df.mean(axis=0)
        runs = runs.append(df, ignore_index=True)
    # print(runs)
    print(eval_config)

    # Plot boxplot
    plt.figure(figsize=figsize)
    plt.boxplot(runs[metric])
    plt.title(f'{metric.capitalize()} boxplot')

    plt.xlabel('Method')
    plt.ylabel(metric.capitalize())
    if save_png:
        if not os.path.exists(f'plots/{experiment_name}'):
            os.makedirs(f'plots/{experiment_name}')
        plt.savefig(f'plots/{experiment_name}/{metric}_boxplot.png')
    plt.show()


