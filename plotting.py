import pandas as pd
import os
import matplotlib.pyplot as plt
from helpers import RESULTS_DIR

def create_plot(experiment_name, figsize=(10,5), save_png=False):
    """
    Creates a plot of the fitness vs generation for the given folder.
    One line for the average best fitness, one for the average mean fitness and a band for the standard deviation of both.
    Data is aggregated from all the experiment's runs in different folders. Thus the lines are averaged over all runs.
    """
    # Find folders
    experiment_folders = [f for f in os.listdir(RESULTS_DIR) if '_'.join(f.split('_')[1:]) == experiment_name]
    
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

def create_boxplot():
    """
    Creates a boxplot for one experiment with multiple runs. Each of these runs has 5 final evaluations of the best solution.
    Each boxplot datapoint represents one run, so it's the mean of that run's 5 evals.
    Can use the gain, default fitness, balanced fitness or number of wins as the metric.
    """
    pass
