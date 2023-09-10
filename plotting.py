import pandas as pd
import os
import matplotlib.pyplot as plt
from helpers import RESULTS_DIR

def create_plot(experiment_name, save_png=False):
    """
    Creates a plot of the fitness vs generation for the given folder.
    One line for the best fitness, one for the mean fitness and a band for the standard deviation.
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
    aggr_df = pd.concat(dfs).groupby('gen').mean() # Not sure whether to take the std of the mean or the mean of the std
    aggr_df = aggr_df.iloc[:min_len] # Take shortest df
    
    # Plot gen vs best fitness
    plt.figure(figsize=(10, 5))
    plt.plot(aggr_df['best'], color='green')
    plt.plot(aggr_df['mean'], color='orange')
    plt.fill_between(range(len(aggr_df['mean'])), aggr_df['mean'] - aggr_df['std'], aggr_df['mean'] + aggr_df['std'], alpha=0.5, color='orange')

    # Add legend
    plt.legend(['Best', 'Mean', 'Std'])

    plt.title('Fitness vs generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    if save_png:
        if not os.path.exists(f'plots/{experiment_name}'):
            os.makedirs(f'plots/{experiment_name}')
        plt.savefig(f'plots/{experiment_name}/plot.png')
    plt.show()
