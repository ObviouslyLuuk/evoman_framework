import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from helpers import RESULTS_DIR

def create_plot(variable, folders1, folders2=None, figsize=(10,5), save_png=False, results_dir=RESULTS_DIR):
    """
    Creates a plot of the fitness vs generation for the given folder.
    One line for the average best fitness, one for the average mean fitness and a band for the standard deviation of both.
    Data is aggregated from all the experiment's runs in different folders. Thus the lines are averaged over all runs.
    """
    if not folders1:
        print('No folders given')
        return

    dfs = []
    for folder in folders1:
        # Read data from csv file
        dfs.append(pd.read_csv(f'{results_dir}/{folder}/results.csv'))

    # Get shortest df
    min_len = min([len(df) for df in dfs])
    
    # Aggregate data
    aggr_df = pd.concat(dfs).groupby('gen').agg({'best': ['mean', 'std'], 'mean': ['mean', 'std']})
    aggr_df = aggr_df.iloc[:min_len] # Clip to shortest df
    
    # Plot gen vs best fitness
    plt.figure(figsize=figsize)
    plt.plot(aggr_df['best','mean'], color='green')
    plt.plot(aggr_df['mean','mean'], color='green', linestyle='dashed')
    plt.fill_between(range(len(aggr_df['best','mean'])), aggr_df['best','mean'] - aggr_df['best','std'], aggr_df['best','mean'] + aggr_df['best','std'], alpha=0.3, color='green')
    plt.fill_between(range(len(aggr_df['mean','mean'])), aggr_df['mean','mean'] - aggr_df['mean','std'], aggr_df['mean','mean'] + aggr_df['mean','std'], alpha=0.1, color='green')

    if folders2:
        dfs = []
        for folder in folders2:
            # Read data from csv file
            dfs.append(pd.read_csv(f'{results_dir}/{folder}/results.csv'))
        
        # Aggregate data
        aggr_df = pd.concat(dfs).groupby('gen').agg({'best': ['mean', 'std'], 'mean': ['mean', 'std']})
        aggr_df = aggr_df.iloc[:min_len]

        # Plot gen vs best fitness
        plt.plot(aggr_df['best','mean'], color='blue')
        plt.plot(aggr_df['mean','mean'], color='blue', linestyle='dashed')
        plt.fill_between(range(len(aggr_df['best','mean'])), aggr_df['best','mean'] - aggr_df['best','std'], aggr_df['best','mean'] + aggr_df['best','std'], alpha=0.3, color='blue')
        plt.fill_between(range(len(aggr_df['mean','mean'])), aggr_df['mean','mean'] - aggr_df['mean','std'], aggr_df['mean','mean'] + aggr_df['mean','std'], alpha=0.1, color='blue')

        # Add legend
        methods = list(variable.values())[0]
        plt.legend(
            [f'Avg Best {methods[0]}', f'Avg Mean {methods[0]}', f'Avg Best {methods[1]}', f'Avg Mean {methods[1]}', f'Best Std {methods[0]}', f'Mean Std {methods[0]}', f'Best Std {methods[1]}', f'Mean Std {methods[1]}'],
            bbox_to_anchor=(1.05, 1), loc='upper left'
        )
    else:
        # Add legend
        plt.legend(['Avg Best', 'Avg Mean', 'Best Std', 'Mean Std'])

    plt.title('Fitness vs generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    if save_png:
        if not os.path.exists(f'plots/{str(variable)}'):
            os.makedirs(f'plots/{str(variable)}')
        plt.savefig(f'plots/{str(variable)}/plot.png')
    plt.show()

def create_boxplot(variable, folders1, folders2=None, metric="gain", figsize=(10,5), save_png=False, results_dir=RESULTS_DIR, randomini_eval=False):
    """
    Creates a boxplot for one experiment with multiple runs. Each of these runs has 5 final evaluations of the best solution in eval_best.json.
    Each boxplot datapoint represents one run, so it's the mean of that run's 5 evals.
    Can use the gain, default fitness, balanced fitness or number of wins as the metric.
    """
    if not folders1:
        print('No folders given')
        return

    add_str = ""
    if randomini_eval:
        add_str = "_randomini"

    runs = []
    for folder in folders1:
        # Create empty df with columns for [gain, fitness, fitness_balanced, n_wins]
        df = pd.DataFrame(columns=['gain', 'fitness', 'fitness_balanced', 'wins'])

        # Read results from eval_best.json
        with open(f'{results_dir}/{folder}/eval_best{add_str}.json', 'r') as f:
            saved = json.load(f)
        df = pd.DataFrame(saved["results"])
        # Turn wins list into number of wins if wins is a list type
        if type(df['wins'][0]) == list:
            df['wins'] = df['wins'].apply(lambda x: sum(x))

        # Average over the 5 evals, keep dims
        df = df.mean(axis=0)
        df = df.to_frame().transpose()
        runs.append(df)
    runs = pd.concat(runs, axis=0)

    data = runs[metric]
    plt.figure(figsize=figsize)

    if folders2:
        runs = []
        for folder in folders2:
            # Create empty df with columns for [gain, fitness, fitness_balanced, n_wins]
            df = pd.DataFrame(columns=['gain', 'fitness', 'fitness_balanced', 'wins'])

            # Read results from eval_best.json
            with open(f'{results_dir}/{folder}/eval_best{add_str}.json', 'r') as f:
                saved = json.load(f)
            df = pd.DataFrame(saved["results"][0], index=[0])
            for result in saved["results"][1:]:
                df = pd.concat([df, pd.DataFrame(result, index=[0])], ignore_index=True)
            # Turn wins list into number of wins
            if type(df['wins'][0]) == list:
                df['wins'] = df['wins'].apply(lambda x: sum(x))

            # Average over the 5 evals
            df = df.mean(axis=0)
            df = df.to_frame().transpose()
            runs.append(df)
        runs = pd.concat(runs, axis=0)

        data = [data, runs[metric]]

    # Plot boxplot(s)
    plt.boxplot(data)
    plt.xticks([1,2], list(variable.values())[0])
    plt.title(f'{metric.capitalize()} boxplot')

    plt.xlabel('Method')
    plt.ylabel(metric.capitalize())
    if save_png:
        if not os.path.exists(f'plots/{str(variable)}'):
            os.makedirs(f'plots/{str(variable)}')
        plt.savefig(f'plots/{str(variable)}/{metric}_boxplot.png')
    plt.show()


