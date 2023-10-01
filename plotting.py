import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from helpers import RESULTS_DIR

def create_plot(variable, folders1, folders2=None, figsize=(10,5), save_png=False, results_dir=RESULTS_DIR, fitness_method="default", plot_separate_lines=False):
    """
    Creates a plot of the fitness vs generation for the given folder.
    One line for the average best fitness, one for the average mean fitness and a band for the standard deviation of both.
    Data is aggregated from all the experiment's runs in different folders. Thus the lines are averaged over all runs.
    """
    if not folders1:
        print('No folders given')
        return
    
    enemies = None

    dfs = []
    for folder in folders1:
        # Read data from csv file
        dfs.append(pd.read_csv(f'{results_dir}/{folder}/results.csv'))

        if not enemies:
            with open(f'{results_dir}/{folder}/config.json', 'r') as f:
                saved_config = json.load(f)
            enemies = saved_config['enemies']

    # Get shortest df
    min_len = min([len(df) for df in dfs])
    
    # Aggregate data
    # Check if this fitness method is in the df
    key = f'best_{fitness_method}'
    if key not in dfs[0].columns:
        fitness_method = "default"

    aggr_df = pd.concat(dfs).groupby('gen').agg({
        f'best_{fitness_method}': ['mean', 'std'], 
        f'mean_{fitness_method}': ['mean', 'std'],
    })
    aggr_df = aggr_df.iloc[:min_len] # Clip to shortest df
    
    # Plot gen vs best fitness
    plt.figure(figsize=figsize)
    if not plot_separate_lines:
        plt.plot(aggr_df[f'best_{fitness_method}','mean'], color='green')
        plt.plot(aggr_df[f'mean_{fitness_method}','mean'], color='green', linestyle='dashed')
        plt.fill_between(range(len(aggr_df[f'best_{fitness_method}','mean'])), aggr_df[f'best_{fitness_method}','mean'] - aggr_df[f'best_{fitness_method}','std'], aggr_df[f'best_{fitness_method}','mean'] + aggr_df[f'best_{fitness_method}','std'], alpha=0.3, color='green')
        plt.fill_between(range(len(aggr_df[f'mean_{fitness_method}','mean'])), aggr_df[f'mean_{fitness_method}','mean'] - aggr_df[f'mean_{fitness_method}','std'], aggr_df[f'mean_{fitness_method}','mean'] + aggr_df[f'mean_{fitness_method}','std'], alpha=0.1, color='green')
    else:
        for df in dfs:
            plt.plot(df[f'best_{fitness_method}'], color='green', alpha=0.3)
            plt.plot(df[f'mean_{fitness_method}'], color='green', linestyle='dashed', alpha=0.3)

    if folders2:
        dfs = []
        for folder in folders2:
            # Read data from csv file
            dfs.append(pd.read_csv(f'{results_dir}/{folder}/results.csv'))
        
        # Aggregate data
        aggr_df = pd.concat(dfs).groupby('gen').agg({f'best_{fitness_method}': ['mean', 'std'], f'mean_{fitness_method}': ['mean', 'std']})
        aggr_df = aggr_df.iloc[:min_len]

        # Plot gen vs best fitness
        if not plot_separate_lines:
            plt.plot(aggr_df[f'best_{fitness_method}','mean'], color='blue')
            plt.plot(aggr_df[f'mean_{fitness_method}','mean'], color='blue', linestyle='dashed')
            plt.fill_between(range(len(aggr_df[f'best_{fitness_method}','mean'])), aggr_df[f'best_{fitness_method}','mean'] - aggr_df[f'best_{fitness_method}','std'], aggr_df[f'best_{fitness_method}','mean'] + aggr_df[f'best_{fitness_method}','std'], alpha=0.3, color='blue')
            plt.fill_between(range(len(aggr_df[f'mean_{fitness_method}','mean'])), aggr_df[f'mean_{fitness_method}','mean'] - aggr_df[f'mean_{fitness_method}','std'], aggr_df[f'mean_{fitness_method}','mean'] + aggr_df[f'mean_{fitness_method}','std'], alpha=0.1, color='blue')
        else:
            for df in dfs:
                plt.plot(df[f'best_{fitness_method}'], color='blue', alpha=0.3)
                plt.plot(df[f'mean_{fitness_method}'], color='blue', linestyle='dashed', alpha=0.3)

        # Add legend
        methods = list(variable.values())[0]
        if not plot_separate_lines:
            plt.legend(
                [f'Avg Best {methods[0]}', f'Avg Mean {methods[0]}', f'Best Std {methods[0]}', f'Mean Std {methods[0]}', f'Avg Best {methods[1]}', f'Avg Mean {methods[1]}', f'Best Std {methods[1]}', f'Mean Std {methods[1]}'],
                bbox_to_anchor=(1.05, 1), loc='upper left'
            )
        else:
            plt.legend([methods[0], methods[1]], bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # Add legend
        if not plot_separate_lines:
            plt.legend(['Avg Best', 'Avg Mean', 'Best Std', 'Mean Std'])
        else:
            plt.legend(['Best', 'Mean'])

    plt.title(f'Fitness by generation - Enemy {enemies}')
    plt.xlabel('Generation')
    plt.ylabel(f'Fitness ({fitness_method})')
    if save_png:
        if not os.path.exists(f'plots/{str(variable)}'):
            os.makedirs(f'plots/{str(variable)}')
        plt.savefig(f'plots/{str(variable)}/plot.png')
    plt.show()

def create_boxplot(variable, folders1, folders2=None, metric="gain", figsize=(10,5), save_png=False, results_dir=RESULTS_DIR, randomini_eval=False, multi_ini_eval=False):
    """
    Creates a boxplot for one experiment with multiple runs. Each of these runs has 5 final evaluations of the best solution in eval_best.json.
    Each boxplot datapoint represents one run, so it's the mean of that run's 5 evals.
    Can use the gain, default fitness, balanced fitness or number of wins as the metric.
    Also print the results of a t-test comparing the results from folders1 and folders2.
    """
    if not folders1:
        print('No folders given')
        return
    
    enemies = None

    add_str = ""
    if randomini_eval:
        add_str = "_randomini"
    elif multi_ini_eval:
        add_str = "_multi-ini"

    runs = []
    for folder in folders1:
        if not enemies:
            with open(f'{results_dir}/{folder}/config.json', 'r') as f:
                saved_config = json.load(f)
            enemies = saved_config['enemies']

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
    plt.title(f'{metric.capitalize()} boxplot - Enemy {enemies}')

    plt.xlabel('Method')
    plt.ylabel(metric.capitalize())
    if save_png:
        if not os.path.exists(f'plots/{str(variable)}'):
            os.makedirs(f'plots/{str(variable)}')
        plt.savefig(f'plots/{str(variable)}/{metric}_boxplot.png')
    plt.show()

    # Print t-test results
    if folders2:
        print(f'T-test results for {metric}: {ttest_ind(data[1], data[0])}')


