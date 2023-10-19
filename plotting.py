import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from helpers import RESULTS_DIR

def create_plot(variable, folders1, folders2=None, figsize=(10,5), save_png=False, results_dir=RESULTS_DIR, fitness_method="default", plot_separate_lines=False, exploration_island=False, min_len=None):
    """
    Creates a plot of the fitness vs generation for the given folder.
    One line for the average best fitness, one for the average mean fitness and a band for the standard deviation of both.
    Data is aggregated from all the experiment's runs in different folders. Thus the lines are averaged over all runs.
    """
    if not folders1:
        if folders2:
            folders1 = folders2
        else:
            print('No folders given')
            return
    
    enemies = None
    exploration_gens = None

    dfs = []
    for folder in folders1:
        # Read data from csv file
        dfs.append(pd.read_csv(f'{results_dir}/{folder}/results.csv'))

        if not enemies:
            with open(f'{results_dir}/{folder}/config.json', 'r') as f:
                saved_config = json.load(f)
            enemies = saved_config['enemies']
        if exploration_island and saved_config['exploration_island'] and not exploration_gens:
            exploration_gens = saved_config['exploration_gens']

    # Get shortest df
    if not min_len:
        min_len = min([len(df) for df in dfs])
        print(f'Shortest length: {min_len}')
    
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
    
    methods = ['']
    if folders2:
        methods = list(variable.values())[0]
    # Plot gen vs best fitness
    plt.figure(figsize=figsize)
    if not plot_separate_lines:
        plt.plot(aggr_df[f'best_{fitness_method}','mean'], color='green')
        plt.plot(aggr_df[f'mean_{fitness_method}','mean'], color='green', linestyle='dashed')
        plt.fill_between(range(len(aggr_df[f'best_{fitness_method}','mean'])), aggr_df[f'best_{fitness_method}','mean'] - aggr_df[f'best_{fitness_method}','std'], aggr_df[f'best_{fitness_method}','mean'] + aggr_df[f'best_{fitness_method}','std'], alpha=0.3, color='green')
        plt.fill_between(range(len(aggr_df[f'mean_{fitness_method}','mean'])), aggr_df[f'mean_{fitness_method}','mean'] - aggr_df[f'mean_{fitness_method}','std'], aggr_df[f'mean_{fitness_method}','mean'] + aggr_df[f'mean_{fitness_method}','std'], alpha=0.1, color='green')
    else:
        for df in dfs:
            plt.plot(df[f'best_{fitness_method}'].iloc[:min_len], color='green', alpha=0.3, label=f'Best {methods[0]}')
            # plt.plot(df[f'mean_{fitness_method}'], color='green', linestyle='dashed', alpha=0.3)

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
                plt.plot(df[f'best_{fitness_method}'].iloc[:min_len], color='blue', alpha=0.3, label=f'Best {methods[1]}')
                # plt.plot(df[f'mean_{fitness_method}'], color='blue', linestyle='dashed', alpha=0.3)

        # Add legend
        if not plot_separate_lines:
            plt.legend(
                [f'Avg Best {methods[0]}', f'Avg Mean {methods[0]}', f'Best Std {methods[0]}', f'Mean Std {methods[0]}', f'Avg Best {methods[1]}', f'Avg Mean {methods[1]}', f'Best Std {methods[1]}', f'Mean Std {methods[1]}'],
                bbox_to_anchor=(1.05, 1), loc='upper left'
            )
        else:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # Add legend
        if not plot_separate_lines:
            plt.legend(['Avg Best', 'Avg Mean', 'Best Std', 'Mean Std'])
        else:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if exploration_gens:
        # Make vertical dotted lines at each exploration merge, where each merge happens at gen % exploration_merge == 0
        for i in range(0, min_len, exploration_gens):
            plt.axvline(x=i, color='black', linestyle='dotted')

    plt.title(f'Fitness by generation - Enemy {enemies}')
    plt.xlabel('Generation')
    plt.ylabel(f'Fitness ({fitness_method})')
    if save_png:
        if not os.path.exists(f'plots/{str(variable)}'):
            os.makedirs(f'plots/{str(variable)}')
        plt.savefig(f'plots/{str(variable)}/plot.png')
    plt.show()

def create_boxplot(config_updates, folders_list, metric="gain", figsize=(10,5), save_png=False, results_dir=RESULTS_DIR, randomini_eval=False, multi_ini_eval=False, all_enemies_eval=False, box_width=0.8, y_lim=None, color=None):
    """
    Creates a boxplot for one experiment with multiple runs. Each of these runs has 5 final evaluations of the best solution in eval_best.json.
    Each boxplot datapoint represents one run, so it's the mean of that run's 5 evals.
    Can use the gain, default fitness, balanced fitness or number of wins as the metric.
    Also print the results of a t-test comparing the results from folders1 and folders2.
    """
    no_folders = True
    for folders in folders_list:
        if folders:
            no_folders = False
    if no_folders:
        print('No folders given')
        return

    data = []
    for folders, conf_update in zip(folders_list, config_updates.values()):
        if "max_gen" not in conf_update:
            max_gen = None
        else:
            max_gen = conf_update['max_gen']
        if "min_gen" not in conf_update:
            min_gen = 0
        else:
            min_gen = conf_update['min_gen']

        # Get enemies from config.json
        with open(f'{results_dir}/{folders[0]}/config.json', 'r') as f:
            saved_config = json.load(f)
        enemies = saved_config['enemies']

        add_str = ""
        if randomini_eval:
            add_str = "_randomini"
        elif multi_ini_eval:
            add_str = "_multi-ini"
        elif all_enemies_eval and not enemies == [1, 2, 3, 4, 5, 6, 7, 8]:
            add_str = "_all-enemies"
        
        if max_gen:
            # Filter folders where {results_dir}/{folder}/bests exists
            folders = [folder for folder in folders if os.path.exists(f'{results_dir}/{folder}/bests')]
            add_str += f'_gen{max_gen}'
        if min_gen:
            # Filter folders where config gen is >= min_gen
            folders = [folder for folder in folders if not os.path.exists(f'{results_dir}/{folder}/bests')]
        print(f'Folders after filtering: {len(folders)}')

        runs = []
        for folder in folders:
            # Create empty df with columns for [gain, fitness, fitness_balanced, n_wins]
            df = pd.DataFrame(columns=['gain', 'fitness', 'fitness_balanced', 'wins'])

            # Read results from eval_best.json
            with open(f'{results_dir}/{folder}/eval_best{add_str}.json', 'r') as f:
                saved = json.load(f)
            df = pd.DataFrame(saved["results"])
            if "gains" in df.columns:
                df = df.drop(columns=['gains'])
            # Turn wins list into number of wins if wins is a list type
            if type(df['wins'][0]) == list:
                df['wins'] = df['wins'].apply(lambda x: sum(x))

            # Average over the 5 evals, keep dims
            df = df.mean(axis=0)
            df = df.to_frame().transpose()
            runs.append(df)
        runs = pd.concat(runs, axis=0)

        data.append(runs[metric])

    if len(data) == 1:
        data = data[0]

    plt.figure(figsize=figsize)

    # Plot boxplot(s)
    if not color:
        plt.boxplot(data, widths=box_width)
    else: # Use color as facecolor and black as edgecolor and the mean line
        plt.boxplot(data, widths=box_width, patch_artist=True, boxprops=dict(facecolor=color, color='black', linewidth=1.5), medianprops=dict(color='black', linewidth=1.5))
    plt.xticks(range(1,len(folders_list)+1), list(config_updates.keys()))
    plt.title(f'{metric.capitalize()} boxplot\nEnemy {enemies}')

    if y_lim:
        plt.ylim(y_lim)

    plt.xlabel('Method')
    plt.ylabel(metric.capitalize())
    # if save_png:
    #     if not os.path.exists(f'plots/{str(variable)}'):
    #         os.makedirs(f'plots/{str(variable)}')
    #     plt.savefig(f'plots/{str(variable)}/{metric}_boxplot.png')
    plt.show()

    # Print t-test results
    if len(folders_list) == 1:
        return
    
    # Do T-test for each pair of folders
    for i in range(len(folders_list)):
        for j in range(len(folders_list)):
            if i <= j:
                continue
            method_i = list(config_updates.keys())[i].replace("\n"," ")
            method_j = list(config_updates.keys())[j].replace("\n"," ")
            print(f'T-test results for {metric} between {method_j} and {method_i}: {ttest_ind(data[i], data[j])}')


