from optimization_dummy import main, run_test
import time
import argparse
from helpers import RESULTS_DIR, find_folders, compare_configs

if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--randomini',              type=str, default='no',         help='usage: --randomini yes/no')
    parser.add_argument('--multi_ini',              type=bool, default=False,       help='usage: --multi_ini True/False')
    parser.add_argument('--normalization_method',   type=str, default='default',    help='usage: --normalization_method default/domain_specific/around_0')
    parser.add_argument('--fitness_method',         type=str, default='default',    help='usage: --fitness_method default/balanced')
    parser.add_argument('--pick_parent_method',     type=str, default='tournament', help='usage: --pick_parent_method multinomial/tournament/greedy')
    parser.add_argument('--survivor_method',        type=str, default='multinomial', help='usage: --survivor_method greedy/multinomial')
    parser.add_argument('--crossover_method',       type=str, default='none',       help='usage: --crossover_method none/default')

    args = parser.parse_args()

    enemy_sets = [[1], [2], [3], [4], [5], [6], [7], [8]]

    RUN_EVOLUTION = True
    RANDOMINI_TEST = "yes"

    runs_per_experiment = 10
    if not RUN_EVOLUTION:
        runs_per_experiment = 1

    for i in range(runs_per_experiment):
        print(f'\n\nRun {i+1} of {runs_per_experiment}\n\n')
        # Run evolution for each set of enemies
        for enemies in enemy_sets:
            config = {
                # "experiment_name":      'optimization_test',
                "enemies":              enemies,            # [1, 2, 3, 4, 5, 6, 7, 8]
                "randomini":            args.randomini,     # "yes", "no"
                "multi_ini":            args.multi_ini,     # True, False
                "normalization_method": args.normalization_method, # "default", "domain_specific", "around_0"
                "fitness_method":       args.fitness_method,       # "default", "balanced"
                "pick_parent_method":   args.pick_parent_method,   # "multinomial", "tournament", "greedy"
                "survivor_method":      args.survivor_method,      # "greedy", "multinomial"
                "crossover_method":     args.crossover_method,     # "none", "default"
                "gens":                 30,
                "n_hidden_neurons":     10,
                "pop_size":             100,
            }

            config["experiment_name"] = f'{config["enemies"]}_{config["n_hidden_neurons"]}_inp-norm-{config["normalization_method"]}_f-{config["fitness_method"]}'

            if RUN_EVOLUTION:
                logged_runs = compare_configs(find_folders(config), config=config, results_dir=RESULTS_DIR)
                if len(logged_runs) >= runs_per_experiment:
                    print(f'Experiment {config["experiment_name"]} already run {runs_per_experiment} times, skipping')
                    continue

                start_time = time.time()
                main(**config)
                # Print time in minutes and seconds
                print(f'\nTotal runtime: {round((time.time() - start_time) / 60, 2)} minutes')
                print(f'Total runtime: {round((time.time() - start_time), 2)} seconds')
            else:
                run_test(config, randomini_test=RANDOMINI_TEST)