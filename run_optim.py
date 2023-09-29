from optimization_dummy import main, run_test
import time
import json
import argparse
from helpers import RESULTS_DIR, find_folders, compare_configs

if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--randomini',              type=str, default='no',         help='usage: --randomini yes/no')
    parser.add_argument('--multi_ini',              type=bool, default=False,       help='usage: --multi_ini True/False')
    parser.add_argument('--normalization_method',   type=str, default='default',    help='usage: --normalization_method default/domain_specific/around_0')
    parser.add_argument('--fitness_method',         type=str, default='default',    help='usage: --fitness_method default/balanced/rank')
    parser.add_argument('--pick_parent_method',     type=str, default='tournament', help='usage: --pick_parent_method multinomial/tournament/greedy')
    parser.add_argument('--survivor_method',        type=str, default='multinomial', help='usage: --survivor_method greedy/multinomial/tournament')
    parser.add_argument('--crossover_method',       type=str, default='none',       help='usage: --crossover_method none/default/ensemble')
    parser.add_argument('--mutation_type',          type=str, default='normal',     help='usage: --mutation_type normal/stochastic_decaying')
    parser.add_argument('--enemy_sets',             type=str, default='1,2,3,4,5,6,7,8',   help='usage: --enemy_sets 1,2,3,4,5,6,7,8/12,13/1')

    args = parser.parse_args()

    # Get sets like [[1], [2], [3], [4], [5], [6], [7], [8]] or [[1, 2], [1, 3]] or [[1]] from string
    args.enemy_sets = [[int(e) for e in eset] for eset in args.enemy_sets.split(',')]

    RUN_EVOLUTION = True
    RANDOMINI_TEST = "no"
    MULTI_INI_TEST = False

    # runs_per_experiment = 10
    runs_per_experiment = 1 # 10 is done with the cluster script for parallelization
    if not RUN_EVOLUTION:
        runs_per_experiment = 1

    for i in range(runs_per_experiment):
        print(f'\n\nRun {i+1} of {runs_per_experiment}\n\n')
        # Run evolution for each set of enemies
        for enemies in args.enemy_sets:
            config = {
                # "experiment_name":      'optimization_test',
                "enemies":              enemies,            # [1, 2, 3, 4, 5, 6, 7, 8]
                "randomini":            args.randomini,     # "yes", "no"
                "multi_ini":            args.multi_ini,     # True, False
                "normalization_method": args.normalization_method, # "default", "domain_specific", "around_0"
                "fitness_method":       args.fitness_method,       # "default", "balanced"
                "pick_parent_method":   args.pick_parent_method,   # "multinomial", "tournament", "greedy"
                "survivor_method":      args.survivor_method,      # "greedy", "multinomial"
                "crossover_method":     args.crossover_method,     # "none", "default", "ensemble"
                "mutation_type":        args.mutation_type,        # "normal", "stochastic_decaying"
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
                config = {
                    "enemies": enemies,
                }
                based_on_eval_best = ''
                if MULTI_INI_TEST:
                    based_on_eval_best = '_multi-ini'
                run_test(config, randomini_test=RANDOMINI_TEST, multi_ini_test=MULTI_INI_TEST, based_on_eval_best=based_on_eval_best)
