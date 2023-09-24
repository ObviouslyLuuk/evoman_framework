from optimization_dummy import main, run_test
import time
import os
from helpers import RESULTS_DIR, find_folders, compare_configs

if __name__ == '__main__':
    enemy_sets = [[1], [2], [3], [4], [5], [6], [7], [8]]

    RUN_EVOLUTION = False
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
                "randomini":            "no",               # "yes", "no"
                "normalization_method": "domain_specific",  # "default", "domain_specific", "around_0"
                "fitness_method":       "balanced",         # "default", "balanced"
                "pick_parent_method":   "multinomial",
                "survivor_method":      "greedy",
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