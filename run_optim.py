from optimization_dummy import main, run_test
import time
import os
from helpers import RESULTS_DIR

if __name__ == '__main__':
    enemy_sets = [[1], [2], [3], [4], [5], [6], [7], [8], [7,8]]

    RUN_EVOLUTION = True

    runs_per_experiment = 10

    for i in range(runs_per_experiment):
        print(f'\n\nRun {i+1} of {runs_per_experiment}\n\n')
        # Run evolution for each set of enemies
        for enemies in enemy_sets:
            n_hidden_neurons = 10
            normalization_method = "domain_specific" # "default", "domain_specific", "around_0"
            fitness_method = "balanced" # "balanced", "default"
            experiment_name = f'{enemies}_{n_hidden_neurons}_inp-norm-{normalization_method}_f-{fitness_method}'

            logged_runs = [f for f in os.listdir(RESULTS_DIR) if '_'.join(f.split('_')[1:]) == experiment_name]
            if len(logged_runs) >= runs_per_experiment:
                print(f'Experiment {experiment_name} already run {runs_per_experiment} times, skipping')
                continue

            if RUN_EVOLUTION:
                start_time = time.time()
                main(
                    experiment_name=experiment_name, enemies=enemies, n_hidden_neurons=n_hidden_neurons, normalization_method=normalization_method, fitness_method=fitness_method,
                    gens=15,
                )
                # Print time in minutes and seconds
                print(f'\nTotal runtime: {round((time.time() - start_time) / 60, 2)} minutes')
                print(f'Total runtime: {round((time.time() - start_time), 2)} seconds')
            else:
                run_test(experiment_name=experiment_name, enemies=enemies, n_hidden_neurons=n_hidden_neurons, normalization_method=normalization_method, fitness_method=fitness_method)
