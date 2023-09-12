import os
import json
import numpy as np
from tqdm import tqdm

from evoman.environment import Environment
from custom_controller import player_controller

from helpers import RESULTS_DIR


def run_test(folder, enemies=None):
    best_solution = np.loadtxt(f'{RESULTS_DIR}/{folder}/best.txt')
    with open(f"{RESULTS_DIR}/{folder}/config.json", "r") as f:
        config = json.load(f)
    if not enemies:
        enemies             = config["enemies"]
    n_hidden_neurons        = config["n_hidden_neurons"]
    normalization_method    = config["normalization_method"]

    gain = 0
    fs = []
    wins = []
    for enemy in enemies:
        env = Environment(experiment_name=f'{RESULTS_DIR}/{folder}',
                        multiplemode="no",
                        enemies=[enemy],
                        playermode="ai",
                        player_controller=player_controller(n_hidden_neurons, normalization_method), # you  can insert your own controller here
                        enemymode="static",
                        level=2,
                        speed="fastest",
                        visuals=False)

        f,p,e,t = env.play(pcont=best_solution)
        gain += p-e
        fs.append(f)
        wins.append(e<=0)

    fitness = fs[0]
    if len(enemies) > 1:
        fitness = np.mean(fs) - np.std(fs)

    return {
        "gain": gain,
        "fitness": fitness,
        # TODO: Also add custom fitness function
        "wins": wins,
    }


if __name__ == "__main__":
    # Get all results folders
    results_folders = [f for f in os.listdir(RESULTS_DIR) if os.path.isdir(f"{RESULTS_DIR}/{f}")]

    test_n = 5

    for folder in tqdm(results_folders):

        # TODO: Skip run if already eval_best.json for the config.json gen

        test_results = []
        for i in range(test_n):
            test_results.append(run_test(folder))
    
        with open(f"{RESULTS_DIR}/{folder}/eval_best.json", "w") as f:
            json.dump(test_results, f, indent=4)
