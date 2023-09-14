import os
import json
import numpy as np
from tqdm import tqdm

from evoman.environment import Environment
from custom_controller import player_controller
from optimization_dummy import fitness_balanced

from helpers import RESULTS_DIR


def run_test(folder, enemies=None, randomini="no"):
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
                        visuals=False,
                        randomini=randomini)

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
        "fitness_balanced": fitness_balanced(p, e, t),
        "wins": wins,
    }


# Takes about 30 seconds to run
if __name__ == "__main__":
    # Get all results folders
    results_folders = [f for f in os.listdir(RESULTS_DIR) if os.path.isdir(f"{RESULTS_DIR}/{f}")]

    test_n = 5
    randomini = "no"
    enemies = None
    test_all_enemies = False

    if test_all_enemies:
        enemies = [1,2,3,4,5,6,7,8]

    for folder in tqdm(results_folders):
        # Get gen from config.json
        with open(f"{RESULTS_DIR}/{folder}/config.json", "r") as f:
            config = json.load(f)
        gen = config["gen"]
        if enemies != [1,2,3,4,5,6,7,8]:
            enemies = config["enemies"]

        # Skip run if already eval_best.json for the config and latest generation
        if os.path.exists(f"{RESULTS_DIR}/{folder}/eval_best.json"):
            with open(f"{RESULTS_DIR}/{folder}/eval_best.json", "r") as f:
                saved = json.load(f)
            if gen == saved["gen"] and saved["randomini"] == randomini and saved["enemies"] == enemies:
                print(f"Skipping {folder} because already evaluated gen {gen} with randomini {randomini} and enemies {enemies}")
                continue

        test_results = {"gen": gen, "randomini": randomini, "enemies": enemies, "results": []}
        for i in range(test_n):
            test_results["results"].append(run_test(folder, enemies=enemies, randomini=randomini))
    
        with open(f"{RESULTS_DIR}/{folder}/eval_best.json", "w") as f:
            json.dump(test_results, f, indent=4)
