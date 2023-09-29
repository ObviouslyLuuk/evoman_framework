import os
import json
import numpy as np
from tqdm import tqdm

from evoman.environment import Environment
from custom_controller import player_controller
from optimization_dummy import fitness_balanced

from helpers import RESULTS_DIR, ENEMY_POSITIONS


def run_test(folder, enemies=None, randomini_test="no", multi_ini_test=False, results_dir=RESULTS_DIR):
    best_solution = np.loadtxt(f'{results_dir}/{folder}/best.txt')
    with open(f"{results_dir}/{folder}/config.json", "r") as f:
        config = json.load(f)
    if not enemies:
        enemies             = config["enemies"]
    n_hidden_neurons        = config["n_hidden_neurons"]
    normalization_method    = config["normalization_method"]

    env = Environment(experiment_name=f'{results_dir}/{folder}',
                    multiplemode="no",
                    enemies=[enemies[0]],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons, normalization_method), # you can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False,
                    randomini=randomini_test)
    env.player_controller.env = env

    gain = 0
    fs = []
    wins = []
    for enemy in enemies:
        env.enemies = [enemy]

        if not multi_ini_test:
            f,p,e,t = env.play(pcont=best_solution)
            gain += p-e
            fs.append(f)
            wins.append(e<=0)
        else:
            enemy_positions = ENEMY_POSITIONS[enemy]

            gain_ = 0
            fs_ = []
            wins_ = []
            for enemy_position in enemy_positions:
                env.player_controller.x_dist = enemy_position
                f,p,e,t = env.play(pcont=best_solution)
                gain_ += p-e
                fs_.append(f)
                wins_.append(e<=0)
            print(len(fs_))
            gain += gain_ / len(enemy_positions)
            fs.append(np.mean(fs_))
            wins.append(np.mean(wins_))


    fitness = fs[0]
    if len(enemies) > 1:
        fitness = np.mean(fs) - np.std(fs)

    return {
        "gain": gain,
        "fitness": fitness,
        "fitness_balanced": fitness_balanced(p, e, t),
        "wins": wins,
    }


# Takes about 30 seconds per 100 experiment runs to run this
if __name__ == "__main__":
    folders = [f for f in os.listdir(RESULTS_DIR) if os.path.isdir(f"{RESULTS_DIR}/{f}")]

    test_n = 5
    enemies = None
    test_all_enemies = False

    # randomini_values = ["yes", "no"]
    randomini_values = ["no"]
    multi_ini_values = [True, False]

    if test_all_enemies:
        enemies = [1,2,3,4,5,6,7,8]

    for folder in tqdm(folders):
        # Get gen from config.json
        with open(f"{RESULTS_DIR}/{folder}/config.json", "r") as f:
            config = json.load(f)
        gen = config["gen"]
        if enemies != [1,2,3,4,5,6,7,8]:
            enemies = config["enemies"]

        for randomini_test in randomini_values:            
            for multi_ini_test in multi_ini_values:
                if randomini_test == "yes" and multi_ini_test:
                    continue
                add_str = ""
                use_n = test_n
                if randomini_test == "yes":
                    add_str = "_randomini"
                    use_n = test_n
                elif multi_ini_test:
                    add_str = "_multi-ini"
                    use_n = 1

                # Skip run if already eval_best.json for the config and latest generation
                if os.path.exists(f"{RESULTS_DIR}/{folder}/eval_best{add_str}.json"):
                    with open(f"{RESULTS_DIR}/{folder}/eval_best{add_str}.json", "r") as f:
                        saved = json.load(f)
                    if gen == saved["gen"] and saved["enemies"] == enemies:
                        print(f"Skipping {folder} because already evaluated gen {gen} with enemies {enemies}")
                        continue

                test_results = {"gen": gen, "enemies": enemies, "results": []}
                for i in range(use_n):
                    test_results["results"].append(run_test(folder, enemies=enemies, randomini_test=randomini_test, multi_ini_test=multi_ini_test, results_dir=RESULTS_DIR))
            
                with open(f"{RESULTS_DIR}/{folder}/eval_best{add_str}.json", "w") as f:
                    json.dump(test_results, f, indent=4)
