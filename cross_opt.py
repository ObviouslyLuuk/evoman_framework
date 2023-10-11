"""
This file takes the current best individuals as its population and runs evolution on them, keeping them as separate species.
"""

import os
import json
import time
import numpy as np
from tqdm import tqdm
import pandas as pd

# Library for playing a noise when the evolution is done
from beepy import beep

from evoman.environment import Environment
from demo_controller import player_controller

from helpers import RESULTS_DIR, get_random_str, find_folders, load_population, get_best
from boost_opt import aggregate_best

CROSS_RESULTS_DIR = 'cross_results'


def crossover(parents):
    """Perform crossover on parents, return children.
    parents is a numpy array of parents, where each parent is a numpy array of weights and biases.
    takes two parents and weighs the child's genes by a weight between 0 and 1.
    10 children are produced per pair of parents."""
    children = []
    parents_log = []
    for p1_i, p1 in enumerate(parents):
        for p2_i, p2 in enumerate(parents):
            if p1_i >= p2_i:
                continue
            # weigh parents
            step = 0.01
            for weight in np.arange(step, 1, step):
                children.append( p1*weight + p2*(1-weight) )
                parents_log.append( (p1_i, p2_i, weight) )
    return children, parents_log

def parent_log_str(p1_i, p2_i, weight):
    return f"{p1_i}+{p2_i} ({weight:.2f}:{1-weight:.2f})"

def main(
        experiment_name='run',
        results_dir=CROSS_RESULTS_DIR,
    ):
    milliseconds = int(round(time.time() * 1000))
    experiment_name = f"{milliseconds}_{experiment_name}_{get_random_str()}"

    # Make results dir if it doesn't exist
    dir_path = f'{results_dir}/{experiment_name}'
    os.makedirs(dir_path, exist_ok=True)
    
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    n_hidden_neurons = 10
    enemies = [1, 2, 3, 4, 5, 6, 7, 8]

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=dir_path,
                    multiplemode="no",
                    enemies=[enemies[0]],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False,
                    randomini="no")

    # Load best population
    # pop, gains = load_best_population(dir_path, min_beaten=5)
    pop, gains = aggregate_best(results_dir="boost_results")

    pop, parents_log = crossover(pop)

    # Evaluate population
    results = np.ones((len(pop), len(enemies), 3)) # 3 for [pe, ee, t]
    results[:,:,0] *= 0 # Set all pe to 0
    results[:,:,1] *= 100 # Set all ee to 100
    results[:,:,2] *= -1 # Set all t to -1 to show that it's not evaluated yet
    for i, ind in tqdm(enumerate(pop), total=len(pop)): # ~5 iterations per second
        ps = []
        es = []
        ts = []
        for j, enemy in enumerate(enemies):
            env.enemies = [enemy]
            f,p,e,t = env.play(pcont=ind)
            results[i][j] = [p, e, t]
            ps.append(p)
            es.append(e)
            ts.append(t)

        results_dict = {
            "parents": parent_log_str(*parents_log[i]),
            "parents_beaten": f'{np.sum(np.array(gains[parents_log[i][0]]) > 0)} & {np.sum(np.array(gains[parents_log[i][1]]) > 0)}',
            "beaten": np.sum(np.array(es) <= 0),
            "p": np.round(ps),
            "e": np.round(es),
            "t": np.round(ts),
        }

        # Save results using pandas
        # Load csv if it exists
        if os.path.exists(f'{dir_path}/results.csv'):
            df = pd.read_csv(f'{dir_path}/results.csv')
        else:
            df = pd.DataFrame([0 for _ in range(len(results_dict.keys()))], index=results_dict.keys()).transpose()

        # Concat new row
        new_row = pd.DataFrame([results_dict.values()], columns=results_dict.keys())
        df = pd.concat([df, new_row], ignore_index=True)

        # Save to csv
        df.to_csv(f'{dir_path}/results.csv', index=False)


if __name__ == "__main__":
    main(
        experiment_name='cross',
        results_dir=CROSS_RESULTS_DIR,
    )
