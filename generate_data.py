"""
File to generate fitness evaluations for random initializations to get a better idea of the landscape.
"""

import os
import numpy as np
from tqdm import tqdm

from evoman.environment import Environment
from demo_controller import player_controller
from helpers import RESULTS_DIR, get_random_str

RESULTS_DIR = 'landscape_data'


def main(
        experiment_name='run',
        domain_lower=-1,
        domain_upper=1,
        pop_size=100,
        results_dir=RESULTS_DIR
    ):
    experiment_name += f"_{get_random_str()}"

    # Make results dir if it doesn't exist
    os.makedirs(f'{results_dir}/{experiment_name}', exist_ok=True)
    
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    n_hidden_neurons = 10
    enemies = [1, 2, 3, 4, 5, 6, 7, 8]

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=f'{RESULTS_DIR}/{experiment_name}',
                    multiplemode="no",
                    enemies=[enemies[0]],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False,
                    randomini="no")

    # number of weights for multilayer with 10 hidden neurons
    if n_hidden_neurons > 0:
        n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
    else:
        n_vars = (env.get_num_sensors()+1)*5

    # Initialize population
    pop = np.random.uniform(domain_lower, domain_upper, (pop_size, n_vars))

    # Save population
    np.save(f'{results_dir}/{experiment_name}/pop.npy', pop)

    # Evaluate population
    results = np.ones((pop_size, len(enemies), 3)) # 3 for [pe, ee, t]
    results[:,:,0] *= 0 # Set all pe to 0
    results[:,:,1] *= 100 # Set all ee to 100
    results[:,:,2] *= -1 # Set all t to -1 to show that it's not evaluated yet
    for i, ind in tqdm(enumerate(pop), total=len(pop)): # ~5 iterations per second
        for j, enemy in enumerate(enemies):
            env.enemies = [enemy]
            f,p,e,t = env.play(pcont=ind)
            results[i][j] = [p, e, t]

        # Save results
        np.save(f'{results_dir}/{experiment_name}/results.npy', results)


if __name__ == "__main__":
    main(
        pop_size=10000
    )
