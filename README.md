Evoman is a video game playing framework to be used as a testbed for optimization algorithms.

Project by Alexander, Mamoune, Marisini, and Luuk

OUR CODE:
- optimization_dummy.py: contains the optimization algorithm in main() and run_test() which runs the game with the best solution and prints the results
- run_optim.py: code to run the optimization algorithm for a number of runs and different enemies at once, or run tests
- helpers.py: contains code for saving results and loading solutions/populations
- custom_controller.py: contains our custom controller that is used to play the game (main addition is normalizing the inputs differently)
- eval_best.py: code to test every run with the best solution 5 times and save the results
- environment.yml: conda environment file

Log:
- changed input normalization
- changed fitness function
- parent selection multinomial (fitness proportionate selection) seems to do better than greedy and tournament
- same thing for survivor selection
- tried crossover like described in the multi_evolution baseline paper, and also an ensemble crossover method, but didn't improve
- implemented adaptive decaying mutation noise based on average fitness of the population
- experimented with evaluating each individual on multiple enemy positions, this improves randomini performance obviously

Notes:
- could be useful tracking the duplicate fitness scores to see how diverse the population is
- look into randomini in the environment, for evolving?
    - this is interesting. Evolving with randomini does way worse. Maybe because the fittest individuals might be complete shit in the next gen, so there's too much
    randomness? Could try only starting randomini after reaching a certain fitness. Could also try multiple randomini evals per generation to deemphasize chance
    - Implemented multi-ini where the randomness is taken away and it just evaluates every individual on every enemy position
- for task II be mindful of whether they use their player_controller for the competition or not (this would affect whether we train with theirs or our own)
inputs are limited in the following ways:
- no info about where the player is on the map
- no info about the map itself (like if there's a hill in front of the player)
- no info about the state of the enemy (for enemy 4 specifically)
- only a single timestep of inputs
