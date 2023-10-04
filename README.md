Evoman is a video game playing framework to be used as a testbed for optimization algorithms.

A demo can be found here:  https://www.youtube.com/watch?v=ZqaMjd1E4ZI

Project by Alexander, Mamoune, Marisini, and Luuk

OUR CODE:
- optimization_dummy.py: contains the optimization algorithm in main() and run_test() which runs the game with the best solution and prints the results
- run_optim.py: code to run the optimization algorithm for a number of runs and different enemies at once, or run tests
- helpers.py: contains code for saving results and loading solutions/populations
- custom_controller.py: contains our custom controller that is used to play the game (main addition is normalizing the inputs differently)
- eval_best.py: code to test every run with the best solution 5 times and save the results
- plotting.py: code to retrieve and plot the results
- plotting.ipynb: notebook for visualizing the results using plotting.py

Questions:
- When using two different fitness functions in Task I, do you pick one of the two functions to plot for both? Otherwise the differences seen in the plots don't necessarily say anything meaningful (as a simple example one fitness function could be much more strict than the other, whilst it might actually result in better agents if compared by the same function). On the other hand maybe the comparison between methods in these lines matters less than the progress that is seen in the line, in that case of course their own respective fitness functions should be used. Which is expected of us?
- for task II be mindful of whether they use their player_controller for the competition or not (this would affect whether we train with theirs or our own controller)
- In the slides (workshop-standardassigment-2023.pdf) it says on slide 21 that the metric to be shown in the boxplots is energy for task I and gain/number of defeats for task II, but in the FAQ the answer to question 24 says you should use the measure that was used to select individuals (this seems to make less sense because the measure might be different per method, thus making the results incomparable)

Notes:
- what if we have two islands, one where each 10 generations (because usually most of the progress is made in the first 10-15 generations) there's a complete reset,
    where the best individuals are carried over to the other island if they're good enough (as a separate species), and then a new population is initialized.
    The other island thus has the best individuals, and can maintain diversity because there's separate species.
- could be an idea to track where the best fitness solution came from
    - in which generation was it added
    - in which generations was it mutated
    - which parents did it crossover from etc
    this could give some insight into what works and what doesn't
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

Todo:
- make plot legends and text larger

Task I Log:
- changed input normalization
- changed fitness function
- parent selection multinomial (fitness proportionate selection) seems to do better than greedy and tournament
- same thing for survivor selection
- tried crossover like described in the multi_evolution baseline paper, and also an ensemble crossover method, but didn't improve
- implemented adaptive decaying mutation noise based on average fitness of the population
- experimented with evaluating each individual on multiple enemy positions, this improves randomini performance obviously

Task II enemies:
- enemy 1: not essential because it only learns to jump and shoot
- enemy 2: not essential because it's the easiest
- enemy 3: maybe, it learns to dodge bullets
- enemy 4: maybe, it learns to dodge the enemy at all times
- enemy 5: not essential because it's the easiest
- enemy 6: maybe, it learns to dodge the enemy mostly, but weird movement
- enemy 7: maybe, only enemy with different physics and ceiling spikes
- enemy 8: not essential because it's very easy

