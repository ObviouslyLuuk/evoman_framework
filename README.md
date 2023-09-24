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
- could be useful tracking the duplicate fitness scores to see how diverse the population is
- look into randomini in the environment, for evolving?
    - this is interesting. Evolving with randomini does way worse. Maybe because the fittest individuals might be complete shit in the next gen, so there's too much
    randomness? Could try only starting randomini after reaching a certain fitness. Could also try multiple randomini evals per generation to deemphasize chance
- for task II be mindful of whether they use their player_controller for the competition or not (this would affect whether we train with theirs or our own)
inputs are limited in the following ways:
- no info about where the player is on the map
- no info about the map itself (like if there's a hill in front of the player)

Todo:
- when using different fitness function, save both in the logs
- automatically plot comparisons between methods
- make plot legends and text larger
- make box plots of those 5 tests (where each point represents a run, so the average of 5 tests), and run a t-test on the results

Log:
- changed input normalization
- changed fitness function (this had a huge impact)
- parent selection multinomial seems to do better than greedy and tournament
