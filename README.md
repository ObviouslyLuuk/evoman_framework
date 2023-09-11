Evoman is a video game playing framework to be used as a testbed for optimization algorithms.

A demo can be found here:  https://www.youtube.com/watch?v=ZqaMjd1E4ZI

Project by Alexander, Mamoune, Marisini, and Luuk

Notes:
- could be useful tracking the duplicate fitness scores to see how diverse the population is
- look into randomini in the environment, for evolving?
- for task II be mindful of whether they use their player_controller for the competition or not (this would affect whether we train with theirs or our own)
inputs are limited in the following ways:
- no info about where the player is on the map
- no info about the map itself (like if there's a hill in front of the player)

Todo:
- make plot legends and text larger
- write code to do 5 evaluations of best for each experiment run (saved results should be energy (p-e, also called individual gain) for task I, and gain or n defeats for task II)
- make box plots of those 5 tests (where each point represents a run, so the average of 5 tests), and run a t-test on the results

Log:
- changed input normalization
- changed fitness function (this had a huge impact)
- parent selection multinomial seems to do better than greedy and tournament
