Evoman is a video game playing framework to be used as a testbed for optimization algorithms.

A demo can be found here:  https://www.youtube.com/watch?v=ZqaMjd1E4ZI

Project by Alexander, Mamoune, Marisini, and Luuk

Questions:
- When using two different fitness functions in Task I, do you pick one of the two functions to plot for both? Otherwise the differences seen in the plots don't necessarily say anything meaningful (as a simple example one fitness function could be much more strict than the other, whilst it might actually result in better agents if compared by the same function). On the other hand maybe the comparison between methods in these lines matters less than the progress that is seen in the line, in that case of course their own respective fitness functions should be used. Which is expected of us?
- for task II be mindful of whether they use their player_controller for the competition or not (this would affect whether we train with theirs or our own controller)
- In the slides (workshop-standardassigment-2023.pdf) it says on slide 21 that the metric to be shown in the boxplots is energy for task I and gain/number of defeats for task II, but in the FAQ the answer to question 24 says you should use the measure that was used to select individuals (this seems to make less sense because the measure might be different per method, thus making the results incomparable)

Notes:
- could be useful tracking the duplicate fitness scores to see how diverse the population is
- look into randomini in the environment, for evolving?
- for task II be mindful of whether they use their player_controller for the competition or not (this would affect whether we train with theirs or our own)
inputs are limited in the following ways:
- no info about where the player is on the map
- no info about the map itself (like if there's a hill in front of the player)

Todo:
- make plot legends and text larger
- make box plots of those 5 tests (where each point represents a run, so the average of 5 tests), and run a t-test on the results

Log:
- changed input normalization
- changed fitness function (this had a huge impact)
- parent selection multinomial seems to do better than greedy and tournament
