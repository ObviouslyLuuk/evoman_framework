Evoman is a video game playing framework to be used as a testbed for optimization algorithms.

A demo can be found here:  https://www.youtube.com/watch?v=ZqaMjd1E4ZI

Project by Alexander, Mamoune, Marisini, and Luuk

Notes:
- could be useful tracking the duplicate fitness scores to see how diverse the population is
inputs are limited in the following ways:
- no info about where the player is on the map
- no info about the map itself (like if there's a hill in front of the player)

Log:
- changed input normalization
- changed fitness function (this had a huge impact)
- parent selection multinomial seems to do better than greedy and tournament
