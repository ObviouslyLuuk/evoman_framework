The competition results are visible on: http://pesquisa.ufabc.edu.br/hal/Evoman.html#results

In [Increasing the Upper Bound for the EvoMan Game
Competition] the most important notes are that it used PPO and got second place in the competition by using the enemy set {1,2,6,7}. It seems that the Gain it achieved (by our assignment's formula) was actually higher than the winner. It got a Gain of 443.04 beating 6/8 enemies.
After the competition they did more runs with PPO for specialized agents which performed pretty much the same as our best results for Task I. The results are in the table below. Looking at very small differences we performed slightly better in 5/8 specialized agents.
They also did runs for a generalized agent training on all 8 enemies and got amazing results, getting a total Gain of 689.47 by our formula and of course beating all enemies.
They Also tried PSO which didn't perform as well and they tried different difficulties.

Our Best Specialized Agent Gain:
- 1: 100
- 2: 94
- 3: 94
- 4: 93.4
- 5: 100
- 6: 95.8
- 7: 98.2
- 8: 94.6

In [Playing Mega Man II with Neuroevolution] they used a regular GA with crossover, small population size of 10, a different fitness function, adaptive mutation std, hidden layers of 32 and 12, greedy selection, 150 generations and enemies {1,4,6,7}, but most importantly they handcrafted the inputs using a domain specific normalization whilst dropping the inputs of the furthest projectiles (we already tried this) and a special function for the distances. They got a gain of 416.8 beating all enemies.