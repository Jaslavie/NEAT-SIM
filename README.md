## Implementing the NEAT genetic algorithm
original paper: https://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pd

### Simulation design
We use the **Battle of Trafalgar** as our game setup (a classic demonstration of the Lanchester square problem of concentrated firepower). This was a close combat naval engagement between the british and french/spanish forces. The british General Nelson famously used concentrated firepower to beat the franco-spanish forces which he was outnumbered by 27-to-33. 

The goal of this simulation is to test if emergent tactics can be derived beyond the strategies employed during battle that could yield equally fruitful outcomes.

Fitness function is measured on:
1. victory condition: we reward victories. victory is a significant reduction in enemy forces
2. kill ratio: reward teams with higher ratios of enemy assets eliminated for each friendly
3. lanchester bonus: we reward forces with initial fighting strength significantly greater than its opponents.
