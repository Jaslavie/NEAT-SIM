# Emergent Naval Tactics via evolutionary search on latent strategy representation
> "Can evolutionary search over latent knowledge representation spaces discover novel military tactics beyond those historically employed — using the Battle of Trafalgar as a testbed?"

The military simulation space is unsuprisingly sparse. A key bottleneck in wargamining is data -- data which is much more difficult to get through layers of government classifications and beaurcracy. 

So how can we create faithful representations of possible realities with limited data? One approach is to distill knowledge into compressed representations (representation learning) and get a model to learn from this sparse data probilistically and try to recreate an approximate representation (Variational autoencoders). We can then use a smart search algorithm to find different variations of an optimal strategy from the sparse knowledge representations (evolutionary search). 

## Sources
- Evolutionary planning in latent space knowledge representation: https://arxiv.org/pdf/2011.11293
- Discovering representations with black box optimization: https://dl.acm.org/doi/abs/10.1145/3377930.3390221
- NEAT paper: https://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pd
- battle of trafalgar order of battle: https://en.wikipedia.org/wiki/Battle_of_Trafalgar_order_of_battle 
- battle timeline: https://www.rmg.co.uk/stories/maritime-history/battle-trafalgar-timeline 

## Simulation design
We use the **Battle of Trafalgar** as our game setup (a classic demonstration of the Lanchester square problem of concentrated firepower). This was a close combat naval engagement between the british and french/spanish forces. The british General Nelson famously used concentrated firepower to beat the franco-spanish forces which he was outnumbered by 27-to-33. 

The goal of this simulation is to test if emergent tactics can be derived beyond the strategies employed during battle that could yield equally fruitful outcomes.

Fitness function is measured on:
1. victory condition: we reward victories. victory is a significant reduction in enemy forces
2. kill ratio: reward teams with higher ratios of enemy assets eliminated for each friendly
3. lanchester bonus: we reward forces with initial fighting strength significantly greater than its opponents.

## Evaluation
- **Approximate Nash Equilibria**
- 