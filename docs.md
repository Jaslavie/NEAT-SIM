# NEAT-SIM Architecture
## Rational Tactic Discovery via Regret Minimization in Learned Latent Strategy Space

> An agent that reasons about what it doesn't know — maintaining explicit beliefs about an adversary's hidden state, navigating a learned latent representation of tactic space through counterfactual regret minimization, and converging toward Nash-equilibrium-approaching strategies without enumerating the game tree.

---

## The Core Idea (Plain Language)

The original approach (NEAT → VAE → random search) builds a good map of tactic space but navigates it blindly. It doesn't know Red exists during search — it just tries things and scores them.

The revised approach treats this as what it actually is: an **imperfect information game**. Blue can't fully observe Red's state. Red is adapting. The right question isn't *"what diverse tactics can I find?"* but *"what is the best strategy distribution given that my opponent is also a strategic agent with hidden state?"*

CFR answers that question. It does so by tracking **regret**: at every point in the game, how much better would Blue have done by playing a different tactic? Over many iterations, the strategy shifts away from high-regret choices and toward low-regret ones. This provably converges toward Nash equilibrium in two-player zero-sum games — the point where neither player can improve by unilaterally changing strategy.

The latent space makes this tractable. Enumerating the full game tree of a 100-timestep continuous simulation is impossible. But the VAE latent space is a compressed, smooth representation of the strategy space — it serves as the abstract action space over which CFR operates.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                   NEAT-SIM Revised Pipeline                         │
│                                                                     │
│  Phase 1: Representation Learning                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                       │
│  │Simulation│───▶│   NEAT   │───▶│   VAE    │                       │
│  │ Engine   │    │Evolution │    │ Training │                       │
│  └──────────┘    └──────────┘    └─────┬────┘                       │
│                                        │ latent tactic space z      │
│  Phase 2: Rational Strategy Discovery  │                            │
│                                  ┌─────▼──────────────────────┐    │
│                                  │   Deep CFR in Latent Space  │    │
│                                  │                             │    │
│   belief state b(t)  ──────────▶ │  Regret Network R(b, z)    │    │
│   (what Blue knows               │  Strategy Net  π(z | b)    │    │
│    about Red's                   │  World Model   T(b, z)     │    │
│    hidden state)                 │                             │    │
│                                  └─────────────┬───────────────┘    │
│                                                │ strategy π         │
│                                  ┌─────────────▼───────────────┐    │
│                                  │       Simulation            │    │
│                                  │  (regret computed here)     │    │
│                                  └─────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

**Phase 1 (unchanged)**: NEAT generates diverse trajectories → VAE learns to compress them into a continuous latent tactic space.

**Phase 2 (new)**: Deep CFR navigates the latent space. Instead of randomly sampling z vectors, a regret network estimates "how much would I regret not playing tactic z given what I currently believe about Red?" The strategy distribution over z shifts iteratively toward Nash equilibrium.

---

## Component 1: Simulation Engine
**File**: `environment.py`

The wargame environment. Implements the Battle of Trafalgar as a Lanchester attrition model.

```
Inputs:  Blue action (4 floats: engage, retreat, attack, hold)
         Red action  (from state machine: effective_units, fire_type)

State:   blue_units, red_units, blue_pos, red_pos, firepower, wind_speed

Hidden:  Red's current FSM state (LINE_FORM / DEFENSE / RETREAT)
         Red's actual firing effectiveness (30% or 80%)
         Red's damage threshold proximity

Step:    1. Blue executes action → damages Red or repositions
         2. Red FSM selects action → damages Blue
         3. Lanchester attrition applied to both sides
         4. Return PARTIAL observation dict + done flag
            (Blue cannot see Red's hidden state)
```

**Why partial information matters**: Red transitions silently between states. Blue observes unit counts and positions but not Red's internal FSM state. This is a genuine imperfect information game — there exist multiple possible Red states consistent with Blue's observations at any timestep.

**Red State Machine (FSM)**
```
     entry
       │
   [LINE_FORM] ──── close range ────▶ [DEFENSE]
       │                                  │
       └─────── units < threshold ──▶ [RETREAT]
```

**Fitness Function** (Lanchester square law):
```
fitness = 0.7 × kill_ratio  +  0.2 × victory_bonus  +  0.1 × lanchester_bonus
```

---

## Component 2: NEAT Evolution (Phase 1 — Representation Seeding)
**File**: `NEAT.ipynb §1–6`

Unchanged in purpose. Evolves a diverse population of neural network policies to generate a rich corpus of tactic trajectories — the training data for the VAE.

```
Each genome produces one trajectory:
  trajectory = [(a₀, a₁, a₂, a₃), ...] for 100 timesteps
             = 100 × 4 floats = 400 numbers
```

**Role in the revised pipeline**: NEAT is now purely a data generator. It explores the tactic space stochastically to give the VAE enough diversity to build a meaningful latent space. The interesting reasoning happens in Phase 2.

---

## Component 3: Trajectory Collection
**File**: `NEAT.ipynb §7` *(to implement)*

Modify `eval_single_genome` to save trajectories and crucially — partial observations.

```
training_data = []

for each genome in each generation:
    trajectory  = genome_actions          # (100, 4) — Blue's actions
    obs_history = observations_seen       # (100, 5) — what Blue observed (partial)
    fitness     = genome.fitness
    training_data.append({
        'trajectory':   trajectory,
        'obs_history':  obs_history,      # NEW: needed for belief state training
        'fitness':      fitness,
        'actual_length': steps_before_done
    })
```

**New addition**: save `obs_history` — the sequence of partial observations Blue received. This is needed to train the belief state encoder in Component 5.

---

## Component 4: Variational Autoencoder (VAE)
**File**: `NEAT.ipynb §8` *(to implement)*

Unchanged architecture. Learns to compress trajectories into a smooth latent space z.

The latent space z now has a specific role: it is the **abstract action space for CFR**. Each point z ∈ ℝ⁸ represents a tactic. CFR will maintain a strategy distribution π(z) over this space and update it via regret.

### 4a. Encoder
```
Input:  trajectory  [batch, 400]
           ↓  Linear(400 → 64) + ReLU
           ↓  Linear(64 → 16)  + ReLU
           ↓
        μ  = Linear(16 → 8)     ← mean
        σ  = Linear(16 → 8)     ← log-variance
```

### 4b. Reparameterization
```
z = μ + exp(σ/2) · ε       ε ~ N(0,1)
```

### 4c. Decoder
```
Input:  z  [batch, 8]
           ↓  Linear(8 → 16)  + ReLU
           ↓  Linear(16 → 64) + ReLU
           ↓  Linear(64 → 400) + Sigmoid
        Output: reconstructed trajectory  [batch, 400]
```

### 4d. Loss
```
Total Loss = MSE(x, x̂)  +  β · KL(N(μ,σ) ∥ N(0,1))
```
KL annealing: β: 0 → 1.0 over first 50 epochs to prevent posterior collapse.

---

## Component 5: Belief State Encoder (NEW)
**File**: `NEAT.ipynb §9` *(to implement)*

**This is the new component** — the piece that makes the agent reason about what it doesn't know.

At each timestep, Blue has received a sequence of partial observations: unit counts, positions, Red's visible behavior. From this history, it must infer a distribution over Red's hidden state (which FSM state is Red in? How close to transitioning?).

```
obs_history [t, 5]          ← sequence of partial observations up to time t
      ↓  GRU encoder
belief_state b [16]         ← compressed belief about Red's hidden state

This is a distribution: "I believe Red is probably in DEFENSE state,
transitioning to RETREAT within ~10 steps, with 70% confidence."
```

**Architecture**:
```
GRU(input_size=5, hidden_size=16, num_layers=1)
→ takes obs_history[0:t] as sequence
→ outputs hidden state b(t) at each timestep
→ b(t) summarizes everything Blue has learned about Red so far
```

**Training**: supervised auxiliary task — predict Red's next FSM transition from obs_history. This forces b(t) to encode genuinely useful belief about Red's hidden state.

---

## Component 6: Deep CFR in Latent Space (NEW — replaces random search)
**File**: `NEAT.ipynb §10` *(to implement)*

This is the core of the revised approach. Instead of randomly sampling the latent space (QD search), we use counterfactual regret minimization to iteratively improve the strategy distribution over z.

### 6a. The Game-Theoretic Setup

```
Information set h = (obs_history[0:t], belief_state b(t))
                  = "everything Blue knows at time t"

Strategy π(z | h)  = distribution over latent tactics z
                   = "how likely am I to play each tactic given what I know?"

Counterfactual value V(z | h)
                   = "how well would I have done if I had played tactic z,
                      assuming I reach information set h?"
                   = estimated by the Regret Network
```

### 6b. Regret Network R

A neural network that estimates counterfactual regret for each tactic z given current belief state:

```
Input:   [z (8 dims),  b (16 dims)]  ←  candidate tactic + current belief
Output:  regret estimate r ∈ ℝ       ←  "how much would I regret not playing z?"

Architecture:
  concat(z, b)  [24]
       ↓  Linear(24 → 64) + ReLU
       ↓  Linear(64 → 64) + ReLU
       ↓  Linear(64 → 1)
  output: scalar regret
```

### 6c. CFR Update Loop

```
Initialize: strategy π = uniform over z-space
            regret_sum = zeros

For each CFR iteration T:
    1. Sample current belief state b from obs_history
    2. Sample K candidate tactics z₁...zₖ from latent space

    3. For each zᵢ:
       a. Decode zᵢ → trajectory via VAE decoder
       b. Run trajectory in sim (partial info: Blue doesn't see Red's hidden state)
       c. Compute actual value v(zᵢ | h)

    4. Compute counterfactual regret for each zᵢ:
       r(zᵢ) = v(zᵢ | h) - Σⱼ π(zⱼ | h) · v(zⱼ | h)
             = "how much better is zᵢ than what I was playing on average?"

    5. Update regret_sum(zᵢ) += max(r(zᵢ), 0)  (positive regret only)

    6. Update strategy via regret matching:
       π(zᵢ | h) = regret_sum(zᵢ) / Σⱼ regret_sum(zⱼ)
```

**Why regret matching works**: tactics you regret NOT playing get higher probability. Over many iterations, the strategy concentrates on the tactics that consistently perform better than your average. This converges to Nash equilibrium.

### 6d. What Makes This Different From QD Search

| Property | QD / Random Search | Deep CFR in Latent Space |
|---|---|---|
| Search mechanism | Random sampling | Regret-directed updating |
| Opponent model | None | Belief state b(t) encodes Red's hidden state |
| Convergence target | Coverage of archive | Nash equilibrium strategy |
| Information use | Fitness score only | Counterfactual: "what if I'd played z instead?" |
| Adaptivity | Static after training | Updates strategy based on what opponent does |

---

## Component 7: Evaluation
**File**: `NEAT.ipynb §11` *(to implement)*

Three measurements:

**Measurement 1: Strategy Convergence**
Does the CFR strategy distribution converge over iterations? Plot entropy of π(z) over CFR iterations — should decrease as strategy concentrates.

**Measurement 2: Exploitability**
How exploitable is the converged strategy? Train a best-response Red agent and measure how much it can beat the CFR-converged Blue strategy. Lower exploitability = closer to Nash.

**Measurement 3: Comparative Diversity (H1 connection)**
Compare behavioral coverage: what regions of tactic space does CFR visit vs. random QD search? CFR should visit fewer regions but with higher fitness — it is not trying to be diverse, it is trying to be optimal.

---

## Data Flow Summary

```
environment.py          →  WargameEnv (partial obs: Blue can't see Red's FSM state)
NEAT.ipynb §1–6         →  Trajectory corpus (diverse tactic behaviors)
NEAT.ipynb §7           →  training_data: [trajectory, obs_history, fitness]
NEAT.ipynb §8           →  VAE: trajectory[400] ↔ z[8]   (frozen after training)
NEAT.ipynb §9           →  Belief encoder: obs_history[t, 5] → b[16]
NEAT.ipynb §10          →  Deep CFR: (b, z) → regret → strategy π(z | b)
NEAT.ipynb §11          →  Exploitability + convergence + diversity comparison
```

---

## Implementation Order

| Step | What | Why first |
|---|---|---|
| 1 | Fix `environment.py` bugs | Sim must work |
| 2 | Make partial observations explicit | CFR requires Blue can't see Red's FSM state |
| 3 | `pip install torch numpy` | |
| 4 | §7 Trajectory + observation collection | |
| 5 | §8 VAE (unchanged from original) | Latent space must exist before CFR |
| 6 | §9 Belief state encoder (GRU) | Must encode uncertainty before CFR can use it |
| 7 | §10 Regret network + CFR loop | Core contribution |
| 8 | §11 Evaluation | Convergence + exploitability |

---

## Research Hypotheses (Revised)

| Hypothesis | Question | Measurement |
|---|---|---|
| H1 | Does CFR in latent space converge to a stable strategy? | Entropy of π(z) over iterations |
| H2 | Is the converged strategy less exploitable than NEAT? | Best-response Red agent performance |
| H3 | Does the belief state encoder actually reduce strategic uncertainty? | Belief state accuracy vs. Red's true FSM state |
| H4 | Does CFR visit qualitatively different regions of latent space than QD? | Tactic cluster comparison |

---

## Key Concepts Reference

**Information set**: All game states that look identical to Blue given its observations. Blue can't distinguish "Red is in LINE_FORM with 70% effectiveness" from "Red is in DEFENSE with 80% effectiveness" if unit counts happen to match.

**Counterfactual regret**: How much better Blue would have done at a specific information set if it had played tactic z — assuming it always reaches that information set. The "counterfactual" part means you compute the value of z as if you always arrived there, not just when you happened to.

**Nash equilibrium**: A strategy pair where neither player can improve by unilaterally changing their strategy. Blue's CFR-converged strategy, combined with Red's optimal response, forms a Nash equilibrium.

**Exploitability**: How much a best-response opponent can beat your strategy. Zero exploitability = Nash equilibrium. Used to measure CFR convergence without knowing the theoretical Nash.

**Belief state**: A probability distribution over Red's possible hidden states, conditioned on everything Blue has observed. "I think there's a 60% chance Red is in DEFENSE state based on how it's been firing."
