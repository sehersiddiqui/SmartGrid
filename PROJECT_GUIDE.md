# Project Guide and Viva Map

## Research objective
Develop a smart-building controller that forecasts near-term demand, uses evolutionary search for a day-ahead battery plan, and uses deep reinforcement learning to adjust operations as conditions change, while balancing cost, peak demand and occupant comfort.

## Modules

| Module | Responsibility | Key concepts |
|---|---|---|
| `01_data_acquisition` | Seeded synthetic data and optional UCI import | Data provenance, reproducibility |
| `02_data_preprocessing` | Cleaning, features, chronological split | Leakage prevention, time-series validation |
| `03_load_forecasting` | LSTM and persistence baseline | Backpropagation through time, MAE/RMSE |
| `04_solar_forecasting` | Solar prediction helper | Daylight shape, residual uncertainty |
| `05_energy_environment` | Gymnasium-compatible building simulator | MDP, transition dynamics, constraints |
| `06_genetic_optimizer` | DEAP GA schedule search | Population, chromosome, selection, crossover, mutation, fitness |
| `07_rl_agent` | PPO train/save/load wrapper | Policy, value function, reward, PPO clipping |
| `08_evaluation` | Comparable controller evaluation | Baselines, ablation, cost and comfort metrics |
| `09_dashboard` | Streamlit + Plotly | Explainable monitoring UI |
| `10_deployment` | SQLite and optional MQTT adapter | Persistence, telemetry interface |

## Why the three core algorithms are not interchangeable

- **LSTM predicts** what demand may look like based on history and context.
- **GA optimizes** a day-ahead battery schedule against a defined objective.
- **PPO controls** by selecting an action from the current state, allowing adaptation to disturbances.
- **Simulator evaluates** the result and returns a reward/next state. Without a simulator or real environment, RL has no meaningful interaction loop.

## MDP definition

`M = (S, A, P, R, γ)`

- State: normalized load, outdoor/indoor temperature, occupancy, solar, price, battery SOC, time features, and planned battery action.
- Action: continuous battery power, HVAC setpoint, EV charging power and lighting fraction.
- Transition: simulator equations, exogenous profiles and equipment constraints.
- Reward: negative operating cost, including comfort and peak penalties.
- Discount factor: PPO's `gamma` parameter.

## GA chromosome
A real-valued vector with one battery power command per planning step. Positive means charging; negative means discharging. The fitness simulates SOC forward and penalizes infeasible schedules, expensive grid use, peaks, and battery throughput. Elitism and tournament selection preserve good candidates; two-point crossover and Gaussian mutation create variation.

## Fuzzy logic
A small Mamdani-style rule layer uses triangular membership functions for temperature, occupancy and price. It returns an interpretable cooling-priority score; it is an auxiliary decision aid and not a replacement for the learned policy.

## Uncertainty
The forecasting notebook estimates empirical prediction intervals from held-out residual quantiles. These intervals are approximate and depend on residual representativeness; they are not formal Bayesian credible intervals. The GA can evaluate low/expected/high demand scenarios as a simple robust planning extension.

## Fair evaluation protocol
1. Use the exact same load, solar, price and weather trace for every controller.
2. Start every controller from the same initial SOC and indoor temperature.
3. Compare no-optimization, rule-based, GA-only, PPO-only (when trained), and GA+PPO.
4. Report cost, peak grid demand, energy, comfort degree-hours and battery throughput.
5. Repeat across multiple seeds; report mean and standard deviation.
6. Avoid claiming generalization from synthetic-only experiments.

## Suggested viva questions
- Why use chronological rather than random splitting?
- How can scaling on the full dataset leak future information?
- What is the difference between prediction and control?
- Why must the reward penalize discomfort?
- How do crossover and mutation affect exploration in GA?
- How does PPO's clipped objective limit large policy updates?
- Why is the no-optimization baseline essential?
- Which variables are measured and which are simulated?
- What are the limitations of a custom thermal simulator compared with EnergyPlus?

## Executable controller benchmark
`08_evaluation/controller_benchmark.py` runs matched-profile comparisons for no optimization, rule-based control, GA-only, PPO-only and GA+PPO. PPO-dependent rows are omitted if no trained policy is passed. The hybrid controller combines the GA battery plan with a conservative PPO battery adjustment while retaining PPO's HVAC/EV/lighting actions. Use the same profile and seed across methods; interpret results only after checking comfort and SOC constraints.
