# Tabular Q-Learning vs Deep Reinforcement Learning for Risk-Aware Portfolio Management

This project implements and compares a range of **tabular** and **deep reinforcement learning** algorithms for multi-asset portfolio allocation. All agents operate in a unified Gym-style trading environment with **transaction costs**, **drawdown penalties**, and consistent **walk-forward data splits**. The framework supports training, evaluation, stability studies, market-regime analysis, and exploration sensitivity experiments.

---

## 📌 Project Overview

The goal is to understand how different RL methods behave in realistic financial conditions.

We compare:
* **Tabular RL:** Q-Learning, Double-Q Learning, UCB-Q Learning
* **Deep RL:** DDPG, TD3, PPO
* **Baselines:** Equal-Weight, Momentum, Cash

All agents share:
* The same feature set and state representation
* The same reward shaping (cost-aware, drawdown-sensitive)
* The same Train / Validation / Test windows
* The same evaluation metrics (Return, Sharpe, MaxDD, Turnover)

The project includes:
* Walk-forward backtesting
* Multi-seed statistical stability analysis
* Transaction-cost sensitivity experiments
* Reward scaling & exploration analysis
* Bullish / Neutral / Bearish regime testing

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone [https://github.com/navyapriyanandi-tamu/DRL-Project.git](https://github.com/navyapriyanandi-tamu/DRL-Project.git)
```

### 2. Navigate into the project folder
```bash

cd DRL-Project
```
### 3. Create and activate a virtual environment
```bash

python3 -m venv venv
source venv/bin/activate
```
### 4. Install all dependencies

```bash
pip install -r requirements_modern.txt
```

Running Experiments
Train Tabular RL Agents

Bash

python train_q_learning.py
python train_double_q_learning.py
python train_ucb_q_learning.py
Train Deep RL Agents
Bash

python train_ddpg.py
python train_td3.py
python train_ppo.py
Run Full Multi-Agent Comparison
Bash

python compare_strategies.py
Market Benchmark Comparison
Bash

python plot_vs_market.py
Multi-Seed Stability Evaluation
Bash

python stability_study/stability_study_tabular.py
python stability_study/stability_study_deep_rl.py
Exploration / Reward Scaling Sensitivity
Bash

python exploration_analysis.py
Core RL Modules (Inside agents/)
q_learning.py, double_q_learning.py, ucb_q_learning.py: Tabular RL algorithms and exploration logic

baselines.py: Portfolio baselines: Equal-Weight, Cash, Momentum

discrete_actions.py: Maps continuous portfolio weights to a discrete rebalancing grid

state_discretizer.py: Converts raw market observations into discrete RL states

reward_shaper.py: Adds transaction costs and drawdown penalties into reward signals

sb3_wrapper.py: Unified wrapper for SB3 deep RL agents to match the tabular interface

Market Regime Testing
Synthetic datasets are included to test robustness across different environments:

poloniex_30m_test_dummy_bullish.hf

poloniex_30m_test_dummy_neutral.hf

poloniex_30m_test_dummy_bearish.hf

These allow evaluating how agents behave in up-trending, sideways, and down-trending markets.

License & Citation
This project is intended for academic and research use. If you use or extend this work, please cite the repository and the original Poloniex dataset source.
