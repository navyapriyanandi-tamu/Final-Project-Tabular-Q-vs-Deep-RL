Tabular Q-Learning vs Deep Reinforcement Learning for Risk-Aware Portfolio Management

This project implements and compares a range of tabular and deep reinforcement learning algorithms for multi-asset portfolio allocation. All agents operate in a unified Gym-style trading environment that incorporates transaction costs, drawdown penalties, and walk-forward data splits. The framework supports training, evaluation, stability studies, exploration sensitivity analyses, and regime-based stress testing.

Project Overview

The goal of this project is to study how RL methods—both tabular and deep—behave in realistic financial environments. The framework compares:

Tabular RL: Q-Learning, Double-Q, UCB-Q

Deep RL: DDPG, TD3, PPO

Financial baselines: Equal-Weight, Momentum, Cash

All agents share identical state/features, reward shaping, transaction costs, and evaluation methodology. Experiments include walk-forward backtesting, multi-seed stability analysis, transaction-cost sensitivity, exploration analysis, and performance under synthetic bullish/neutral/bearish market regimes.


Installation
1. Clone the repository
git clone https://github.com/navyapriyanandi-tamu/DRL-Project.git

2. Navigate into the project folder
cd DRL-Project

3. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

4. Install dependencies
pip install -r requirements_modern.txt


Your environment is now ready to run training and evaluation scripts.

Running Experiments
Train Tabular RL Agents
python train_q_learning.py
python train_double_q_learning.py
python train_ucb_q_learning.py

Train Deep RL Agents
python train_ddpg.py
python train_td3.py
python train_ppo.py

Run Full Strategy Comparison
python compare_strategies.py

Compare RL Agents to Market Assets
python plot_vs_market.py

Perform Multi-Seed Stability Tests
python stability_study/stability_study_tabular.py
python stability_study/stability_study_deep_rl.py

Exploration / Reward Scaling Sensitivity
python exploration_analysis.py

Core Modules (agents/)

q_learning.py, double_q_learning.py, ucb_q_learning.py – Tabular RL implementations

baselines.py – Static benchmarks (Cash, Equal-Weight, Momentum)

discrete_actions.py – Discretized action grid for tabular RL

state_discretizer.py – Converts raw market observations into discrete states

reward_shaper.py – Applies transaction costs and drawdown penalties

sb3_wrapper.py – Wraps SB3 deep RL policies into the project environment

These ensure consistent behavior and evaluation across tabular and deep RL agents.

Market Regime Datasets (data/)

*_bullish.hf, *_neutral.hf, *_bearish.hf
Used for testing robustness across different synthetic market conditions.
