"""
Stability Study - Tabular Agents (Q-Learning, Double-Q, UCB-Q)

stretch goal:
"Stability study: Compare standard and improved Tabular Q-learning (Double Q, UCB-ε)
against deep RL baselines across multiple random seeds, higher transaction costs (25–50 bps),
and varying market regimes."

This script trains tabular agents across:
- 3 transaction costs: 2.5 bps (baseline), 25 bps, 50 bps
- 3 random seeds: 42, 123, 456

"""

import numpy as np
import random
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from rl_portfolio_management.environments import PortfolioEnv
from rl_portfolio_management.wrappers import DiscreteActionWrapper, StateDiscretizer, RewardShaper
from rl_portfolio_management.agents import QLearningAgent, DoubleQLearningAgent, UCBQLearningAgent


# ============================================
# Stability Study Configuration
# ============================================
SEEDS = [42, 123, 456]

TRANSACTION_COSTS = {
    '2.5bps': 0.00025,   # Baseline (current)
    '25bps': 0.0025,     # 10x higher
    '50bps': 0.005,      # 20x higher
}

# Base config (same as original training)
BASE_CONFIG = {
    'delta': 0.05,
    'max_steps': 256,
    'window_length': 50,
    'val_split': 0.2,
    'n_episodes': 30000,
    'print_every': 5000,

    # Agent hyperparameters
    'learning_rate': 0.1,
    'lr_min': 0.01,
    'lr_decay': 0.9999,
    'discount_factor': 0.99,
    'epsilon': 0.9999,
    'epsilon_min': 0.03,
    'epsilon_decay': 0.99985,

    # Reward shaping
    'reward_scale': 100,
    'cost_penalty': 0.1,
    'drawdown_threshold': 0.15,
    'drawdown_penalty': 0.05,
}

# Output directories
OUTPUT_DIR = 'stability_study'
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


def create_env(df, config, random_reset=True):
    """Create wrapped environment for tabular agents."""
    env = PortfolioEnv(
        df=df,
        steps=config['max_steps'],
        trading_cost=config['trading_cost'],
        time_cost=0.0,
        window_length=config['window_length'],
        output_mode='EIIE',
        scale=True,
        random_reset=random_reset
    )
    env = DiscreteActionWrapper(env, delta=config['delta'])
    env = StateDiscretizer(env)
    env = RewardShaper(
        env,
        scale=config['reward_scale'],
        cost_penalty=config['cost_penalty'],
        drawdown_threshold=config['drawdown_threshold'],
        drawdown_penalty=config['drawdown_penalty']
    )
    return env


def evaluate_agent(agent, env, n_episodes=50):
    """Evaluate agent and return metrics."""
    portfolio_values = []
    all_actions = []

    for _ in range(n_episodes):
        state, info = env.reset()
        done = False
        pv = 1.0
        episode_actions = []

        while not done:
            action = agent.choose_action(state, training=False)
            episode_actions.append(action)
            state, reward, terminated, truncated, info = env.step(action)
            pv = info.get('portfolio_value', pv)
            done = terminated or truncated

        portfolio_values.append(pv)
        all_actions.append(episode_actions)

    portfolio_values = np.array(portfolio_values)

    # Compute metrics
    mean_return = (np.mean(portfolio_values) - 1) * 100
    std_return = np.std(portfolio_values) * 100

    # Sharpe (simplified)
    returns = portfolio_values - 1
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(48 * 365)

    # Max drawdown (worst case across episodes)
    max_dd = np.max(1 - np.min(portfolio_values) / 1.0)  # From starting value

    # Turnover (action changes)
    turnovers = []
    for actions in all_actions:
        if len(actions) > 1:
            changes = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i-1])
            turnovers.append(changes / (len(actions) - 1) * 100)
    avg_turnover = np.mean(turnovers) if turnovers else 0

    return {
        'return_mean': mean_return,
        'return_std': std_return,
        'sharpe': sharpe,
        'max_dd': max_dd * 100,
        'turnover': avg_turnover,
    }


def train_q_learning(df_train, df_test, config, seed):
    """Train Q-Learning agent."""
    set_seed(seed)

    env_train = create_env(df_train, config, random_reset=True)

    n_states = env_train.observation_space.n
    n_actions = 81

    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=config['learning_rate'],
        lr_min=config['lr_min'],
        lr_decay=config['lr_decay'],
        discount_factor=config['discount_factor'],
        epsilon=config['epsilon'],
        epsilon_min=config['epsilon_min'],
        epsilon_decay=config['epsilon_decay']
    )

    # Training loop
    for episode in range(1, config['n_episodes'] + 1):
        state, info = env_train.reset()

        for step in range(config['max_steps']):
            action = agent.choose_action(state, training=True)
            next_state, reward, terminated, truncated, info = env_train.step(action)
            agent.learn(state, action, reward, next_state, terminated or truncated)
            state = next_state

            if terminated or truncated:
                break

        agent.decay_epsilon()
        agent.decay_lr()

        if episode % config['print_every'] == 0:
            print(f"    Episode {episode}/{config['n_episodes']}")

    # Evaluate on test (with same transaction cost)
    env_test = create_env(df_test, config, random_reset=True)
    test_metrics = evaluate_agent(agent, env_test, n_episodes=50)

    return agent, test_metrics


def train_double_q_learning(df_train, df_test, config, seed):
    """Train Double Q-Learning agent."""
    set_seed(seed)

    env_train = create_env(df_train, config, random_reset=True)

    n_states = env_train.observation_space.n
    n_actions = 81

    agent = DoubleQLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=config['learning_rate'],
        lr_min=config['lr_min'],
        lr_decay=config['lr_decay'],
        discount_factor=config['discount_factor'],
        epsilon=config['epsilon'],
        epsilon_min=config['epsilon_min'],
        epsilon_decay=config['epsilon_decay']
    )

    # Training loop
    for episode in range(1, config['n_episodes'] + 1):
        state, info = env_train.reset()

        for step in range(config['max_steps']):
            action = agent.choose_action(state, training=True)
            next_state, reward, terminated, truncated, info = env_train.step(action)
            agent.learn(state, action, reward, next_state, terminated or truncated)
            state = next_state

            if terminated or truncated:
                break

        agent.decay_epsilon()
        agent.decay_lr()

        if episode % config['print_every'] == 0:
            print(f"    Episode {episode}/{config['n_episodes']}")

    # Evaluate on test (with same transaction cost)
    env_test = create_env(df_test, config, random_reset=True)
    test_metrics = evaluate_agent(agent, env_test, n_episodes=50)

    return agent, test_metrics


def train_ucb_q_learning(df_train, df_test, config, seed):
    """Train UCB-epsilon Q-Learning agent."""
    set_seed(seed)

    env_train = create_env(df_train, config, random_reset=True)

    n_states = env_train.observation_space.n
    n_actions = 81

    agent = UCBQLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=config['learning_rate'],
        lr_min=config['lr_min'],
        lr_decay=config['lr_decay'],
        discount_factor=config['discount_factor'],
        epsilon=config['epsilon'],
        epsilon_min=config['epsilon_min'],
        epsilon_decay=config['epsilon_decay'],
        ucb_c=2.0
    )

    # Training loop
    for episode in range(1, config['n_episodes'] + 1):
        state, info = env_train.reset()

        for step in range(config['max_steps']):
            action = agent.choose_action(state, training=True)
            next_state, reward, terminated, truncated, info = env_train.step(action)
            agent.learn(state, action, reward, next_state, terminated or truncated)
            state = next_state

            if terminated or truncated:
                break

        agent.decay_epsilon()
        agent.decay_lr()

        if episode % config['print_every'] == 0:
            print(f"    Episode {episode}/{config['n_episodes']}")

    # Evaluate on test (with same transaction cost)
    env_test = create_env(df_test, config, random_reset=True)
    test_metrics = evaluate_agent(agent, env_test, n_episodes=50)

    return agent, test_metrics


AGENT_TRAINERS = {
    'Q-Learning': train_q_learning,
    'Double-Q': train_double_q_learning,
    'UCB-Q': train_ucb_q_learning,
}


def main():
    print("=" * 70)
    print("STABILITY STUDY - TABULAR AGENTS")
    print("=" * 70)
    print(f"Agents: {list(AGENT_TRAINERS.keys())}")
    print(f"Transaction costs: {list(TRANSACTION_COSTS.keys())}")
    print(f"Seeds: {SEEDS}")
    print(f"Total runs: {len(AGENT_TRAINERS) * len(TRANSACTION_COSTS) * len(SEEDS)}")
    print(f"\nEvaluation: Train with cost X → Test with cost X (Option A)")
    print("=" * 70)

    # Create output directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'poloniex_30m.hf')
    df_train_full = pd.read_hdf(data_path, key='train')
    df_test = pd.read_hdf(data_path, key='test')

    # Split train into train/val (use only train portion)
    val_split_idx = int(len(df_train_full) * (1 - BASE_CONFIG['val_split']))
    df_train = df_train_full.iloc[:val_split_idx]

    print(f"\nData splits:")
    print(f"  Train: {len(df_train)} periods (~{len(df_train)//48} days)")
    print(f"  Test:  {len(df_test)} periods (~{len(df_test)//48} days)")

    # Results storage
    all_results = []

    # Run all combinations
    total_runs = len(AGENT_TRAINERS) * len(TRANSACTION_COSTS) * len(SEEDS)
    run_count = 0
    start_time = time.time()

    for cost_name, cost_value in TRANSACTION_COSTS.items():
        for seed in SEEDS:
            for agent_name, trainer_fn in AGENT_TRAINERS.items():
                run_count += 1

                print(f"\n{'='*70}")
                print(f"RUN {run_count}/{total_runs}: {agent_name} | Cost: {cost_name} | Seed: {seed}")
                print(f"{'='*70}")

                # Create config for this run
                config = BASE_CONFIG.copy()
                config['trading_cost'] = cost_value
                config['seed'] = seed

                # Train and evaluate
                run_start = time.time()
                agent, test_metrics = trainer_fn(df_train, df_test, config, seed)
                run_time = time.time() - run_start

                # Save model
                model_name = f"{agent_name.lower().replace('-', '_')}_cost{cost_name}_seed{seed}"
                model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
                agent.save(model_path)

                # Store results
                result = {
                    'agent': agent_name,
                    'cost_bps': cost_name,
                    'cost_value': cost_value,
                    'seed': seed,
                    'test_return': test_metrics['return_mean'],
                    'test_return_std': test_metrics['return_std'],
                    'test_sharpe': test_metrics['sharpe'],
                    'test_max_dd': test_metrics['max_dd'],
                    'test_turnover': test_metrics['turnover'],
                    'train_time_sec': run_time,
                    'model_path': model_path,
                }
                all_results.append(result)

                print(f"\n  Test Results (cost={cost_name}):")
                print(f"    Return:   {test_metrics['return_mean']:+.2f}% ± {test_metrics['return_std']:.2f}%")
                print(f"    Sharpe:   {test_metrics['sharpe']:.3f}")
                print(f"    Max DD:   {test_metrics['max_dd']:.2f}%")
                print(f"    Turnover: {test_metrics['turnover']:.1f}%")
                print(f"    Time:     {run_time/60:.1f} min")

                # Save intermediate results (in case of crash)
                results_df = pd.DataFrame(all_results)
                results_df.to_csv(os.path.join(RESULTS_DIR, 'stability_tabular_results.csv'), index=False)

    # Final summary
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("STABILITY STUDY COMPLETE - TABULAR AGENTS")
    print("=" * 70)
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Results saved to: {os.path.join(RESULTS_DIR, 'stability_tabular_results.csv')}")
    print(f"Models saved to: {MODELS_DIR}")

    # Print summary tables
    results_df = pd.DataFrame(all_results)

    print("\n" + "-" * 70)
    print("MEAN TEST RETURN (%) BY AGENT AND COST")
    print("-" * 70)
    pivot_mean = results_df.pivot_table(
        values='test_return',
        index='agent',
        columns='cost_bps',
        aggfunc='mean'
    )
    # Reorder columns
    pivot_mean = pivot_mean[['2.5bps', '25bps', '50bps']]
    print(pivot_mean.round(2).to_string())

    print("\n" + "-" * 70)
    print("STD TEST RETURN (%) ACROSS SEEDS")
    print("-" * 70)
    pivot_std = results_df.pivot_table(
        values='test_return',
        index='agent',
        columns='cost_bps',
        aggfunc='std'
    )
    pivot_std = pivot_std[['2.5bps', '25bps', '50bps']]
    print(pivot_std.round(2).to_string())

    print("\n" + "-" * 70)
    print("MEAN SHARPE RATIO BY AGENT AND COST")
    print("-" * 70)
    pivot_sharpe = results_df.pivot_table(
        values='test_sharpe',
        index='agent',
        columns='cost_bps',
        aggfunc='mean'
    )
    pivot_sharpe = pivot_sharpe[['2.5bps', '25bps', '50bps']]
    print(pivot_sharpe.round(3).to_string())

    return results_df


if __name__ == '__main__':
    results = main()
