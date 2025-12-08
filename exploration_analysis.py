"""
Exploration Analysis - Sensitivity of UCB-ε and Double Q-Learning

stretch goal:
"Exploration analysis: study sensitivity of UCB-ε and Double Q-learning to transaction
costs and reward scaling."

This script trains UCB-Q and Double-Q across:
- 4 reward scales: 10, 50, 100, 200
- 3 transaction costs: 2.5 bps, 25 bps, 50 bps
- 1 seed: 42 (fixed for sensitivity analysis)

"""

import numpy as np
import random
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rl_portfolio_management.environments import PortfolioEnv
from rl_portfolio_management.wrappers import DiscreteActionWrapper, StateDiscretizer, RewardShaper
from rl_portfolio_management.agents import DoubleQLearningAgent, UCBQLearningAgent


# ============================================
# Exploration Analysis Configuration
# ============================================
SEED = 42  # Fixed seed for sensitivity analysis

REWARD_SCALES = [10, 50, 100, 200]

TRANSACTION_COSTS = {
    '2.5bps': 0.00025,
    '25bps': 0.0025,
    '50bps': 0.005,
}

# Base config
BASE_CONFIG = {
    'delta': 0.05,
    'max_steps': 256,
    'window_length': 50,
    'val_split': 0.2,
    'n_episodes': 30000,
    'print_every': 10000,

    # Agent hyperparameters
    'learning_rate': 0.1,
    'lr_min': 0.01,
    'lr_decay': 0.9999,
    'discount_factor': 0.99,
    'epsilon': 0.9999,
    'epsilon_min': 0.03,
    'epsilon_decay': 0.99985,

    # Reward shaping (reward_scale will be varied)
    'cost_penalty': 0.1,
    'drawdown_threshold': 0.15,
    'drawdown_penalty': 0.05,
}

# Output directories
OUTPUT_DIR = 'exploration_analysis'
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')


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

    # Sharpe
    returns = portfolio_values - 1
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(48 * 365)

    # Max drawdown
    max_dd = np.max(1 - np.min(portfolio_values) / 1.0)

    # Turnover
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


def train_double_q_learning(df_train, df_test, config):
    """Train Double Q-Learning agent."""
    set_seed(SEED)

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

    # Evaluate on test
    env_test = create_env(df_test, config, random_reset=True)
    test_metrics = evaluate_agent(agent, env_test, n_episodes=50)

    return agent, test_metrics


def train_ucb_q_learning(df_train, df_test, config):
    """Train UCB-epsilon Q-Learning agent."""
    set_seed(SEED)

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

    # Evaluate on test
    env_test = create_env(df_test, config, random_reset=True)
    test_metrics = evaluate_agent(agent, env_test, n_episodes=50)

    return agent, test_metrics


AGENT_TRAINERS = {
    'Double-Q': train_double_q_learning,
    'UCB-Q': train_ucb_q_learning,
}


def generate_plots(results_df):
    """Generate sensitivity analysis plots."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Plot 1: Heatmaps for each agent
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Exploration Analysis: Sensitivity to Reward Scale and Transaction Cost', fontsize=14)

    for idx, agent_name in enumerate(['Double-Q', 'UCB-Q']):
        ax = axes[idx]
        agent_df = results_df[results_df['agent'] == agent_name]

        pivot = agent_df.pivot_table(
            values='test_return',
            index='reward_scale',
            columns='cost_bps',
            aggfunc='mean'
        )

        # Reorder columns
        col_order = ['2.5bps', '25bps', '50bps']
        pivot = pivot[[c for c in col_order if c in pivot.columns]]

        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax,
                    cbar_kws={'label': 'Return (%)'})
        ax.set_title(f'{agent_name}: Return vs (Scale, Cost)')
        ax.set_xlabel('Transaction Cost')
        ax.set_ylabel('Reward Scale')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'sensitivity_heatmaps.png'), dpi=150)
    print(f"Saved: {os.path.join(PLOTS_DIR, 'sensitivity_heatmaps.png')}")
    plt.show()

    # Plot 2: Line plots - Return vs Reward Scale at each cost level
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Return vs Reward Scale at Different Transaction Costs', fontsize=14)

    colors = {'2.5bps': 'green', '25bps': 'orange', '50bps': 'red'}
    markers = {'2.5bps': 'o', '25bps': 's', '50bps': '^'}

    for idx, agent_name in enumerate(['Double-Q', 'UCB-Q']):
        ax = axes[idx]
        agent_df = results_df[results_df['agent'] == agent_name]

        for cost_name in ['2.5bps', '25bps', '50bps']:
            cost_df = agent_df[agent_df['cost_bps'] == cost_name]
            cost_df = cost_df.sort_values('reward_scale')

            ax.plot(cost_df['reward_scale'], cost_df['test_return'],
                    label=cost_name, color=colors[cost_name],
                    marker=markers[cost_name], linewidth=2, markersize=8)

        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Reward Scale')
        ax.set_ylabel('Test Return (%)')
        ax.set_title(f'{agent_name}')
        ax.legend(title='Cost')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(REWARD_SCALES)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'sensitivity_lines.png'), dpi=150)
    print(f"Saved: {os.path.join(PLOTS_DIR, 'sensitivity_lines.png')}")
    plt.show()

    # Plot 3: Sharpe ratio heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Sharpe Ratio Sensitivity to Reward Scale and Transaction Cost', fontsize=14)

    for idx, agent_name in enumerate(['Double-Q', 'UCB-Q']):
        ax = axes[idx]
        agent_df = results_df[results_df['agent'] == agent_name]

        pivot = agent_df.pivot_table(
            values='test_sharpe',
            index='reward_scale',
            columns='cost_bps',
            aggfunc='mean'
        )

        col_order = ['2.5bps', '25bps', '50bps']
        pivot = pivot[[c for c in col_order if c in pivot.columns]]

        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax,
                    cbar_kws={'label': 'Sharpe'})
        ax.set_title(f'{agent_name}: Sharpe vs (Scale, Cost)')
        ax.set_xlabel('Transaction Cost')
        ax.set_ylabel('Reward Scale')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'sensitivity_sharpe_heatmaps.png'), dpi=150)
    print(f"Saved: {os.path.join(PLOTS_DIR, 'sensitivity_sharpe_heatmaps.png')}")
    plt.show()

    # Plot 4: Turnover heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Turnover Sensitivity to Reward Scale and Transaction Cost', fontsize=14)

    for idx, agent_name in enumerate(['Double-Q', 'UCB-Q']):
        ax = axes[idx]
        agent_df = results_df[results_df['agent'] == agent_name]

        pivot = agent_df.pivot_table(
            values='test_turnover',
            index='reward_scale',
            columns='cost_bps',
            aggfunc='mean'
        )

        col_order = ['2.5bps', '25bps', '50bps']
        pivot = pivot[[c for c in col_order if c in pivot.columns]]

        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                    cbar_kws={'label': 'Turnover (%)'})
        ax.set_title(f'{agent_name}: Turnover vs (Scale, Cost)')
        ax.set_xlabel('Transaction Cost')
        ax.set_ylabel('Reward Scale')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'sensitivity_turnover_heatmaps.png'), dpi=150)
    print(f"Saved: {os.path.join(PLOTS_DIR, 'sensitivity_turnover_heatmaps.png')}")
    plt.show()


def main():
    print("=" * 70)
    print("EXPLORATION ANALYSIS - SENSITIVITY STUDY")
    print("=" * 70)
    print(f"Agents: {list(AGENT_TRAINERS.keys())}")
    print(f"Reward scales: {REWARD_SCALES}")
    print(f"Transaction costs: {list(TRANSACTION_COSTS.keys())}")
    print(f"Seed: {SEED} (fixed)")
    print(f"Total runs: {len(AGENT_TRAINERS) * len(REWARD_SCALES) * len(TRANSACTION_COSTS)}")
    print("=" * 70)

    # Create output directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

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
    total_runs = len(AGENT_TRAINERS) * len(REWARD_SCALES) * len(TRANSACTION_COSTS)
    run_count = 0
    start_time = time.time()

    for reward_scale in REWARD_SCALES:
        for cost_name, cost_value in TRANSACTION_COSTS.items():
            for agent_name, trainer_fn in AGENT_TRAINERS.items():
                run_count += 1

                print(f"\n{'='*70}")
                print(f"RUN {run_count}/{total_runs}: {agent_name} | Scale: {reward_scale} | Cost: {cost_name}")
                print(f"{'='*70}")

                # Create config for this run
                config = BASE_CONFIG.copy()
                config['trading_cost'] = cost_value
                config['reward_scale'] = reward_scale

                # Train and evaluate
                run_start = time.time()
                agent, test_metrics = trainer_fn(df_train, df_test, config)
                run_time = time.time() - run_start

                # Save model
                model_name = f"{agent_name.lower().replace('-', '_')}_scale{reward_scale}_cost{cost_name}"
                model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
                agent.save(model_path)

                # Store results
                result = {
                    'agent': agent_name,
                    'reward_scale': reward_scale,
                    'cost_bps': cost_name,
                    'cost_value': cost_value,
                    'test_return': test_metrics['return_mean'],
                    'test_return_std': test_metrics['return_std'],
                    'test_sharpe': test_metrics['sharpe'],
                    'test_max_dd': test_metrics['max_dd'],
                    'test_turnover': test_metrics['turnover'],
                    'train_time_sec': run_time,
                    'model_path': model_path,
                }
                all_results.append(result)

                print(f"\n  Test Results:")
                print(f"    Return:   {test_metrics['return_mean']:+.2f}% ± {test_metrics['return_std']:.2f}%")
                print(f"    Sharpe:   {test_metrics['sharpe']:.3f}")
                print(f"    Max DD:   {test_metrics['max_dd']:.2f}%")
                print(f"    Turnover: {test_metrics['turnover']:.1f}%")
                print(f"    Time:     {run_time/60:.1f} min")

                # Save intermediate results
                results_df = pd.DataFrame(all_results)
                results_df.to_csv(os.path.join(RESULTS_DIR, 'exploration_results.csv'), index=False)

    # Final summary
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("EXPLORATION ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Results saved to: {os.path.join(RESULTS_DIR, 'exploration_results.csv')}")

    # Print summary tables
    results_df = pd.DataFrame(all_results)

    print("\n" + "-" * 70)
    print("DOUBLE-Q: MEAN TEST RETURN (%) BY SCALE AND COST")
    print("-" * 70)
    dq_df = results_df[results_df['agent'] == 'Double-Q']
    pivot_dq = dq_df.pivot_table(values='test_return', index='reward_scale', columns='cost_bps')
    pivot_dq = pivot_dq[['2.5bps', '25bps', '50bps']]
    print(pivot_dq.round(2).to_string())

    print("\n" + "-" * 70)
    print("UCB-Q: MEAN TEST RETURN (%) BY SCALE AND COST")
    print("-" * 70)
    ucb_df = results_df[results_df['agent'] == 'UCB-Q']
    pivot_ucb = ucb_df.pivot_table(values='test_return', index='reward_scale', columns='cost_bps')
    pivot_ucb = pivot_ucb[['2.5bps', '25bps', '50bps']]
    print(pivot_ucb.round(2).to_string())

    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    generate_plots(results_df)

    return results_df


if __name__ == '__main__':
    results = main()
