"""
Stability Study - Deep RL Agents (DDPG, TD3, PPO)

"Stability study: Compare standard and improved Tabular Q-learning (Double Q, UCB-ε)
against deep RL baselines across multiple random seeds, higher transaction costs (25–50 bps),
and varying market regimes."

This script trains deep RL agents across:
- 3 transaction costs: 2.5 bps (baseline), 25 bps, 50 bps
- 3 random seeds: 42, 123, 456

Usage:

    python stability_study_deep_rl.py --agent DDPG
    python stability_study_deep_rl.py --agent TD3
    python stability_study_deep_rl.py --agent PPO
"""

import numpy as np
import random
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from stable_baselines3 import DDPG, TD3, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from rl_portfolio_management.wrappers import create_sb3_env


# ============================================
# Stability Study Configuration
# ============================================
SEEDS = [42, 123, 456]

TRANSACTION_COSTS = {
    '2.5bps': 0.00025,   # Baseline (current)
    '25bps': 0.0025,     # 10x higher
    '50bps': 0.005,      # 20x higher
}

# Base config for deep RL (same as original training scripts)
BASE_CONFIG = {
    'max_steps': 256,
    'window_length': 50,
    'val_split': 0.2,

    # Training
    'total_timesteps': 1000000,
    'print_every': 100000,

    # Reward shaping (identical across all agents)
    'reward_scale': 100,
    'cost_penalty': 0.1,
    'drawdown_threshold': 0.15,
    'drawdown_penalty': 0.05,
}

# Agent-specific hyperparameters
DDPG_CONFIG = {
    'learning_rate': 1e-4,
    'buffer_size': 100000,
    'learning_starts': 10000,
    'batch_size': 256,
    'tau': 0.005,
    'gamma': 0.99,
    'train_freq': 1,
    'gradient_steps': 1,
    'action_noise_std': 0.1,
    'net_arch': [256, 256],
}

TD3_CONFIG = {
    'learning_rate': 1e-4,
    'buffer_size': 100000,
    'learning_starts': 10000,
    'batch_size': 256,
    'tau': 0.005,
    'gamma': 0.99,
    'train_freq': 1,
    'gradient_steps': 1,
    'action_noise_std': 0.1,
    'policy_delay': 2,
    'target_policy_noise': 0.2,
    'target_noise_clip': 0.5,
    'net_arch': [256, 256],
}

PPO_CONFIG = {
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'net_arch': dict(pi=[256, 256], vf=[256, 256]),
}

# Output directories
OUTPUT_DIR = 'stability_study'
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


def evaluate_agent(model, env, n_episodes=50):
    """Evaluate trained agent on environment."""
    portfolio_values = []
    all_actions = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        pv = 1.0
        episode_actions = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            episode_actions.append(action.tolist() if hasattr(action, 'tolist') else action)
            obs, reward, terminated, truncated, info = env.step(action)
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

    # Turnover (for continuous actions)
    turnovers = []
    for actions in all_actions:
        if len(actions) > 1:
            total_turnover = 0
            for i in range(1, len(actions)):
                prev = np.array(actions[i-1])
                curr = np.array(actions[i])
                total_turnover += np.sum(np.abs(curr - prev))
            turnovers.append(total_turnover / (len(actions) - 1) * 100)
    avg_turnover = np.mean(turnovers) if turnovers else 0

    return {
        'return_mean': mean_return,
        'return_std': std_return,
        'sharpe': sharpe,
        'max_dd': max_dd * 100,
        'turnover': avg_turnover,
    }


def train_ddpg(df_train, df_test, config, seed):
    """Train DDPG agent."""
    set_seed(seed)

    env_train = create_sb3_env(df_train, config, random_reset=True)
    n_actions = env_train.action_space.shape[0]

    # Action noise
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=DDPG_CONFIG['action_noise_std'] * np.ones(n_actions)
    )

    model = DDPG(
        policy="MlpPolicy",
        env=env_train,
        learning_rate=DDPG_CONFIG['learning_rate'],
        buffer_size=DDPG_CONFIG['buffer_size'],
        learning_starts=DDPG_CONFIG['learning_starts'],
        batch_size=DDPG_CONFIG['batch_size'],
        tau=DDPG_CONFIG['tau'],
        gamma=DDPG_CONFIG['gamma'],
        train_freq=DDPG_CONFIG['train_freq'],
        gradient_steps=DDPG_CONFIG['gradient_steps'],
        action_noise=action_noise,
        policy_kwargs=dict(net_arch=DDPG_CONFIG['net_arch']),
        verbose=0,
        seed=seed
    )

    # Train with progress updates
    timesteps_done = 0
    while timesteps_done < config['total_timesteps']:
        model.learn(total_timesteps=config['print_every'], reset_num_timesteps=False)
        timesteps_done += config['print_every']
        print(f"    Timestep: {timesteps_done}/{config['total_timesteps']}")

    # Evaluate on test (with same transaction cost)
    env_test = create_sb3_env(df_test, config, random_reset=True)
    test_metrics = evaluate_agent(model, env_test, n_episodes=50)

    return model, test_metrics


def train_td3(df_train, df_test, config, seed):
    """Train TD3 agent."""
    set_seed(seed)

    env_train = create_sb3_env(df_train, config, random_reset=True)
    n_actions = env_train.action_space.shape[0]

    # Action noise
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=TD3_CONFIG['action_noise_std'] * np.ones(n_actions)
    )

    model = TD3(
        policy="MlpPolicy",
        env=env_train,
        learning_rate=TD3_CONFIG['learning_rate'],
        buffer_size=TD3_CONFIG['buffer_size'],
        learning_starts=TD3_CONFIG['learning_starts'],
        batch_size=TD3_CONFIG['batch_size'],
        tau=TD3_CONFIG['tau'],
        gamma=TD3_CONFIG['gamma'],
        train_freq=TD3_CONFIG['train_freq'],
        gradient_steps=TD3_CONFIG['gradient_steps'],
        action_noise=action_noise,
        policy_delay=TD3_CONFIG['policy_delay'],
        target_policy_noise=TD3_CONFIG['target_policy_noise'],
        target_noise_clip=TD3_CONFIG['target_noise_clip'],
        policy_kwargs=dict(net_arch=TD3_CONFIG['net_arch']),
        verbose=0,
        seed=seed
    )

    # Train with progress updates
    timesteps_done = 0
    while timesteps_done < config['total_timesteps']:
        model.learn(total_timesteps=config['print_every'], reset_num_timesteps=False)
        timesteps_done += config['print_every']
        print(f"    Timestep: {timesteps_done}/{config['total_timesteps']}")

    # Evaluate on test (with same transaction cost)
    env_test = create_sb3_env(df_test, config, random_reset=True)
    test_metrics = evaluate_agent(model, env_test, n_episodes=50)

    return model, test_metrics


def train_ppo(df_train, df_test, config, seed):
    """Train PPO agent."""
    set_seed(seed)

    env_train = create_sb3_env(df_train, config, random_reset=True)

    model = PPO(
        policy="MlpPolicy",
        env=env_train,
        learning_rate=PPO_CONFIG['learning_rate'],
        n_steps=PPO_CONFIG['n_steps'],
        batch_size=PPO_CONFIG['batch_size'],
        n_epochs=PPO_CONFIG['n_epochs'],
        gamma=PPO_CONFIG['gamma'],
        gae_lambda=PPO_CONFIG['gae_lambda'],
        clip_range=PPO_CONFIG['clip_range'],
        ent_coef=PPO_CONFIG['ent_coef'],
        vf_coef=PPO_CONFIG['vf_coef'],
        max_grad_norm=PPO_CONFIG['max_grad_norm'],
        policy_kwargs=dict(net_arch=PPO_CONFIG['net_arch']),
        verbose=0,
        seed=seed
    )

    # Train with progress updates
    timesteps_done = 0
    while timesteps_done < config['total_timesteps']:
        model.learn(total_timesteps=config['print_every'], reset_num_timesteps=False)
        timesteps_done += config['print_every']
        print(f"    Timestep: {timesteps_done}/{config['total_timesteps']}")

    # Evaluate on test (with same transaction cost)
    env_test = create_sb3_env(df_test, config, random_reset=True)
    test_metrics = evaluate_agent(model, env_test, n_episodes=50)

    return model, test_metrics


AGENT_TRAINERS = {
    'DDPG': train_ddpg,
    'TD3': train_td3,
    'PPO': train_ppo,
}


def run_study(agents_to_run=None):
    """Run stability study for specified agents."""
    if agents_to_run is None:
        agents_to_run = list(AGENT_TRAINERS.keys())

    print("=" * 70)
    print("STABILITY STUDY - DEEP RL AGENTS")
    print("=" * 70)
    print(f"Agents: {agents_to_run}")
    print(f"Transaction costs: {list(TRANSACTION_COSTS.keys())}")
    print(f"Seeds: {SEEDS}")
    print(f"Total runs: {len(agents_to_run) * len(TRANSACTION_COSTS) * len(SEEDS)}")
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

    # Check for existing results to resume
    results_file = os.path.join(RESULTS_DIR, 'stability_deep_rl_results.csv')
    completed_runs = set()
    if os.path.exists(results_file):
        existing_df = pd.read_csv(results_file)
        for _, row in existing_df.iterrows():
            completed_runs.add((row['agent'], row['cost_bps'], row['seed']))
            all_results.append(row.to_dict())
        print(f"\nResuming: Found {len(completed_runs)} completed runs")

    # Run all combinations
    total_runs = len(agents_to_run) * len(TRANSACTION_COSTS) * len(SEEDS)
    run_count = len(completed_runs)
    start_time = time.time()

    for cost_name, cost_value in TRANSACTION_COSTS.items():
        for seed in SEEDS:
            for agent_name in agents_to_run:
                # Skip if already completed
                if (agent_name, cost_name, seed) in completed_runs:
                    print(f"\nSkipping (already done): {agent_name} | Cost: {cost_name} | Seed: {seed}")
                    continue

                trainer_fn = AGENT_TRAINERS[agent_name]
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
                model, test_metrics = trainer_fn(df_train, df_test, config, seed)
                run_time = time.time() - run_start

                # Save model
                model_name = f"{agent_name.lower()}_cost{cost_name}_seed{seed}"
                model_path = os.path.join(MODELS_DIR, model_name)
                model.save(model_path)

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
                results_df.to_csv(results_file, index=False)

    # Final summary
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("STABILITY STUDY COMPLETE - DEEP RL AGENTS")
    print("=" * 70)
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Results saved to: {results_file}")
    print(f"Models saved to: {MODELS_DIR}")

    # Print summary tables
    results_df = pd.DataFrame(all_results)

    # Filter to only the agents we ran
    results_df = results_df[results_df['agent'].isin(agents_to_run)]

    if len(results_df) > 0:
        print("\n" + "-" * 70)
        print("MEAN TEST RETURN (%) BY AGENT AND COST")
        print("-" * 70)
        pivot_mean = results_df.pivot_table(
            values='test_return',
            index='agent',
            columns='cost_bps',
            aggfunc='mean'
        )
        if '2.5bps' in pivot_mean.columns:
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
        if '2.5bps' in pivot_std.columns:
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
        if '2.5bps' in pivot_sharpe.columns:
            pivot_sharpe = pivot_sharpe[['2.5bps', '25bps', '50bps']]
        print(pivot_sharpe.round(3).to_string())

    return results_df


def main():
    parser = argparse.ArgumentParser(description='Stability Study - Deep RL Agents')
    parser.add_argument('--agent', type=str, choices=['DDPG', 'TD3', 'PPO', 'all'],
                        default='all', help='Which agent to train (default: all)')
    args = parser.parse_args()

    if args.agent == 'all':
        agents = None  # Run all
    else:
        agents = [args.agent]

    return run_study(agents_to_run=agents)


if __name__ == '__main__':
    results = main()
