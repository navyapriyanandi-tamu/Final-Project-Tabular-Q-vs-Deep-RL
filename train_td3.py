"""
Train TD3 Agent for Portfolio Management using Stable-Baselines3.

- "TD3 and PPO agents: Implemented via Stable-Baselines3 using identical
   rewards, splits, and metrics to compare continuous-action baselines."
- "Continuous version (for DDPG/TD3/PPO): target weight vector in a Box space,
   projected to the simplex."

TD3 (Twin Delayed DDPG) improves on DDPG with:
1. Twin critics (two Q-networks) - reduces overestimation bias
2. Delayed policy updates - more stable learning
3. Target policy smoothing - adds noise to target actions

Configuration matches DDPG and tabular agents for fair comparison.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback

from rl_portfolio_management.wrappers import create_sb3_env


# ============================================
# Set Random Seed for Reproducibility
# ============================================
SEED = 42

def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)

set_seed(SEED)


# ============================================
# Configuration - SAME AS DDPG/TABULAR FOR FAIR COMPARISON
# ============================================
CONFIG = {
    # Environment (identical to DDPG and tabular Q-learning)
    'trading_cost': 0.00025,  # 0.025% = 2.5 bps
    'max_steps': 256,
    'window_length': 50,
    'seed': SEED,

    # Walk-forward split (identical to DDPG and tabular)
    'val_split': 0.2,  # 20% of train data for validation

    # Training
    'total_timesteps': 1000000,  # Same as DDPG
    'print_every': 50000,
    'val_every': 200000,

    # TD3 Hyperparameters (similar to DDPG with TD3-specific additions)
    'learning_rate': 1e-4,
    'buffer_size': 100000,
    'learning_starts': 10000,
    'batch_size': 256,
    'tau': 0.005,
    'gamma': 0.99,
    'train_freq': 1,
    'gradient_steps': 1,
    'action_noise_std': 0.1,

    # TD3-specific parameters
    'policy_delay': 2,  # Delay policy updates
    'target_policy_noise': 0.2,  # Noise added to target policy
    'target_noise_clip': 0.5,  # Clip target noise

    # Network architecture
    'net_arch': [256, 256],

    # Reward shaping (identical to DDPG and tabular)
    'reward_scale': 100,
    'cost_penalty': 0.1,
    'drawdown_threshold': 0.15,
    'drawdown_penalty': 0.05,
}


class ValidationCallback(BaseCallback):
    """Callback for periodic validation during training."""

    def __init__(self, val_env, eval_freq=10000, n_eval_episodes=30, print_freq=10000, verbose=1):
        super().__init__(verbose)
        self.val_env = val_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.print_freq = print_freq
        self.val_results = []

    def _on_step(self) -> bool:
        # Print progress
        if self.n_calls % self.print_freq == 0:
            print(f"  Timestep: {self.n_calls}")

        if self.n_calls % self.eval_freq == 0:
            # Evaluate on validation
            val_returns = []
            for _ in range(self.n_eval_episodes):
                obs, _ = self.val_env.reset()
                episode_return = 0
                done = False
                portfolio_value = 1.0

                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.val_env.step(action)
                    episode_return += reward
                    portfolio_value = info.get('portfolio_value', portfolio_value)
                    done = terminated or truncated

                val_returns.append(portfolio_value)

            mean_val = np.mean(val_returns)
            std_val = np.std(val_returns)
            profit_pct = (mean_val - 1) * 100

            self.val_results.append({
                'timestep': self.n_calls,
                'mean': mean_val,
                'std': std_val,
                'profit_pct': profit_pct
            })

            if self.verbose:
                print(f"  Val @ {self.n_calls}: {profit_pct:+.2f}% (mean: {mean_val:.4f})")

        return True


def evaluate_agent(model, env, n_episodes=50):
    """Evaluate trained agent on environment."""
    portfolio_values = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        pv = 1.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            pv = info.get('portfolio_value', pv)
            done = terminated or truncated

        portfolio_values.append(pv)

    return {
        'mean': np.mean(portfolio_values),
        'std': np.std(portfolio_values),
        'profit_pct': (np.mean(portfolio_values) - 1) * 100,
        'all_values': portfolio_values
    }


def main():
    print("=" * 70)
    print("TD3 Training for Portfolio Management (Stable-Baselines3)")
    print("=" * 70)
    print(f"Random Seed: {SEED}")
    print("TD3 improvements over DDPG: Twin critics, delayed updates, target smoothing")

    # ============================================
    # Load and Split Data (Walk-Forward)
    # ============================================
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'poloniex_30m.hf')
    df_train_full = pd.read_hdf(data_path, key='train')
    df_test = pd.read_hdf(data_path, key='test')

    # Split train into train/val (temporal split - last 20% is val)
    val_split_idx = int(len(df_train_full) * (1 - CONFIG['val_split']))
    df_train = df_train_full.iloc[:val_split_idx]
    df_val = df_train_full.iloc[val_split_idx:]

    print(f"\n--- Walk-Forward Data Splits ---")
    print(f"  Train: {len(df_train):>6} periods (~{len(df_train)//48:>3} days)")
    print(f"  Val:   {len(df_val):>6} periods (~{len(df_val)//48:>3} days)")
    print(f"  Test:  {len(df_test):>6} periods (~{len(df_test)//48:>3} days)")

    # ============================================
    # Create Environments
    # ============================================
    env_train = create_sb3_env(df_train, CONFIG, random_reset=True)
    env_val = create_sb3_env(df_val, CONFIG, random_reset=True)
    env_test = create_sb3_env(df_test, CONFIG, random_reset=True)

    n_actions = env_train.action_space.shape[0]

    print(f"\nEnvironment:")
    print(f"  Observation shape: {env_train.observation_space.shape}")
    print(f"  Action shape: {env_train.action_space.shape}")
    print(f"  N assets: {n_actions}")
    print(f"  Transaction cost: {CONFIG['trading_cost']*100:.3f}%")
    print(f"  Episode length: {CONFIG['max_steps']} steps")

    # ============================================
    # Create TD3 Agent
    # ============================================
    # Add exploration noise
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=CONFIG['action_noise_std'] * np.ones(n_actions)
    )

    model = TD3(
        policy="MlpPolicy",
        env=env_train,
        learning_rate=CONFIG['learning_rate'],
        buffer_size=CONFIG['buffer_size'],
        learning_starts=CONFIG['learning_starts'],
        batch_size=CONFIG['batch_size'],
        tau=CONFIG['tau'],
        gamma=CONFIG['gamma'],
        train_freq=CONFIG['train_freq'],
        gradient_steps=CONFIG['gradient_steps'],
        action_noise=action_noise,
        policy_delay=CONFIG['policy_delay'],
        target_policy_noise=CONFIG['target_policy_noise'],
        target_noise_clip=CONFIG['target_noise_clip'],
        policy_kwargs=dict(net_arch=CONFIG['net_arch']),
        verbose=0,
        seed=SEED
    )

    print(f"\nTD3 Agent:")
    print(f"  Learning rate: {CONFIG['learning_rate']}")
    print(f"  Buffer size: {CONFIG['buffer_size']}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Gamma: {CONFIG['gamma']}")
    print(f"  Tau: {CONFIG['tau']}")
    print(f"  Policy delay: {CONFIG['policy_delay']}")
    print(f"  Target policy noise: {CONFIG['target_policy_noise']}")
    print(f"  Target noise clip: {CONFIG['target_noise_clip']}")
    print(f"  Network: {CONFIG['net_arch']}")
    print(f"  Action noise: Normal with sigma={CONFIG['action_noise_std']}")

    # ============================================
    # Training
    # ============================================
    print(f"\nTraining for {CONFIG['total_timesteps']} timesteps")
    print(f"Validation every {CONFIG['val_every']} timesteps")
    print("-" * 70)

    # Create validation callback
    val_callback = ValidationCallback(
        val_env=env_val,
        eval_freq=CONFIG['val_every'],
        n_eval_episodes=30,
        print_freq=50000,
        verbose=1
    )

    # Train
    model.learn(
        total_timesteps=CONFIG['total_timesteps'],
        callback=val_callback,
        progress_bar=False
    )

    # ============================================
    # Final Results
    # ============================================
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)

    # Final evaluation on all splits
    print("\n--- Final Evaluation (50 episodes each) ---")

    train_final = evaluate_agent(model, env_train, n_episodes=50)
    print(f"Train: {train_final['mean']:.4f} ± {train_final['std']:.4f} ({train_final['profit_pct']:+.2f}%)")

    val_final = evaluate_agent(model, env_val, n_episodes=50)
    print(f"Val:   {val_final['mean']:.4f} ± {val_final['std']:.4f} ({val_final['profit_pct']:+.2f}%)")

    test_final = evaluate_agent(model, env_test, n_episodes=50)
    print(f"Test:  {test_final['mean']:.4f} ± {test_final['std']:.4f} ({test_final['profit_pct']:+.2f}%)")

    # ============================================
    # Plotting
    # ============================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('TD3 Training (Stable-Baselines3)', fontsize=14)

    # Validation progress
    ax1 = axes[0, 0]
    if val_callback.val_results:
        timesteps = [r['timestep'] for r in val_callback.val_results]
        val_profits = [r['profit_pct'] for r in val_callback.val_results]
        ax1.plot(timesteps, val_profits, 'g-o', label='Validation', linewidth=2)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Profit (%)')
    ax1.set_title('Validation Performance During Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Final evaluation distribution
    ax2 = axes[0, 1]
    ax2.hist(train_final['all_values'], bins=20, alpha=0.5, label='Train', color='blue')
    ax2.hist(val_final['all_values'], bins=20, alpha=0.5, label='Val', color='green')
    ax2.hist(test_final['all_values'], bins=20, alpha=0.5, label='Test', color='orange')
    ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax2.set_xlabel('Final Portfolio Value')
    ax2.set_ylabel('Count')
    ax2.set_title('Evaluation Distribution by Split')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Summary bar chart
    ax3 = axes[1, 0]
    splits = ['Train', 'Val', 'Test']
    profits = [train_final['profit_pct'], val_final['profit_pct'], test_final['profit_pct']]
    colors = ['blue' if p > 0 else 'red' for p in profits]
    bars = ax3.bar(splits, profits, color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_ylabel('Profit (%)')
    ax3.set_title('Final Profit by Split')
    for bar, profit in zip(bars, profits):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{profit:+.2f}%', ha='center', va='bottom', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')

    # Box plot comparison
    ax4 = axes[1, 1]
    data = [train_final['all_values'], val_final['all_values'], test_final['all_values']]
    bp = ax4.boxplot(data, labels=splits, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'orange']):
        patch.set_facecolor(color)
    ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax4.set_ylabel('Portfolio Value')
    ax4.set_title('Portfolio Value Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('td3_training.png', dpi=150)
    print(f"\nPlot saved to: td3_training.png")
    plt.show()

    # Save model
    model.save('td3_agent')
    print(f"Model saved to: td3_agent.zip")

    return model, {'train': train_final, 'val': val_final, 'test': test_final}


if __name__ == '__main__':
    model, results = main()
