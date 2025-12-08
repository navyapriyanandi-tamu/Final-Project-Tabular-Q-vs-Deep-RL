"""
Train Q-Learning Agent for Portfolio Management.

This is the main training script for tabular Q-learning.
Uses optimized parameters from hyperparameter tuning.

Features:
- Walk-forward split: Train (80%) / Val (20%) from original train data
- Validation evaluation during training
- Test evaluation at the end
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from rl_portfolio_management.environments import PortfolioEnv
from rl_portfolio_management.wrappers import DiscreteActionWrapper, StateDiscretizer, RewardShaper
from rl_portfolio_management.agents import QLearningAgent


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
# Configuration
# ============================================
CONFIG = {
    # Environment
    'trading_cost': 0.00025,  # 0.025%
    'delta': 0.05,            # 5% weight changes
    'max_steps': 256,
    'window_length': 50,
    'seed': SEED,

    # Walk-forward split
    'val_split': 0.2,  # 20% of train data for validation

    # Training
    'n_episodes': 30000,
    'print_every': 2500,
    'val_every': 5000,  # Evaluate on validation set every N episodes

    # Agent
    'learning_rate': 0.1,
    'lr_min': 0.01,
    'lr_decay': 0.9999,
    'discount_factor': 0.99,
    'epsilon': 0.9999,
    'epsilon_min': 0.03,
    'epsilon_decay': 0.99985,

    # Reward shaping 
    'reward_scale': 100,
    'cost_penalty': 0.1,          # λ multiplier for transaction costs
    'drawdown_threshold': 0.15,   # 15% drawdown before penalty kicks in
    'drawdown_penalty': 0.05,     # Penalty multiplier for excess drawdown
}


def create_env(df, config, random_reset=True):
    """Create wrapped environment."""
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


def evaluate_on_split(agent, df, config, n_episodes=50, split_name="Val"):
    """Evaluate agent on a specific data split."""
    env = create_env(df, config, random_reset=True)
    results = agent.evaluate(env, n_episodes=n_episodes, max_steps=config['max_steps'])
    profit_pct = (results['mean_portfolio_value'] - 1) * 100
    return {
        'mean': results['mean_portfolio_value'],
        'std': results['std_portfolio_value'],
        'profit_pct': profit_pct,
        'all_values': results['all_portfolio_values']
    }


def main():
    print("=" * 70)
    print("Q-Learning Training for Portfolio Management")
    print("=" * 70)
    print(f"Random Seed: {SEED}")

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
    print(f"  Total: {len(df_train)+len(df_val)+len(df_test):>6} periods")

    # Create training environment
    env = create_env(df_train, CONFIG, random_reset=True)

    n_states = env.observation_space.n
    n_actions = 81  # 3^4

    print(f"\nEnvironment:")
    print(f"  States: {n_states}")
    print(f"  Actions: {n_actions}")
    print(f"  Transaction cost: {CONFIG['trading_cost']*100:.3f}%")
    print(f"  Delta: {CONFIG['delta']*100:.0f}%")
    print(f"  Episode length: {CONFIG['max_steps']} steps")

    # Create agent
    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=CONFIG['learning_rate'],
        lr_min=CONFIG['lr_min'],
        lr_decay=CONFIG['lr_decay'],
        discount_factor=CONFIG['discount_factor'],
        epsilon=CONFIG['epsilon'],
        epsilon_min=CONFIG['epsilon_min'],
        epsilon_decay=CONFIG['epsilon_decay']
    )

    print(f"\nAgent:")
    print(f"  Learning rate: {agent.lr} (min: {agent.lr_min}, decay: {agent.lr_decay})")
    print(f"  Discount factor: {agent.gamma}")
    print(f"  Epsilon: {agent.epsilon} (min: {agent.epsilon_min}, decay: {agent.epsilon_decay})")

    # ============================================
    # Training with Validation Checkpoints
    # ============================================
    print(f"\nTraining for {CONFIG['n_episodes']} episodes")
    print(f"Validation every {CONFIG['val_every']} episodes")
    print("-" * 70)

    # Track validation performance
    val_history = {'episodes': [], 'train_profit': [], 'val_profit': []}

    # Custom training loop with validation
    history = {
        'episode_rewards': [],
        'portfolio_values': [],
        'epsilon_values': [],
        'lr_values': []
    }

    for episode in range(1, CONFIG['n_episodes'] + 1):
        state, info = env.reset()
        episode_reward = 0

        for step in range(CONFIG['max_steps']):
            action = agent.choose_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.learn(state, action, reward, next_state, terminated or truncated)
            state = next_state
            episode_reward += reward

            if terminated or truncated:
                break

        agent.decay_epsilon()
        agent.decay_lr()
        history['episode_rewards'].append(episode_reward)
        history['portfolio_values'].append(info.get('portfolio_value', 1.0))
        history['epsilon_values'].append(agent.epsilon)
        history['lr_values'].append(agent.lr)

        # Print progress
        if episode % CONFIG['print_every'] == 0:
            recent_pv = np.mean(history['portfolio_values'][-500:])
            recent_reward = np.mean(history['episode_rewards'][-500:])
            print(f"Episode {episode:>6} | PV: {recent_pv:.4f} | Reward: {recent_reward:>7.2f} | ε: {agent.epsilon:.4f} | lr: {agent.lr:.4f}")

        # Validation checkpoint
        if episode % CONFIG['val_every'] == 0:
            print(f"\n  --- Validation Checkpoint (Episode {episode}) ---")

            # Evaluate on train
            train_eval = evaluate_on_split(agent, df_train, CONFIG, n_episodes=30, split_name="Train")
            print(f"  Train: {train_eval['profit_pct']:+.2f}%")

            # Evaluate on val
            val_eval = evaluate_on_split(agent, df_val, CONFIG, n_episodes=30, split_name="Val")
            print(f"  Val:   {val_eval['profit_pct']:+.2f}%")

            val_history['episodes'].append(episode)
            val_history['train_profit'].append(train_eval['profit_pct'])
            val_history['val_profit'].append(val_eval['profit_pct'])
            print()

    # ============================================
    # Final Results
    # ============================================
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)

    stats = agent.get_q_table_stats()
    print(f"States visited: {stats['states_visited']} / {n_states} ({stats['coverage']:.1f}%)")

    # Final evaluation on all splits
    print("\n--- Final Evaluation (50 episodes each) ---")

    train_final = evaluate_on_split(agent, df_train, CONFIG, n_episodes=50, split_name="Train")
    print(f"Train: {train_final['mean']:.4f} ± {train_final['std']:.4f} ({train_final['profit_pct']:+.2f}%)")

    val_final = evaluate_on_split(agent, df_val, CONFIG, n_episodes=50, split_name="Val")
    print(f"Val:   {val_final['mean']:.4f} ± {val_final['std']:.4f} ({val_final['profit_pct']:+.2f}%)")

    test_final = evaluate_on_split(agent, df_test, CONFIG, n_episodes=50, split_name="Test")
    print(f"Test:  {test_final['mean']:.4f} ± {test_final['std']:.4f} ({test_final['profit_pct']:+.2f}%)")

    # Summary
    print("\n--- Summary ---")
    if train_final['profit_pct'] > 0 and val_final['profit_pct'] > 0 and test_final['profit_pct'] > 0:
        print("PROFITABLE on all splits!")
    else:
        print("Mixed results across splits.")

    if val_final['profit_pct'] > 0 and test_final['profit_pct'] > 0:
        print("Good generalization: Val and Test both profitable")
    elif train_final['profit_pct'] > val_final['profit_pct'] + 1:
        print("Warning: Possible overfitting (Train >> Val)")

    # ============================================
    # Plotting
    # ============================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'Q-Learning Training (Walk-Forward Split)', fontsize=14)

    window = 500

    # Portfolio values
    ax1 = axes[0, 0]
    pv = history['portfolio_values']
    smoothed = np.convolve(pv, np.ones(window)/window, mode='valid')
    ax1.plot(pv, alpha=0.1, color='green')
    ax1.plot(range(window-1, len(pv)), smoothed, color='green', linewidth=2)
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Portfolio Value')
    ax1.set_title('Portfolio Value (Training)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Rewards
    ax2 = axes[0, 1]
    rewards = history['episode_rewards']
    smoothed_r = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax2.plot(rewards, alpha=0.1, color='blue')
    ax2.plot(range(window-1, len(rewards)), smoothed_r, color='blue', linewidth=2)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward')
    ax2.set_title('Episode Rewards')
    ax2.grid(True, alpha=0.3)

    # Epsilon decay
    ax3 = axes[0, 2]
    ax3.plot(history['epsilon_values'], color='orange', linewidth=2)
    ax3.axhline(y=CONFIG['epsilon_min'], color='red', linestyle='--', alpha=0.5, label='Min epsilon')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')
    ax3.set_title('Exploration Rate')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Validation progress
    ax4 = axes[1, 0]
    ax4.plot(val_history['episodes'], val_history['train_profit'], 'b-o', label='Train', linewidth=2)
    ax4.plot(val_history['episodes'], val_history['val_profit'], 'g-s', label='Val', linewidth=2)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Profit (%)')
    ax4.set_title('Train vs Val Performance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Final evaluation distribution
    ax5 = axes[1, 1]
    ax5.hist(train_final['all_values'], bins=20, alpha=0.5, label='Train', color='blue')
    ax5.hist(val_final['all_values'], bins=20, alpha=0.5, label='Val', color='green')
    ax5.hist(test_final['all_values'], bins=20, alpha=0.5, label='Test', color='orange')
    ax5.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax5.set_xlabel('Final Portfolio Value')
    ax5.set_ylabel('Count')
    ax5.set_title('Evaluation Distribution by Split')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Summary bar chart
    ax6 = axes[1, 2]
    splits = ['Train', 'Val', 'Test']
    profits = [train_final['profit_pct'], val_final['profit_pct'], test_final['profit_pct']]
    colors = ['blue' if p > 0 else 'red' for p in profits]
    bars = ax6.bar(splits, profits, color=colors, alpha=0.7, edgecolor='black')
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax6.set_ylabel('Profit (%)')
    ax6.set_title('Final Profit by Split')
    for bar, profit in zip(bars, profits):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{profit:+.2f}%', ha='center', va='bottom', fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('q_learning_training.png', dpi=150)
    print(f"\nPlot saved to: q_learning_training.png")
    plt.show()

    # Save agent
    agent.save('q_learning_agent.pkl')
    print(f"Agent saved to: q_learning_agent.pkl")

    return agent, {'train': train_final, 'val': val_final, 'test': test_final}


if __name__ == '__main__':
    agent, results = main()
