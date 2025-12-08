"""
Compare All Strategies: Tabular RL vs Deep RL vs Baselines

Features:
1. Walk-Forward Split: Evaluates on Train/Val/Test separately
2. K-Seed Runs: Multiple seeds for statistical significance
3. Backtest (sequential through data)
4. Evaluation (random episodes)

Strategies:
- Q-Learning (trained agent)
- Double Q-Learning (trained agent)
- UCB-epsilon Q-Learning (trained agent)
- DDPG (Stable-Baselines3)
- Equal-Weight Buy-and-Hold
- Naive Momentum
- Stay-in-Cash
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from stable_baselines3 import DDPG, TD3, PPO
from rl_portfolio_management.environments import PortfolioEnv
from rl_portfolio_management.wrappers import DiscreteActionWrapper, StateDiscretizer, RewardShaper, create_sb3_env
from rl_portfolio_management.agents import QLearningAgent, DoubleQLearningAgent, UCBQLearningAgent, CashAgent, EqualWeightAgent, MomentumAgent


class DDPGAgentWrapper:
    """Wrapper to make SB3 DDPG agent compatible with our evaluation functions."""

    def __init__(self, model_path='ddpg_agent'):
        self.model = DDPG.load(model_path)
        self.env = None  # Will be set when running

    def set_env(self, env):
        """Set the SB3 environment for this agent."""
        self.env = env

    def reset(self):
        """Reset any internal state."""
        pass

    def choose_action(self, state, training=False):
        """Choose action using the trained DDPG model."""
        action, _ = self.model.predict(state, deterministic=not training)
        return action


class TD3AgentWrapper:
    """Wrapper to make SB3 TD3 agent compatible with our evaluation functions."""

    def __init__(self, model_path='td3_agent'):
        self.model = TD3.load(model_path)
        self.env = None

    def set_env(self, env):
        """Set the SB3 environment for this agent."""
        self.env = env

    def reset(self):
        """Reset any internal state."""
        pass

    def choose_action(self, state, training=False):
        """Choose action using the trained TD3 model."""
        action, _ = self.model.predict(state, deterministic=not training)
        return action


class PPOAgentWrapper:
    """Wrapper to make SB3 PPO agent compatible with our evaluation functions."""

    def __init__(self, model_path='ppo_agent'):
        self.model = PPO.load(model_path)
        self.env = None

    def set_env(self, env):
        """Set the SB3 environment for this agent."""
        self.env = env

    def reset(self):
        """Reset any internal state."""
        pass

    def choose_action(self, state, training=False):
        """Choose action using the trained PPO model."""
        action, _ = self.model.predict(state, deterministic=not training)
        return action


# ============================================
# Metrics
# ============================================
def compute_metrics(portfolio_values, actions=None):
    """Compute Return, Sharpe, Max Drawdown, Turnover."""
    portfolio_values = np.array(portfolio_values)

    if len(portfolio_values) < 2:
        return {'return': 0, 'sharpe': 0, 'max_dd': 0, 'turnover': 0}

    returns = np.diff(portfolio_values) / (portfolio_values[:-1] + 1e-8)
    total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100

    # Sharpe (annualized for 30-min bars: 48 per day)
    std_ret = np.std(returns)
    sharpe = (np.mean(returns) / std_ret * np.sqrt(48 * 365)) if std_ret > 1e-8 else 0

    # Max Drawdown
    peak = portfolio_values[0]
    max_dd = 0
    for v in portfolio_values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)

    # Turnover
    turnover = 0
    if actions and len(actions) > 1:
        changes = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i-1])
        turnover = changes / (len(actions) - 1) * 100

    return {
        'return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd * 100,
        'turnover': turnover
    }


# ============================================
# Environment Creation
# ============================================
def create_env(df, config, random_reset=False):
    """Create wrapped environment for tabular agents."""
    steps = len(df) - config['window_length'] - 10 if not random_reset else config['max_steps']

    env = PortfolioEnv(
        df=df,
        steps=steps,
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


def create_ddpg_env(df, config, random_reset=False):
    """Create SB3-compatible environment for DDPG."""
    # For backtest, we need to set max_steps to cover full data
    config_copy = config.copy()
    if not random_reset:
        config_copy['max_steps'] = len(df) - config['window_length'] - 10

    return create_sb3_env(df, config_copy, random_reset=random_reset)


# ============================================
# Backtest (Sequential)
# ============================================
def run_backtest(agent, env):
    """Run single sequential backtest for tabular agents."""
    agent.reset() if hasattr(agent, 'reset') else None
    state, info = env.reset()

    portfolio_values = [1.0]
    actions = []

    while True:
        action = agent.choose_action(state, training=False)
        actions.append(action)

        state, reward, terminated, truncated, info = env.step(action)
        portfolio_values.append(info.get('portfolio_value', 1.0))

        if terminated or truncated:
            break

    return portfolio_values, actions


def run_backtest_ddpg(agent, env):
    """Run single sequential backtest for DDPG agent."""
    state, info = env.reset()

    portfolio_values = [1.0]
    actions = []

    while True:
        action = agent.choose_action(state, training=False)
        actions.append(action)

        state, reward, terminated, truncated, info = env.step(action)
        portfolio_values.append(info.get('portfolio_value', 1.0))

        if terminated or truncated:
            break

    return portfolio_values, actions


# ============================================
# Evaluation (Random Episodes) with Seed
# ============================================
def run_evaluation_seeded(agent, env, n_episodes=100, seed=None):
    """Run multiple random episodes with a specific seed for tabular agents."""
    if seed is not None:
        np.random.seed(seed)

    all_returns = []
    all_sharpes = []
    all_mdd = []
    all_turnover = []

    for _ in range(n_episodes):
        agent.reset() if hasattr(agent, 'reset') else None
        state, info = env.reset()

        portfolio_values = [1.0]
        actions = []

        while True:
            action = agent.choose_action(state, training=False)
            actions.append(action)

            state, reward, terminated, truncated, info = env.step(action)
            portfolio_values.append(info.get('portfolio_value', 1.0))

            if terminated or truncated:
                break

        metrics = compute_metrics(portfolio_values, actions)
        all_returns.append(metrics['return'])
        all_sharpes.append(metrics['sharpe'])
        all_mdd.append(metrics['max_dd'])
        all_turnover.append(metrics['turnover'])

    return {
        'return_mean': np.mean(all_returns),
        'return_std': np.std(all_returns),
        'sharpe_mean': np.mean(all_sharpes),
        'sharpe_std': np.std(all_sharpes),
        'max_dd_mean': np.mean(all_mdd),
        'max_dd_std': np.std(all_mdd),
        'turnover_mean': np.mean(all_turnover),
        'turnover_std': np.std(all_turnover),
    }


def compute_continuous_turnover(actions):
    """
    Compute turnover for continuous actions (weight changes).
    Turnover = sum of absolute weight changes / number of steps.
    """
    if not actions or len(actions) < 2:
        return 0.0

    total_turnover = 0.0
    for i in range(1, len(actions)):
        prev_weights = np.array(actions[i-1])
        curr_weights = np.array(actions[i])
        # Turnover is sum of absolute weight changes (excluding cash rebalancing effect)
        total_turnover += np.sum(np.abs(curr_weights - prev_weights))

    # Average turnover per step, as percentage
    avg_turnover = (total_turnover / (len(actions) - 1)) * 100
    return avg_turnover


def run_evaluation_seeded_ddpg(agent, df, config, n_episodes=100, seed=None):
    """Run multiple random episodes with a specific seed for DDPG agent."""
    if seed is not None:
        np.random.seed(seed)

    all_returns = []
    all_sharpes = []
    all_mdd = []
    all_turnover = []

    for _ in range(n_episodes):
        # Create fresh env for each episode
        env = create_ddpg_env(df, config, random_reset=True)
        state, info = env.reset()

        portfolio_values = [1.0]
        actions = []

        while True:
            action = agent.choose_action(state, training=False)
            actions.append(action.tolist() if hasattr(action, 'tolist') else action)

            state, reward, terminated, truncated, info = env.step(action)
            portfolio_values.append(info.get('portfolio_value', 1.0))

            if terminated or truncated:
                break

        metrics = compute_metrics(portfolio_values, None)
        all_returns.append(metrics['return'])
        all_sharpes.append(metrics['sharpe'])
        all_mdd.append(metrics['max_dd'])
        # Compute turnover for continuous actions
        all_turnover.append(compute_continuous_turnover(actions))

    return {
        'return_mean': np.mean(all_returns),
        'return_std': np.std(all_returns),
        'sharpe_mean': np.mean(all_sharpes),
        'sharpe_std': np.std(all_sharpes),
        'max_dd_mean': np.mean(all_mdd),
        'max_dd_std': np.std(all_mdd),
        'turnover_mean': np.mean(all_turnover),
        'turnover_std': np.std(all_turnover),
    }


# ============================================
# K-Seed Evaluation
# ============================================
def run_k_seed_evaluation(agent, env, n_episodes=50, k_seeds=5):
    """
    Run evaluation with K different seeds for statistical significance.
    Returns aggregated metrics across all seeds.
    """
    seeds = [42 + i * 100 for i in range(k_seeds)]  # 42, 142, 242, 342, 442

    all_seed_returns = []
    all_seed_sharpes = []
    all_seed_mdd = []
    all_seed_turnover = []

    for seed in seeds:
        result = run_evaluation_seeded(agent, env, n_episodes=n_episodes, seed=seed)
        all_seed_returns.append(result['return_mean'])
        all_seed_sharpes.append(result['sharpe_mean'])
        all_seed_mdd.append(result['max_dd_mean'])
        all_seed_turnover.append(result['turnover_mean'])

    return {
        'return_mean': np.mean(all_seed_returns),
        'return_std': np.std(all_seed_returns),
        'sharpe_mean': np.mean(all_seed_sharpes),
        'sharpe_std': np.std(all_seed_sharpes),
        'max_dd_mean': np.mean(all_seed_mdd),
        'max_dd_std': np.std(all_seed_mdd),
        'turnover_mean': np.mean(all_seed_turnover),
        'turnover_std': np.std(all_seed_turnover),
        'k_seeds': k_seeds,
        'n_episodes_per_seed': n_episodes,
    }


def run_k_seed_evaluation_ddpg(agent, df, config, n_episodes=50, k_seeds=5):
    """
    Run evaluation with K different seeds for DDPG agent.
    """
    seeds = [42 + i * 100 for i in range(k_seeds)]

    all_seed_returns = []
    all_seed_sharpes = []
    all_seed_mdd = []
    all_seed_turnover = []

    for seed in seeds:
        result = run_evaluation_seeded_ddpg(agent, df, config, n_episodes=n_episodes, seed=seed)
        all_seed_returns.append(result['return_mean'])
        all_seed_sharpes.append(result['sharpe_mean'])
        all_seed_mdd.append(result['max_dd_mean'])
        all_seed_turnover.append(result['turnover_mean'])

    return {
        'return_mean': np.mean(all_seed_returns),
        'return_std': np.std(all_seed_returns),
        'sharpe_mean': np.mean(all_seed_sharpes),
        'sharpe_std': np.std(all_seed_sharpes),
        'max_dd_mean': np.mean(all_seed_mdd),
        'max_dd_std': np.std(all_seed_mdd),
        'turnover_mean': np.mean(all_seed_turnover),
        'turnover_std': np.std(all_seed_turnover),
        'k_seeds': k_seeds,
        'n_episodes_per_seed': n_episodes,
    }


# ============================================
# Create Agents
# ============================================
def create_tabular_agents():
    """Create tabular RL agents for comparison."""
    agents = {}

    # Q-Learning (3375 states with volatility feature)
    ql_agent = QLearningAgent(n_states=3375, n_actions=81)
    ql_agent.load('q_learning_agent.pkl')
    agents['Q-Learning'] = ql_agent

    # Double Q-Learning
    dql_agent = DoubleQLearningAgent(n_states=3375, n_actions=81)
    dql_agent.load('double_q_learning_agent.pkl')
    agents['Double-Q'] = dql_agent

    # UCB-epsilon Q-Learning
    ucb_agent = UCBQLearningAgent(n_states=3375, n_actions=81)
    ucb_agent.load('ucb_q_learning_agent.pkl')
    agents['UCB-Q'] = ucb_agent

    # Baselines
    agents['Cash'] = CashAgent()
    agents['Equal-Weight'] = EqualWeightAgent()
    agents['Momentum'] = MomentumAgent(n_assets=4, rebalance_steps=1440, top_k=2)

    return agents


def create_deep_rl_agents():
    """Create deep RL agents for comparison."""
    agents = {}

    # DDPG
    try:
        ddpg_agent = DDPGAgentWrapper('ddpg_agent')
        agents['DDPG'] = ddpg_agent
        print("  Loaded DDPG agent")
    except Exception as e:
        print(f"  Warning: Could not load DDPG agent: {e}")

    # TD3
    try:
        td3_agent = TD3AgentWrapper('td3_agent')
        agents['TD3'] = td3_agent
        print("  Loaded TD3 agent")
    except Exception as e:
        print(f"  Warning: Could not load TD3 agent: {e}")

    # PPO
    try:
        ppo_agent = PPOAgentWrapper('ppo_agent')
        agents['PPO'] = ppo_agent
        print("  Loaded PPO agent")
    except Exception as e:
        print(f"  Warning: Could not load PPO agent: {e}")

    return agents


# ============================================
# Main
# ============================================
def main():
    print("=" * 80)
    print("COMPREHENSIVE STRATEGY COMPARISON: Tabular RL vs Deep RL vs Baselines")
    print("=" * 80)

    # Configuration (matches training config)
    config = {
        'trading_cost': 0.00025,
        'delta': 0.05,
        'window_length': 50,
        'reward_scale': 100,
        'cost_penalty': 0.1,          # λ multiplier for transaction costs
        'drawdown_threshold': 0.15,   # 15% drawdown before penalty
        'drawdown_penalty': 0.05,     # Penalty multiplier
        'max_steps': 256,
    }

    # K-seed configuration
    K_SEEDS = 5
    EPISODES_PER_SEED = 50

    # Load all data splits (Walk-Forward)
    data_path = os.getenv('DATA_PATH') or os.path.join(os.path.dirname(__file__), 'data', 'poloniex_30m.hf')

    print("\n" + "-" * 80)
    print("WALK-FORWARD DATA SPLITS")
    print("-" * 80)

    # Load original data
    df_train_full = pd.read_hdf(data_path, key='train')
    df_test = pd.read_hdf(data_path, key='test')

    # Split train into train/val (temporal split - last 20% is val)
    VAL_SPLIT = 0.2
    val_split_idx = int(len(df_train_full) * (1 - VAL_SPLIT))
    df_train = df_train_full.iloc[:val_split_idx]
    df_val = df_train_full.iloc[val_split_idx:]

    print(f"  Train: {len(df_train):>6} periods (~{len(df_train)//48:>3} days)")
    print(f"  Val:   {len(df_val):>6} periods (~{len(df_val)//48:>3} days)")
    print(f"  Test:  {len(df_test):>6} periods (~{len(df_test)//48:>3} days)")
    print(f"  Total: {len(df_train)+len(df_val)+len(df_test):>6} periods")

    # Create agents
    print("\nLoading agents")
    tabular_agents = create_tabular_agents()
    deep_rl_agents = create_deep_rl_agents()
    print(f"Tabular agents: {list(tabular_agents.keys())}")
    print(f"Deep RL agents: {list(deep_rl_agents.keys())}")

    # ============================================
    # SECTION 1: BACKTEST ON ALL SPLITS (Walk-Forward)
    # ============================================
    print("\n" + "=" * 80)
    print("SECTION 1: WALK-FORWARD BACKTEST (Sequential through each split)")
    print("=" * 80)

    splits = {'Train': df_train, 'Val': df_val, 'Test': df_test}
    all_backtest_results = {}

    for split_name, df_split in splits.items():
        print(f"\n--- {split_name} Data ---")
        backtest_results = {}

        # Tabular agents
        for name, agent in tabular_agents.items():
            env = create_env(df_split, config, random_reset=False)
            values, actions = run_backtest(agent, env)
            metrics = compute_metrics(values, actions)
            metrics['values'] = values
            backtest_results[name] = metrics

        # Deep RL agents (DDPG)
        for name, agent in deep_rl_agents.items():
            env = create_ddpg_env(df_split, config, random_reset=False)
            values, actions = run_backtest_ddpg(agent, env)
            metrics = compute_metrics(values, None)  # Turnover computed separately for continuous
            metrics['turnover'] = compute_continuous_turnover(actions)  # Use continuous turnover
            metrics['values'] = values
            backtest_results[name] = metrics

        all_backtest_results[split_name] = backtest_results

        print(f"{'Strategy':<15} {'Return':>10} {'Sharpe':>10} {'Max DD':>10} {'Turnover':>10}")
        print("-" * 55)
        for name, m in backtest_results.items():
            print(f"{name:<15} {m['return']:>+9.2f}% {m['sharpe']:>10.3f} {m['max_dd']:>9.2f}% {m['turnover']:>9.1f}%")

    # ============================================
    # SECTION 2: K-SEED EVALUATION ON TEST DATA
    # ============================================
    print("\n" + "=" * 80)
    print(f"SECTION 2: K-SEED EVALUATION (K={K_SEEDS} seeds, {EPISODES_PER_SEED} episodes each)")
    print("=" * 80)
    print("This provides statistically significant results by averaging across multiple seeds.\n")

    k_seed_results = {}

    # Tabular agents
    for name, agent in tabular_agents.items():
        print(f"  Running K-seed evaluation: {name}")
        env = create_env(df_test, config, random_reset=True)
        k_seed_results[name] = run_k_seed_evaluation(
            agent, env,
            n_episodes=EPISODES_PER_SEED,
            k_seeds=K_SEEDS
        )

    # Deep RL agents
    for name, agent in deep_rl_agents.items():
        print(f"  Running K-seed evaluation: {name}")
        k_seed_results[name] = run_k_seed_evaluation_ddpg(
            agent, df_test, config,
            n_episodes=EPISODES_PER_SEED,
            k_seeds=K_SEEDS
        )

    print(f"\n{'Strategy':<15} {'Return':>14} {'Sharpe':>14} {'Max DD':>14} {'Turnover':>14}")
    print("-" * 71)
    for name, m in k_seed_results.items():
        print(f"{name:<15} {m['return_mean']:>+6.2f}±{m['return_std']:>5.2f}% "
              f"{m['sharpe_mean']:>6.3f}±{m['sharpe_std']:>5.3f} "
              f"{m['max_dd_mean']:>6.2f}±{m['max_dd_std']:>5.2f}% "
              f"{m['turnover_mean']:>6.1f}±{m['turnover_std']:>4.1f}%")

    # ============================================
    # SECTION 3: SUMMARY STATISTICS TABLE
    # ============================================
    print("\n" + "=" * 80)
    print("SECTION 3: SUMMARY TABLE (All Splits + K-Seed)")
    print("=" * 80)

    all_agents = {**tabular_agents, **deep_rl_agents}

    print(f"\n{'Strategy':<15} {'Train':>12} {'Val':>12} {'Test':>12} {'K-Seed Test':>14}")
    print("-" * 70)
    for name in all_agents.keys():
        train_ret = all_backtest_results['Train'][name]['return']
        val_ret = all_backtest_results['Val'][name]['return']
        test_ret = all_backtest_results['Test'][name]['return']
        kseed_ret = k_seed_results[name]['return_mean']
        kseed_std = k_seed_results[name]['return_std']
        print(f"{name:<15} {train_ret:>+11.2f}% {val_ret:>+11.2f}% {test_ret:>+11.2f}% {kseed_ret:>+6.2f}±{kseed_std:>5.2f}%")

    # ============================================
    # SECTION 4: PLOT EQUITY CURVES
    # ============================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    colors = {
        'Q-Learning': 'blue', 'Double-Q': 'purple', 'UCB-Q': 'cyan',
        'DDPG': 'red', 'TD3': 'darkred', 'PPO': 'magenta',  # Deep RL
        'Cash': 'gray', 'Equal-Weight': 'green', 'Momentum': 'orange'
    }

    # Top row: Equity curves for each split
    for idx, (split_name, df_split) in enumerate(splits.items()):
        ax = axes[0, idx]
        for name, m in all_backtest_results[split_name].items():
            color = colors.get(name, 'black')
            ax.plot(m['values'], label=f"{name} ({m['return']:+.1f}%)",
                   linewidth=2, color=color)
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Portfolio Value')
        ax.set_title(f'{split_name} Data: Equity Curves')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Bottom row: Drawdowns for each split
    for idx, (split_name, df_split) in enumerate(splits.items()):
        ax = axes[1, idx]
        for name, m in all_backtest_results[split_name].items():
            values = np.array(m['values'])
            peak = np.maximum.accumulate(values)
            dd = (peak - values) / peak * 100
            color = colors.get(name, 'black')
            ax.plot(-dd, label=f"{name} (Max: {m['max_dd']:.1f}%)",
                   linewidth=1.5, color=color)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title(f'{split_name} Data: Drawdowns')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('strategy_comparison_walkforward.png', dpi=150)
    print(f"\nPlot saved to: strategy_comparison_walkforward.png")
    plt.show()

    # ============================================
    # SECTION 5: FINAL ANALYSIS
    # ============================================
    print("\n" + "=" * 80)
    print("SECTION 5: FINAL ANALYSIS")
    print("=" * 80)

    ql_test = all_backtest_results['Test']['Q-Learning']
    ew_test = all_backtest_results['Test']['Equal-Weight']
    mom_test = all_backtest_results['Test']['Momentum']
    cash_test = all_backtest_results['Test']['Cash']

    print("\n--- Q-Learning Performance (Test Data Backtest) ---")
    print(f"  Return:       {ql_test['return']:+.2f}%")
    print(f"  Sharpe Ratio: {ql_test['sharpe']:.3f}")
    print(f"  Max Drawdown: {ql_test['max_dd']:.2f}%")
    print(f"  Turnover:     {ql_test['turnover']:.1f}%")

    print("\n--- Comparison vs Baselines (Test Backtest) ---")

    comparisons = [
        ('Equal-Weight', ew_test),
        ('Momentum', mom_test),
        ('Cash', cash_test),
    ]

    for baseline_name, baseline_metrics in comparisons:
        diff = ql_test['return'] - baseline_metrics['return']
        if diff > 0:
            print(f"  vs {baseline_name:<12}: Q-Learning BEATS by {diff:+.2f}%")
        else:
            print(f"  vs {baseline_name:<12}: Q-Learning underperforms by {diff:.2f}%")

    print("\n--- K-Seed Statistical Significance ---")
    ql_kseed = k_seed_results['Q-Learning']
    ew_kseed = k_seed_results['Equal-Weight']

    # Simple statistical test: check if Q-Learning mean - 1 std > Equal-Weight mean + 1 std
    ql_lower = ql_kseed['return_mean'] - ql_kseed['return_std']
    ew_upper = ew_kseed['return_mean'] + ew_kseed['return_std']

    if ql_lower > ew_upper:
        print(f"  Q-Learning ({ql_kseed['return_mean']:+.2f}%) is STATISTICALLY better than Equal-Weight ({ew_kseed['return_mean']:+.2f}%)")
    else:
        print(f"  Results overlap within 1-std: Q-Learning ({ql_kseed['return_mean']:+.2f}±{ql_kseed['return_std']:.2f}%) vs Equal-Weight ({ew_kseed['return_mean']:+.2f}±{ew_kseed['return_std']:.2f}%)")

    print("\n--- Walk-Forward Consistency ---")
    ql_train = all_backtest_results['Train']['Q-Learning']['return']
    ql_val = all_backtest_results['Val']['Q-Learning']['return']
    ql_test_ret = all_backtest_results['Test']['Q-Learning']['return']

    if ql_train > 0 and ql_val > 0 and ql_test_ret > 0:
        print(f"  Q-Learning is PROFITABLE on all splits (Train: {ql_train:+.2f}%, Val: {ql_val:+.2f}%, Test: {ql_test_ret:+.2f}%)")
    else:
        print(f"  Q-Learning performance varies: Train: {ql_train:+.2f}%, Val: {ql_val:+.2f}%, Test: {ql_test_ret:+.2f}%")

    # Check for overfitting
    if ql_train > ql_val + 5:
        print(f"  Warning: Possible overfitting (Train >> Val)")
    if ql_val > 0 and ql_test_ret > 0:
        print(f"  Good generalization: Val and Test both profitable")

    return all_backtest_results, k_seed_results


if __name__ == '__main__':
    bt_results, kseed_results = main()
