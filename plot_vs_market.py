"""
Plot Tabular RL and Deep RL vs Market Performance

Compares:
1. Q-Learning agent equity curve
2. Double Q-Learning agent equity curve
3. UCB-epsilon Q-Learning agent equity curve
4. DDPG (Deep RL) agent equity curve
5. TD3 (Deep RL) agent equity curve
6. PPO (Deep RL) agent equity curve
7. Individual asset performance (BTC, ETH, etc.)
8. Equal-weight portfolio of all assets
9. Baselines (Cash, Momentum)
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
    config_copy = config.copy()
    if not random_reset:
        config_copy['max_steps'] = len(df) - config['window_length'] - 10
    return create_sb3_env(df, config_copy, random_reset=random_reset)


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


def run_backtest_ddpg(model, env):
    """Run single sequential backtest for DDPG agent."""
    state, info = env.reset()

    portfolio_values = [1.0]

    while True:
        action, _ = model.predict(state, deterministic=True)
        state, reward, terminated, truncated, info = env.step(action)
        portfolio_values.append(info.get('portfolio_value', 1.0))

        if terminated or truncated:
            break

    return portfolio_values


def get_asset_prices(df, window_length=50):
    """
    Extract individual asset price series from the dataframe.
    Returns normalized prices (starting at 1.0).
    """
    # Get unique assets
    assets = df.columns.get_level_values(0).unique().tolist()

    # Get close prices for each asset
    asset_prices = {}
    for asset in assets:
        if asset != 'CASH':  # Skip cash if present
            try:
                close_prices = df[asset]['close'].values[window_length:]
                # Normalize to start at 1.0
                normalized = close_prices / close_prices[0]
                asset_prices[asset] = normalized
            except:
                pass

    return asset_prices


def compute_equal_weight_market(asset_prices):
    """Compute equal-weight buy-and-hold of all assets."""
    if not asset_prices:
        return None

    # Get minimum length
    min_len = min(len(prices) for prices in asset_prices.values())

    # Compute equal weight average
    equal_weight = np.zeros(min_len)
    for asset, prices in asset_prices.items():
        equal_weight += prices[:min_len]
    equal_weight /= len(asset_prices)

    return equal_weight


def main():
    print("=" * 70)
    print("TABULAR RL & DEEP RL vs MARKET PERFORMANCE")
    print("=" * 70)

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

    # Load data
    # Allow overriding data path via environment variable `DATA_PATH`
    data_path = os.getenv('DATA_PATH') or os.path.join(os.path.dirname(__file__), 'data', 'poloniex_30m.hf')
    df_train_full = pd.read_hdf(data_path, key='train')
    df_test = pd.read_hdf(data_path, key='test')

    # Split train into train/val
    VAL_SPLIT = 0.2
    val_split_idx = int(len(df_train_full) * (1 - VAL_SPLIT))
    df_train = df_train_full.iloc[:val_split_idx]
    df_val = df_train_full.iloc[val_split_idx:]

    print(f"\nData splits:")
    print(f"  Train: {len(df_train)} periods")
    print(f"  Val:   {len(df_val)} periods")
    print(f"  Test:  {len(df_test)} periods")

    # Create tabular agents
    print("\nLoading agents")
    ql_agent = QLearningAgent(n_states=3375, n_actions=81)
    ql_agent.load('q_learning_agent.pkl')

    dql_agent = DoubleQLearningAgent(n_states=3375, n_actions=81)
    dql_agent.load('double_q_learning_agent.pkl')

    ucb_agent = UCBQLearningAgent(n_states=3375, n_actions=81)
    ucb_agent.load('ucb_q_learning_agent.pkl')

    momentum_agent = MomentumAgent(n_assets=4, rebalance_steps=1440, top_k=2)

    # Load DDPG agent
    ddpg_model = None
    try:
        ddpg_model = DDPG.load('ddpg_agent')
        print("  Loaded DDPG agent")
    except Exception as e:
        print(f"  Warning: Could not load DDPG agent: {e}")

    # Load TD3 agent
    td3_model = None
    try:
        td3_model = TD3.load('td3_agent')
        print("  Loaded TD3 agent")
    except Exception as e:
        print(f"  Warning: Could not load TD3 agent: {e}")

    # Load PPO agent
    ppo_model = None
    try:
        ppo_model = PPO.load('ppo_agent')
        print("  Loaded PPO agent")
    except Exception as e:
        print(f"  Warning: Could not load PPO agent: {e}")

    # Run backtests on TEST data
    print("\nRunning backtests on TEST data")

    env_test = create_env(df_test, config, random_reset=False)
    ql_values, _ = run_backtest(ql_agent, env_test)

    env_test2 = create_env(df_test, config, random_reset=False)
    dql_values, _ = run_backtest(dql_agent, env_test2)

    env_test3 = create_env(df_test, config, random_reset=False)
    ucb_values, _ = run_backtest(ucb_agent, env_test3)

    env_test4 = create_env(df_test, config, random_reset=False)
    mom_values, _ = run_backtest(momentum_agent, env_test4)

    # DDPG backtest
    ddpg_values = None
    if ddpg_model is not None:
        env_ddpg = create_ddpg_env(df_test, config, random_reset=False)
        ddpg_values = run_backtest_ddpg(ddpg_model, env_ddpg)

    # TD3 backtest
    td3_values = None
    if td3_model is not None:
        env_td3 = create_ddpg_env(df_test, config, random_reset=False)
        td3_values = run_backtest_ddpg(td3_model, env_td3)

    # PPO backtest
    ppo_values = None
    if ppo_model is not None:
        env_ppo = create_ddpg_env(df_test, config, random_reset=False)
        ppo_values = run_backtest_ddpg(ppo_model, env_ppo)

    # Get actual asset prices
    print("Extracting asset prices")
    asset_prices = get_asset_prices(df_test, config['window_length'])
    print(f"  Found {len(asset_prices)} assets: {list(asset_prices.keys())}")

    # Compute equal-weight market portfolio
    equal_weight_market = compute_equal_weight_market(asset_prices)

    # Align lengths
    lengths = [
        len(ql_values),
        len(dql_values),
        len(ucb_values),
        len(mom_values),
        len(equal_weight_market) if equal_weight_market is not None else float('inf'),
        min(len(p) for p in asset_prices.values()) if asset_prices else float('inf')
    ]
    if ddpg_values is not None:
        lengths.append(len(ddpg_values))
    if td3_values is not None:
        lengths.append(len(td3_values))
    if ppo_values is not None:
        lengths.append(len(ppo_values))
    min_len = min(lengths)

    ql_values = ql_values[:min_len]
    dql_values = dql_values[:min_len]
    ucb_values = ucb_values[:min_len]
    mom_values = mom_values[:min_len]
    if ddpg_values is not None:
        ddpg_values = ddpg_values[:min_len]
    if td3_values is not None:
        td3_values = td3_values[:min_len]
    if ppo_values is not None:
        ppo_values = ppo_values[:min_len]
    if equal_weight_market is not None:
        equal_weight_market = equal_weight_market[:min_len]

    # ============================================
    # Create Plots
    # ============================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Tabular RL & Deep RL vs Market Performance (Test Data)', fontsize=14)

    # ============================================
    # Plot 1: RL Agents vs Individual Assets
    # ============================================
    ax1 = axes[0, 0]
    ax1.plot(ql_values, label=f'Q-Learning ({(ql_values[-1]-1)*100:+.1f}%)',
             linewidth=2.5, color='blue')
    ax1.plot(dql_values, label=f'Double-Q ({(dql_values[-1]-1)*100:+.1f}%)',
             linewidth=2.5, color='purple')
    ax1.plot(ucb_values, label=f'UCB-Q ({(ucb_values[-1]-1)*100:+.1f}%)',
             linewidth=2.5, color='cyan')
    if ddpg_values is not None:
        ax1.plot(ddpg_values, label=f'DDPG ({(ddpg_values[-1]-1)*100:+.1f}%)',
                 linewidth=2.5, color='red')
    if td3_values is not None:
        ax1.plot(td3_values, label=f'TD3 ({(td3_values[-1]-1)*100:+.1f}%)',
                 linewidth=2.5, color='darkred')
    if ppo_values is not None:
        ax1.plot(ppo_values, label=f'PPO ({(ppo_values[-1]-1)*100:+.1f}%)',
                 linewidth=2.5, color='magenta')

    asset_colors = ['green', 'orange', 'brown', 'pink']
    for i, (asset, prices) in enumerate(asset_prices.items()):
        ret = (prices[min_len-1] - 1) * 100
        ax1.plot(prices[:min_len], label=f'{asset} ({ret:+.1f}%)',
                 linewidth=1.5, alpha=0.7, color=asset_colors[i % len(asset_colors)])

    ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Normalized Value')
    ax1.set_title('RL Agents vs Individual Assets')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ============================================
    # Plot 2: RL Agents vs Market Benchmarks
    # ============================================
    ax2 = axes[0, 1]
    ax2.plot(ql_values, label=f'Q-Learning ({(ql_values[-1]-1)*100:+.1f}%)',
             linewidth=2.5, color='blue')
    ax2.plot(dql_values, label=f'Double-Q ({(dql_values[-1]-1)*100:+.1f}%)',
             linewidth=2.5, color='purple')
    ax2.plot(ucb_values, label=f'UCB-Q ({(ucb_values[-1]-1)*100:+.1f}%)',
             linewidth=2.5, color='cyan')
    if ddpg_values is not None:
        ax2.plot(ddpg_values, label=f'DDPG ({(ddpg_values[-1]-1)*100:+.1f}%)',
                 linewidth=2.5, color='red')
    if td3_values is not None:
        ax2.plot(td3_values, label=f'TD3 ({(td3_values[-1]-1)*100:+.1f}%)',
                 linewidth=2.5, color='darkred')
    if ppo_values is not None:
        ax2.plot(ppo_values, label=f'PPO ({(ppo_values[-1]-1)*100:+.1f}%)',
                 linewidth=2.5, color='magenta')
    if equal_weight_market is not None:
        ax2.plot(equal_weight_market, label=f'Equal-Weight Market ({(equal_weight_market[-1]-1)*100:+.1f}%)',
                 linewidth=2, color='green', linestyle='--')
    ax2.plot(mom_values, label=f'Momentum ({(mom_values[-1]-1)*100:+.1f}%)',
             linewidth=2, color='orange', linestyle=':')

    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Normalized Value')
    ax2.set_title('RL Agents vs Market Benchmarks')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ============================================
    # Plot 3: Tabular vs Deep RL Comparison
    # ============================================
    ax3 = axes[1, 0]
    ax3.plot(ql_values, label=f'Q-Learning ({(ql_values[-1]-1)*100:+.1f}%)',
             linewidth=2.5, color='blue')
    ax3.plot(dql_values, label=f'Double-Q ({(dql_values[-1]-1)*100:+.1f}%)',
             linewidth=2.5, color='purple')
    ax3.plot(ucb_values, label=f'UCB-Q ({(ucb_values[-1]-1)*100:+.1f}%)',
             linewidth=2.5, color='cyan')
    if ddpg_values is not None:
        ax3.plot(ddpg_values, label=f'DDPG ({(ddpg_values[-1]-1)*100:+.1f}%)',
                 linewidth=2.5, color='red')
    if td3_values is not None:
        ax3.plot(td3_values, label=f'TD3 ({(td3_values[-1]-1)*100:+.1f}%)',
                 linewidth=2.5, color='darkred')
    if ppo_values is not None:
        ax3.plot(ppo_values, label=f'PPO ({(ppo_values[-1]-1)*100:+.1f}%)',
                 linewidth=2.5, color='magenta')

    ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Normalized Value')
    ax3.set_title('Tabular RL vs Deep RL Comparison')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    # ============================================
    # Plot 4: Drawdown Comparison
    # ============================================
    ax4 = axes[1, 1]

    def compute_drawdown(values):
        values = np.array(values)
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak * 100
        return drawdown

    ql_dd = compute_drawdown(ql_values)
    ax4.fill_between(range(len(ql_dd)), 0, -ql_dd, alpha=0.5, color='blue',
                     label=f'Q-Learning (Max: {max(ql_dd):.1f}%)')

    dql_dd = compute_drawdown(dql_values)
    ax4.fill_between(range(len(dql_dd)), 0, -dql_dd, alpha=0.3, color='purple',
                     label=f'Double-Q (Max: {max(dql_dd):.1f}%)')

    ucb_dd = compute_drawdown(ucb_values)
    ax4.fill_between(range(len(ucb_dd)), 0, -ucb_dd, alpha=0.3, color='cyan',
                     label=f'UCB-Q (Max: {max(ucb_dd):.1f}%)')

    if ddpg_values is not None:
        ddpg_dd = compute_drawdown(ddpg_values)
        ax4.plot(-ddpg_dd, linewidth=2, color='red',
                 label=f'DDPG (Max: {max(ddpg_dd):.1f}%)')

    if td3_values is not None:
        td3_dd = compute_drawdown(td3_values)
        ax4.plot(-td3_dd, linewidth=2, color='darkred',
                 label=f'TD3 (Max: {max(td3_dd):.1f}%)')

    if ppo_values is not None:
        ppo_dd = compute_drawdown(ppo_values)
        ax4.plot(-ppo_dd, linewidth=2, color='magenta',
                 label=f'PPO (Max: {max(ppo_dd):.1f}%)')

    if equal_weight_market is not None:
        market_dd = compute_drawdown(equal_weight_market)
        ax4.plot(-market_dd, linewidth=2, color='green', linestyle='--',
                 label=f'Equal-Weight Market (Max: {max(market_dd):.1f}%)')

    mom_dd = compute_drawdown(mom_values)
    ax4.plot(-mom_dd, linewidth=2, color='orange', linestyle=':',
             label=f'Momentum (Max: {max(mom_dd):.1f}%)')

    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Drawdown (%)')
    ax4.set_title('Drawdown Comparison')
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('q_learning_vs_market.png', dpi=150)
    print(f"\nPlot saved to: q_learning_vs_market.png")
    plt.show()

    # ============================================
    # Print Summary Statistics
    # ============================================
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY (Test Data)")
    print("=" * 70)

    print(f"\n{'Strategy':<25} {'Return':>12} {'Max DD':>12}")
    print("-" * 50)
    print(f"{'Q-Learning':<25} {(ql_values[-1]-1)*100:>+11.2f}% {max(ql_dd):>11.2f}%")
    print(f"{'Double Q-Learning':<25} {(dql_values[-1]-1)*100:>+11.2f}% {max(dql_dd):>11.2f}%")
    print(f"{'UCB-Q':<25} {(ucb_values[-1]-1)*100:>+11.2f}% {max(ucb_dd):>11.2f}%")

    if ddpg_values is not None:
        print(f"{'DDPG':<25} {(ddpg_values[-1]-1)*100:>+11.2f}% {max(ddpg_dd):>11.2f}%")

    if td3_values is not None:
        print(f"{'TD3':<25} {(td3_values[-1]-1)*100:>+11.2f}% {max(td3_dd):>11.2f}%")

    if ppo_values is not None:
        print(f"{'PPO':<25} {(ppo_values[-1]-1)*100:>+11.2f}% {max(ppo_dd):>11.2f}%")

    if equal_weight_market is not None:
        print(f"{'Equal-Weight Market':<25} {(equal_weight_market[-1]-1)*100:>+11.2f}% {max(market_dd):>11.2f}%")

    print(f"{'Momentum':<25} {(mom_values[-1]-1)*100:>+11.2f}% {max(mom_dd):>11.2f}%")

    for asset, prices in asset_prices.items():
        asset_dd = compute_drawdown(prices[:min_len])
        print(f"{asset:<25} {(prices[min_len-1]-1)*100:>+11.2f}% {max(asset_dd):>11.2f}%")

    # Alpha calculation
    if equal_weight_market is not None:
        print(f"\n{'Alpha (Q-Learning vs Market)':<30} {((ql_values[-1]-1)-(equal_weight_market[-1]-1))*100:>+11.2f}%")
        print(f"{'Alpha (UCB-Q vs Market)':<30} {((ucb_values[-1]-1)-(equal_weight_market[-1]-1))*100:>+11.2f}%")
        if ddpg_values is not None:
            print(f"{'Alpha (DDPG vs Market)':<30} {((ddpg_values[-1]-1)-(equal_weight_market[-1]-1))*100:>+11.2f}%")
        if td3_values is not None:
            print(f"{'Alpha (TD3 vs Market)':<30} {((td3_values[-1]-1)-(equal_weight_market[-1]-1))*100:>+11.2f}%")
        if ppo_values is not None:
            print(f"{'Alpha (PPO vs Market)':<30} {((ppo_values[-1]-1)-(equal_weight_market[-1]-1))*100:>+11.2f}%")


if __name__ == '__main__':
    main()
