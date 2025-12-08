"""
Stability Analysis - Combine results and generate plots

This script:
1. Loads results from tabular and deep RL stability studies
2. Combines into unified analysis
3. Generates publication-ready plots and tables

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Output directories
OUTPUT_DIR = 'stability_study'
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')


def load_results():
    """Load results from both tabular and deep RL studies."""
    tabular_file = os.path.join(RESULTS_DIR, 'stability_tabular_results.csv')
    deep_rl_file = os.path.join(RESULTS_DIR, 'stability_deep_rl_results.csv')

    dfs = []

    if os.path.exists(tabular_file):
        df_tabular = pd.read_csv(tabular_file)
        df_tabular['agent_type'] = 'Tabular'
        dfs.append(df_tabular)
        print(f"Loaded tabular results: {len(df_tabular)} rows")
    else:
        print(f"Warning: {tabular_file} not found")

    if os.path.exists(deep_rl_file):
        df_deep = pd.read_csv(deep_rl_file)
        df_deep['agent_type'] = 'Deep RL'
        dfs.append(df_deep)
        print(f"Loaded deep RL results: {len(df_deep)} rows")
    else:
        print(f"Warning: {deep_rl_file} not found")

    if not dfs:
        raise FileNotFoundError("No results files found. Run stability studies first.")

    return pd.concat(dfs, ignore_index=True)


def print_summary_tables(df):
    """Print summary tables to console."""
    print("\n" + "=" * 80)
    print("STABILITY STUDY - COMBINED RESULTS")
    print("=" * 80)

    # Mean return by agent and cost
    print("\n" + "-" * 80)
    print("TABLE 1: MEAN TEST RETURN (%) BY AGENT AND TRANSACTION COST")
    print("-" * 80)
    pivot_return = df.pivot_table(
        values='test_return',
        index='agent',
        columns='cost_bps',
        aggfunc='mean'
    )
    if '2.5bps' in pivot_return.columns:
        pivot_return = pivot_return[['2.5bps', '25bps', '50bps']]
    print(pivot_return.round(2).to_string())

    # Std across seeds
    print("\n" + "-" * 80)
    print("TABLE 2: STD OF TEST RETURN (%) ACROSS SEEDS (Stability Measure)")
    print("-" * 80)
    pivot_std = df.pivot_table(
        values='test_return',
        index='agent',
        columns='cost_bps',
        aggfunc='std'
    )
    if '2.5bps' in pivot_std.columns:
        pivot_std = pivot_std[['2.5bps', '25bps', '50bps']]
    print(pivot_std.round(2).to_string())

    # Sharpe ratio
    print("\n" + "-" * 80)
    print("TABLE 3: MEAN SHARPE RATIO BY AGENT AND TRANSACTION COST")
    print("-" * 80)
    pivot_sharpe = df.pivot_table(
        values='test_sharpe',
        index='agent',
        columns='cost_bps',
        aggfunc='mean'
    )
    if '2.5bps' in pivot_sharpe.columns:
        pivot_sharpe = pivot_sharpe[['2.5bps', '25bps', '50bps']]
    print(pivot_sharpe.round(3).to_string())

    # Max drawdown
    print("\n" + "-" * 80)
    print("TABLE 4: MEAN MAX DRAWDOWN (%) BY AGENT AND TRANSACTION COST")
    print("-" * 80)
    pivot_dd = df.pivot_table(
        values='test_max_dd',
        index='agent',
        columns='cost_bps',
        aggfunc='mean'
    )
    if '2.5bps' in pivot_dd.columns:
        pivot_dd = pivot_dd[['2.5bps', '25bps', '50bps']]
    print(pivot_dd.round(2).to_string())

    # Performance degradation (% drop from baseline to 50bps)
    print("\n" + "-" * 80)
    print("TABLE 5: PERFORMANCE DEGRADATION (Return drop from 2.5bps to 50bps)")
    print("-" * 80)
    if '2.5bps' in pivot_return.columns and '50bps' in pivot_return.columns:
        degradation = pivot_return['2.5bps'] - pivot_return['50bps']
        degradation_pct = (degradation / pivot_return['2.5bps'].abs()) * 100
        deg_df = pd.DataFrame({
            'Return @ 2.5bps': pivot_return['2.5bps'],
            'Return @ 50bps': pivot_return['50bps'],
            'Absolute Drop': degradation,
            'Relative Drop (%)': degradation_pct
        })
        print(deg_df.round(2).to_string())

    return pivot_return, pivot_std, pivot_sharpe


def plot_heatmaps(df):
    """Generate heatmap visualizations."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Stability Study: Performance Across Transaction Costs and Seeds', fontsize=14)

    # Prepare pivot tables
    pivot_return = df.pivot_table(values='test_return', index='agent', columns='cost_bps', aggfunc='mean')
    pivot_std = df.pivot_table(values='test_return', index='agent', columns='cost_bps', aggfunc='std')
    pivot_sharpe = df.pivot_table(values='test_sharpe', index='agent', columns='cost_bps', aggfunc='mean')
    pivot_dd = df.pivot_table(values='test_max_dd', index='agent', columns='cost_bps', aggfunc='mean')

    # Reorder columns if possible
    col_order = ['2.5bps', '25bps', '50bps']
    for pivot in [pivot_return, pivot_std, pivot_sharpe, pivot_dd]:
        available_cols = [c for c in col_order if c in pivot.columns]
        if available_cols:
            pivot = pivot[available_cols]

    # Reorder rows: Tabular first, then Deep RL
    row_order = ['Q-Learning', 'Double-Q', 'UCB-Q', 'DDPG', 'TD3', 'PPO']
    available_rows = [r for r in row_order if r in pivot_return.index]

    # Heatmap 1: Mean Return
    ax1 = axes[0, 0]
    data1 = pivot_return.reindex(available_rows)
    sns.heatmap(data1, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax1,
                cbar_kws={'label': 'Return (%)'})
    ax1.set_title('Mean Test Return (%)')
    ax1.set_xlabel('Transaction Cost')
    ax1.set_ylabel('Agent')

    # Heatmap 2: Std across seeds (stability)
    ax2 = axes[0, 1]
    data2 = pivot_std.reindex(available_rows)
    sns.heatmap(data2, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax2,
                cbar_kws={'label': 'Std (%)'})
    ax2.set_title('Std of Return Across Seeds (Lower = More Stable)')
    ax2.set_xlabel('Transaction Cost')
    ax2.set_ylabel('Agent')

    # Heatmap 3: Sharpe Ratio
    ax3 = axes[1, 0]
    data3 = pivot_sharpe.reindex(available_rows)
    sns.heatmap(data3, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax3,
                cbar_kws={'label': 'Sharpe'})
    ax3.set_title('Mean Sharpe Ratio')
    ax3.set_xlabel('Transaction Cost')
    ax3.set_ylabel('Agent')

    # Heatmap 4: Max Drawdown
    ax4 = axes[1, 1]
    data4 = pivot_dd.reindex(available_rows)
    sns.heatmap(data4, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax4,
                cbar_kws={'label': 'Max DD (%)'})
    ax4.set_title('Mean Max Drawdown (%) - Lower is Better')
    ax4.set_xlabel('Transaction Cost')
    ax4.set_ylabel('Agent')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'stability_heatmaps.png'), dpi=150)
    print(f"\nSaved: {os.path.join(PLOTS_DIR, 'stability_heatmaps.png')}")
    plt.show()


def plot_degradation_lines(df):
    """Plot performance degradation as transaction cost increases."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Performance Degradation with Increasing Transaction Costs', fontsize=14)

    # Colors for agents
    colors = {
        'Q-Learning': 'blue', 'Double-Q': 'purple', 'UCB-Q': 'cyan',
        'DDPG': 'red', 'TD3': 'darkred', 'PPO': 'magenta'
    }
    markers = {
        'Q-Learning': 'o', 'Double-Q': 's', 'UCB-Q': '^',
        'DDPG': 'D', 'TD3': 'v', 'PPO': 'p'
    }

    # X-axis: cost in bps
    cost_map = {'2.5bps': 2.5, '25bps': 25, '50bps': 50}

    # Plot 1: Return vs Cost
    ax1 = axes[0]
    for agent in df['agent'].unique():
        agent_df = df[df['agent'] == agent]
        means = agent_df.groupby('cost_bps')['test_return'].mean()
        stds = agent_df.groupby('cost_bps')['test_return'].std()

        x = [cost_map.get(c, 0) for c in means.index]
        y = means.values
        yerr = stds.values

        color = colors.get(agent, 'gray')
        marker = markers.get(agent, 'o')

        ax1.errorbar(x, y, yerr=yerr, label=agent, color=color, marker=marker,
                     linewidth=2, markersize=8, capsize=5)

    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Transaction Cost (bps)')
    ax1.set_ylabel('Test Return (%)')
    ax1.set_title('Return vs Transaction Cost')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_xticks([2.5, 25, 50])
    ax1.set_xticklabels(['2.5', '25', '50'])

    # Plot 2: Sharpe vs Cost
    ax2 = axes[1]
    for agent in df['agent'].unique():
        agent_df = df[df['agent'] == agent]
        means = agent_df.groupby('cost_bps')['test_sharpe'].mean()
        stds = agent_df.groupby('cost_bps')['test_sharpe'].std()

        x = [cost_map.get(c, 0) for c in means.index]
        y = means.values
        yerr = stds.values

        color = colors.get(agent, 'gray')
        marker = markers.get(agent, 'o')

        ax2.errorbar(x, y, yerr=yerr, label=agent, color=color, marker=marker,
                     linewidth=2, markersize=8, capsize=5)

    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Transaction Cost (bps)')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Sharpe Ratio vs Transaction Cost')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_xticks([2.5, 25, 50])
    ax2.set_xticklabels(['2.5', '25', '50'])

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'stability_degradation.png'), dpi=150)
    print(f"Saved: {os.path.join(PLOTS_DIR, 'stability_degradation.png')}")
    plt.show()


def plot_boxplots(df):
    """Generate box plots showing distribution across seeds."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Return Distribution Across Seeds by Transaction Cost', fontsize=14)

    cost_levels = ['2.5bps', '25bps', '50bps']
    agent_order = ['Q-Learning', 'Double-Q', 'UCB-Q', 'DDPG', 'TD3', 'PPO']

    for idx, cost in enumerate(cost_levels):
        ax = axes[idx]
        cost_df = df[df['cost_bps'] == cost]

        if len(cost_df) > 0:
            # Filter to available agents and maintain order
            available_agents = [a for a in agent_order if a in cost_df['agent'].unique()]
            cost_df_ordered = cost_df[cost_df['agent'].isin(available_agents)]

            sns.boxplot(data=cost_df_ordered, x='agent', y='test_return', ax=ax,
                        order=available_agents, palette='Set2')
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('Agent')
            ax.set_ylabel('Test Return (%)')
            ax.set_title(f'Transaction Cost: {cost}')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'stability_boxplots.png'), dpi=150)
    print(f"Saved: {os.path.join(PLOTS_DIR, 'stability_boxplots.png')}")
    plt.show()


def plot_tabular_vs_deep(df):
    """Compare tabular vs deep RL performance."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Tabular RL vs Deep RL: Stability Comparison', fontsize=14)

    # Group by agent type
    tabular_agents = ['Q-Learning', 'Double-Q', 'UCB-Q']
    deep_agents = ['DDPG', 'TD3', 'PPO']

    df['agent_category'] = df['agent'].apply(
        lambda x: 'Tabular' if x in tabular_agents else 'Deep RL'
    )

    # Plot 1: Mean return by category and cost
    ax1 = axes[0]
    category_means = df.groupby(['agent_category', 'cost_bps'])['test_return'].mean().unstack()
    if '2.5bps' in category_means.columns:
        category_means = category_means[['2.5bps', '25bps', '50bps']]
    category_means.plot(kind='bar', ax=ax1, width=0.8)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_xlabel('Agent Category')
    ax1.set_ylabel('Mean Test Return (%)')
    ax1.set_title('Mean Return: Tabular vs Deep RL')
    ax1.legend(title='Cost')
    ax1.tick_params(axis='x', rotation=0)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Stability (std across seeds) by category and cost
    ax2 = axes[1]
    category_stds = df.groupby(['agent_category', 'cost_bps'])['test_return'].std().unstack()
    if '2.5bps' in category_stds.columns:
        category_stds = category_stds[['2.5bps', '25bps', '50bps']]
    category_stds.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_xlabel('Agent Category')
    ax2.set_ylabel('Std of Return Across Seeds (%)')
    ax2.set_title('Stability: Tabular vs Deep RL (Lower = More Stable)')
    ax2.legend(title='Cost')
    ax2.tick_params(axis='x', rotation=0)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'stability_tabular_vs_deep.png'), dpi=150)
    print(f"Saved: {os.path.join(PLOTS_DIR, 'stability_tabular_vs_deep.png')}")
    plt.show()


def save_latex_tables(df):
    """Save results as LaTeX tables for paper."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Mean return table
    pivot_return = df.pivot_table(values='test_return', index='agent', columns='cost_bps', aggfunc='mean')
    if '2.5bps' in pivot_return.columns:
        pivot_return = pivot_return[['2.5bps', '25bps', '50bps']]

    latex_return = pivot_return.round(2).to_latex(
        caption='Mean Test Return (\\%) by Agent and Transaction Cost',
        label='tab:stability_return'
    )

    with open(os.path.join(PLOTS_DIR, 'table_return.tex'), 'w') as f:
        f.write(latex_return)
    print(f"Saved: {os.path.join(PLOTS_DIR, 'table_return.tex')}")

    # Sharpe ratio table
    pivot_sharpe = df.pivot_table(values='test_sharpe', index='agent', columns='cost_bps', aggfunc='mean')
    if '2.5bps' in pivot_sharpe.columns:
        pivot_sharpe = pivot_sharpe[['2.5bps', '25bps', '50bps']]

    latex_sharpe = pivot_sharpe.round(3).to_latex(
        caption='Mean Sharpe Ratio by Agent and Transaction Cost',
        label='tab:stability_sharpe'
    )

    with open(os.path.join(PLOTS_DIR, 'table_sharpe.tex'), 'w') as f:
        f.write(latex_sharpe)
    print(f"Saved: {os.path.join(PLOTS_DIR, 'table_sharpe.tex')}")


def main():
    print("=" * 80)
    print("STABILITY ANALYSIS")
    print("=" * 80)

    # Load results
    df = load_results()
    print(f"\nTotal results: {len(df)} rows")
    print(f"Agents: {df['agent'].unique().tolist()}")
    print(f"Costs: {df['cost_bps'].unique().tolist()}")
    print(f"Seeds: {df['seed'].unique().tolist()}")

    # Print summary tables
    print_summary_tables(df)

    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    plot_heatmaps(df)
    plot_degradation_lines(df)
    plot_boxplots(df)
    plot_tabular_vs_deep(df)

    # Save LaTeX tables
    save_latex_tables(df)

    # Save combined results
    combined_file = os.path.join(RESULTS_DIR, 'stability_combined_results.csv')
    df.to_csv(combined_file, index=False)
    print(f"\nSaved combined results: {combined_file}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return df


if __name__ == '__main__':
    df = main()
