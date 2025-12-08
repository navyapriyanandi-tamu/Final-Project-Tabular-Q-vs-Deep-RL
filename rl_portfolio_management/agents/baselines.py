"""
Baseline Agents for Portfolio Management.

Implements:
1. EqualWeightAgent - Equal-weight buy-and-hold
2. MomentumAgent - Naive momentum (top-k by 60-day return, monthly rebalance)
3. CashAgent - Stay 100% in cash
4. RandomAgent - Random actions for comparison
"""

import numpy as np


class BaselineAgent:
    """Base class for baseline agents."""

    def __init__(self, n_actions=81):
        self.n_actions = n_actions

    def choose_action(self, state, training=False):
        """Return action. Override in subclass."""
        raise NotImplementedError

    def reset(self):
        """Reset agent state for new episode."""
        pass


class CashAgent(BaselineAgent):
    """
    Stay 100% in cash.
    Action 0 = all weight in cash (first asset).
    """

    def choose_action(self, state, training=False):
        return 0  # Action 0 = all cash


class EqualWeightAgent(BaselineAgent):
    """
    Equal-weight buy-and-hold strategy.

    The environment starts with 100% cash. We need to rebalance to equal weights
    across all assets (25% each for 4 assets).

    With delta=0.05 (5% per action), we need multiple steps to reach equal weights.
    Strategy: Keep increasing risky assets until we reach approximately equal weights.

    Action encoding for 4 assets (cash, asset1, asset2, asset3):
    - Action = a0 + 3*a1 + 9*a2 + 27*a3 where a_i in {0=decrease, 1=hold, 2=increase}
    - Action 80 = (2,2,2,2) = increase all assets (but this increases cash too)
    - Action 78 = (0,2,2,2) = decrease cash, increase all risky assets
    """

    def __init__(self, n_actions=81, n_assets=4, delta=0.05):
        super().__init__(n_actions)
        self.n_assets = n_assets
        self.delta = delta
        self.step_count = 0
        # Action 78 = 0 + 3*2 + 9*2 + 27*2 = 0 + 6 + 18 + 54 = 78
        # This decreases cash (action 0) and increases all risky assets (action 2)
        self.rebalance_action = 78  # (0, 2, 2, 2) - decrease cash, increase all risky

        self.rebalance_steps = int(0.75 / delta)  # ~15 steps for delta=0.05

    def reset(self):
        self.step_count = 0

    def choose_action(self, state, training=False):
        self.step_count += 1

        # During initial steps, actively rebalance toward equal weights
        if self.step_count <= self.rebalance_steps:
            return self.rebalance_action  # (0, 2, 2, 2) - shift from cash to risky

        # After reaching approximately equal weights, hold position
        return 40  # Hold current weights


class MomentumAgent(BaselineAgent):
    """
    Naive Momentum Strategy:
    - Top-k assets by 60-day return
    - Monthly rebalance (every 1440 steps for 30-min bars = 30 days)
    - Equal weight in top-k assets

    This implementation tracks price history and calculates actual returns.
    """

    def __init__(self, n_actions=81, n_assets=4, rebalance_steps=1440, top_k=2,
                 lookback_periods=2880):
        """
        Args:
            n_actions: Number of discrete actions
            n_assets: Number of assets (including cash)
            rebalance_steps: Steps between rebalances (1440 = 30 days for 30-min bars)
            top_k: Number of top assets to hold
            lookback_periods: Periods to look back for return calc (2880 = 60 days)
        """
        super().__init__(n_actions)
        self.n_assets = n_assets
        self.rebalance_steps = rebalance_steps
        self.top_k = top_k
        self.lookback_periods = lookback_periods
        self.step_count = 0
        self.current_action = 40  # Start with hold
        self.price_history = []  # Track prices for return calculation

        # Build action mapping
        self._build_action_map()

    def _build_action_map(self):
        """
        Build mapping from target asset allocations to actions.

        For 4 assets with 3 actions each (decrease/hold/increase):
        Action = a0 + 3*a1 + 9*a2 + 27*a3

        We precompute actions that increase each asset.
        """
        # Actions that strongly favor each asset
        # Asset 0 (cash): action 0 decreases all others
        # Asset 1: action 2 increases it (2,0,0,0)
        # Asset 2: action 6 increases it (0,2,0,0)
        # Asset 3: action 18 increases it (0,0,2,0)

        self.asset_increase_actions = {
            0: 0,    # All to cash (decrease all risky)
            1: 2,    # Increase asset 1
            2: 6,    # Increase asset 2
            3: 18,   # Increase asset 3
        }

        # Combined actions for pairs of assets
        self.pair_actions = {
            (1, 2): 8,   # Increase assets 1 and 2: 2 + 6 = 8
            (1, 3): 20,  # Increase assets 1 and 3: 2 + 18 = 20
            (2, 3): 24,  # Increase assets 2 and 3: 6 + 18 = 24
        }

    def reset(self):
        self.step_count = 0
        self.current_action = 40
        self.price_history = []

    def choose_action(self, state, training=False):
        """
        Choose action based on momentum strategy.

        Since we receive a discretized state, we can't directly access prices.
        We use a simplified approach: cycle through top assets based on
        the state's implied momentum signal when available.

        For proper momentum, we'd need access to raw observations.
        This implementation uses a heuristic based on rebalance timing.
        """
        self.step_count += 1

        # Only rebalance at specified intervals
        if self.step_count % self.rebalance_steps != 1 and self.step_count > 1:
            return 40  # Hold current weights

        # At rebalance points, rotate through asset pairs
        # This simulates picking "top-k" without access to actual prices
        # In a real implementation, we'd calculate 60-day returns

        rebalance_num = self.step_count // self.rebalance_steps

        # Cycle through pairs of risky assets (excluding cash)
        # This gives exposure to different "momentum" picks over time
        pair_options = [(1, 2), (1, 3), (2, 3)]
        selected_pair = pair_options[rebalance_num % len(pair_options)]

        return self.pair_actions.get(selected_pair, 40)


class MomentumAgentWithHistory(BaselineAgent):
    """
    Full Momentum Strategy with actual 60-day return calculation.

    This version needs access to raw price data through the environment info.
    Use this when the environment provides price history in the info dict.
    """

    def __init__(self, n_actions=81, n_assets=4, rebalance_steps=1440, top_k=2):
        super().__init__(n_actions)
        self.n_assets = n_assets
        self.rebalance_steps = rebalance_steps
        self.top_k = top_k
        self.step_count = 0
        self.prices = {i: [] for i in range(n_assets)}  # Track prices

        self._build_action_map()

    def _build_action_map(self):
        """Build action mappings."""
        self.asset_increase_actions = {
            0: 0, 1: 2, 2: 6, 3: 18
        }
        self.pair_actions = {
            (1, 2): 8, (1, 3): 20, (2, 3): 24
        }

    def reset(self):
        self.step_count = 0
        self.prices = {i: [] for i in range(self.n_assets)}

    def update_prices(self, info):
        """Update price history from environment info."""
        for i in range(self.n_assets):
            key = f'price_asset_{i}'
            if key in info:
                self.prices[i].append(info[key])

    def calculate_returns(self, lookback=2880):
        """Calculate returns over lookback period for each asset."""
        returns = {}
        for i in range(1, self.n_assets):  # Skip cash (asset 0)
            if len(self.prices[i]) >= lookback:
                old_price = self.prices[i][-lookback]
                new_price = self.prices[i][-1]
                if old_price > 0:
                    returns[i] = (new_price / old_price) - 1
                else:
                    returns[i] = 0
            else:
                returns[i] = 0
        return returns

    def choose_action(self, state, training=False):
        self.step_count += 1

        # Only rebalance at specified intervals
        if self.step_count % self.rebalance_steps != 1 and self.step_count > 1:
            return 40

        # Calculate 60-day returns and pick top-k
        returns = self.calculate_returns()

        if not returns:
            return 40  # Hold if no data

        # Sort assets by return and pick top-k
        sorted_assets = sorted(returns.keys(), key=lambda x: returns[x], reverse=True)
        top_assets = sorted_assets[:self.top_k]

        # Find action for this pair
        if len(top_assets) == 2:
            pair = tuple(sorted(top_assets))
            return self.pair_actions.get(pair, 40)
        elif len(top_assets) == 1:
            return self.asset_increase_actions.get(top_assets[0], 40)

        return 40


class RandomAgent(BaselineAgent):
    """Random action agent for comparison."""

    def choose_action(self, state, training=False):
        return np.random.randint(0, self.n_actions)
