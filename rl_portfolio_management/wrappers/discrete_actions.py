"""
Discrete Action Wrapper for Portfolio Environment.

Converts the continuous portfolio weight actions into discrete actions
where each asset can be adjusted by {-Δ, 0, +Δ}.

This enables tabular Q-learning on the portfolio environment.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DiscreteActionWrapper(gym.ActionWrapper):
    """
    Wraps a continuous portfolio environment with discrete actions.

    Each asset (including cash) can be adjusted by one of three actions:
        0: Decrease weight by delta (-Δ)
        1: Keep weight unchanged (0)
        2: Increase weight by delta (+Δ)

    Actions are flattened to a single integer for easy Q-table indexing.
    Total actions = 3^n_assets (e.g., 3^4 = 81 for 4 assets)

    Parameters:
    -----------
    env : gym.Env
        The continuous portfolio environment to wrap
    delta : float
        The amount to adjust weights by (default: 0.05 = 5%)

    """

    def __init__(self, env, delta=0.05):
        super().__init__(env)
        self.delta = delta
        self.n_assets = env.action_space.shape[0]
        self.n_actions = 3 ** self.n_assets

        # Single discrete action space
        self.action_space = spaces.Discrete(self.n_actions)

        # Store current weights (initialized after reset)
        self._current_weights = None

    def reset(self, **kwargs):
        """Reset and initialize weights."""
        obs, info = self.env.reset(**kwargs)
        self._current_weights = obs['weights'].copy()
        return obs, info

    def index_to_action(self, index):
        """
        Convert single integer index to per-asset action array.

        Returns array where each element is in {0, 1, 2}:
            0 = decrease by delta
            1 = no change
            2 = increase by delta
        """
        action = []
        for _ in range(self.n_assets):
            action.append(index % 3)
            index //= 3
        return np.array(action)

    def action(self, flat_action):
        """
        Convert flat discrete action to continuous portfolio weights.

        Parameters:
        -----------
        flat_action : int
            Single integer action in range [0, 3^n_assets)

        Returns:
        --------
        continuous_action : np.ndarray
            Portfolio weights that sum to 1
        """
        # Convert flat action to per-asset actions
        discrete_action = self.index_to_action(flat_action)

        # Map {0, 1, 2} to {-1, 0, +1}
        action_deltas = discrete_action - 1

        # Apply changes to current weights
        if self._current_weights is None:
            self._current_weights = np.ones(self.n_assets) / self.n_assets

        new_weights = self._current_weights + action_deltas * self.delta

        # Project to simplex: clip to [0, 1] and normalize
        new_weights = np.clip(new_weights, 0.0, 1.0)
        weight_sum = new_weights.sum()
        if weight_sum > 0:
            new_weights = new_weights / weight_sum
        else:
            # If all weights are 0, default to cash
            new_weights = np.zeros(self.n_assets)
            new_weights[0] = 1.0

        return new_weights.astype(np.float32)

    def step(self, flat_action):
        """Take a step with discrete action."""
        continuous_action = self.action(flat_action)
        obs, reward, terminated, truncated, info = self.env.step(continuous_action)

        # Update current weights
        self._current_weights = obs['weights'].copy()

        # Add action info for debugging
        info['discrete_action'] = self.index_to_action(flat_action)
        info['continuous_action'] = continuous_action

        return obs, reward, terminated, truncated, info
