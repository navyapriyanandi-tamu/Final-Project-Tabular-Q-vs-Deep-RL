"""
Reward Shaper for Portfolio Environment.

Implements reward function :
"Reward (R): one-step portfolio log-return minus λ·transaction_cost,
with a small penalty on excess drawdown"

The environment already computes:
- log_return: one-step portfolio log-return (includes transaction costs in portfolio value)
- cost: transaction cost incurred at this step

We use log_return directly and add explicit cost penalty + drawdown penalty.
"""

import numpy as np
import gymnasium as gym


class RewardShaper(gym.Wrapper):
    """
    Shapes rewards using log-return with transaction cost and drawdown penalty.

    Reward = log_return * scale - λ * cost * scale - drawdown_penalty

    Parameters:
    -----------
    env : gym.Env
        The portfolio environment to wrap
    scale : float
        Scaling factor for rewards (default: 100)
        Higher values make the reward signal stronger.
    cost_penalty : float
        Lambda (λ) multiplier for transaction costs (default: 1.0)
        Adds explicit penalty to discourage excessive trading.
    drawdown_threshold : float
        Drawdown level above which penalty applies (default: 0.15 = 15%)
    drawdown_penalty : float
        Penalty multiplier for excess drawdown (default: 0.05)
    """

    def __init__(self, env, scale=100, cost_penalty=0.1,
                 drawdown_threshold=0.15, drawdown_penalty=0.05):
        super().__init__(env)
        self.scale = scale
        self.cost_penalty = cost_penalty
        self.drawdown_threshold = drawdown_threshold
        self.drawdown_penalty = drawdown_penalty
        self.peak_value = 1.0

    def reset(self, **kwargs):
        """Reset environment and track initial portfolio value."""
        obs, info = self.env.reset(**kwargs)
        self.peak_value = info.get('portfolio_value', 1.0)
        return obs, info

    def step(self, action):
        """
        Execute action and return shaped reward.

        Reward = log_return * scale - λ * cost * scale - drawdown_penalty

        Components:
        1. Log return: one-step portfolio log-return (from environment)
        2. Transaction cost penalty: λ * cost (explicit penalty for trading)
        3. Drawdown penalty: penalize excess drawdown beyond threshold
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        current_value = info.get('portfolio_value', 1.0)
        log_return = info.get('log_return', 0.0)
        transaction_cost = info.get('cost', 0.0)

        # Update peak value for drawdown calculation
        if current_value > self.peak_value:
            self.peak_value = current_value

        # Calculate current drawdown
        current_drawdown = (self.peak_value - current_value) / (self.peak_value + 1e-8)

        # Base reward: LOG-RETURN 
        shaped_reward = log_return * self.scale

        # Subtract transaction cost penalty: λ * cost 
        shaped_reward -= self.cost_penalty * transaction_cost * self.scale

        # Apply drawdown penalty if exceeds threshold 
        if current_drawdown > self.drawdown_threshold:
            excess_drawdown = current_drawdown - self.drawdown_threshold
            shaped_reward -= excess_drawdown * self.drawdown_penalty * self.scale

        return obs, shaped_reward, terminated, truncated, info
