"""
Stable-Baselines3 Compatible Wrapper for Portfolio Environment.

- Action space: "Continuous version (for DDPG/TD3/PPO): target weight vector
  in a Box space, projected to the simplex."
- Reward: "one-step portfolio log-return minus λ·transaction_cost, with a
  small penalty on excess drawdown."
- Goal: "identical reward/costs/splits for an apples-to-apples stability comparison"

This wrapper:
1. Flattens Dict observation to Box (SB3 requirement)
2. Uses continuous Box action space with simplex projection
3. Applies same reward shaping as tabular agents
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SB3PortfolioWrapper(gym.Wrapper):
    """
    Wrapper to make PortfolioEnv compatible with Stable-Baselines3.

    Changes from base env:
    1. Observation: Flattens Dict to Box (SB3 doesn't support Dict natively)
    2. Action: Continuous Box [0,1]^n_assets, projected to simplex
    3. Reward: Shaped reward (log-return - λ·cost - drawdown_penalty)

    This ensures identical reward/cost handling as tabular agents for fair comparison.
    """

    def __init__(
        self,
        env,
        reward_scale=100.0,
        cost_penalty=0.1,
        drawdown_threshold=0.15,
        drawdown_penalty=0.05
    ):
        """
        Args:
            env: Base PortfolioEnv
            reward_scale: Multiplier for reward (same as tabular: 100)
            cost_penalty: λ multiplier for transaction costs (same as tabular: 0.1)
            drawdown_threshold: Drawdown threshold before penalty (same as tabular: 0.15)
            drawdown_penalty: Penalty multiplier for excess drawdown (same as tabular: 0.05)
        """
        super().__init__(env)

        # Reward shaping parameters (identical to tabular)
        self.reward_scale = reward_scale
        self.cost_penalty = cost_penalty
        self.drawdown_threshold = drawdown_threshold
        self.drawdown_penalty = drawdown_penalty

        # Track for drawdown calculation
        self.peak_value = 1.0

        # Get dimensions from base env
        self.n_assets = env.action_space.shape[0]  # Number of assets (including cash)

        # Create flattened observation space
        # Original obs is Dict with 'history' and 'weights'
        # history shape: (n_assets, window_length, n_features)
        # weights shape: (n_assets,)
        sample_obs = env.observation_space.sample()

        if isinstance(env.observation_space, spaces.Dict):
            history_shape = env.observation_space['history'].shape
            weights_shape = env.observation_space['weights'].shape

            # Flatten: history (flattened) + weights
            self.history_size = int(np.prod(history_shape))
            self.weights_size = int(np.prod(weights_shape))
            total_size = self.history_size + self.weights_size

            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(total_size,),
                dtype=np.float32
            )
        else:
            # Already flat, keep as-is
            self.observation_space = env.observation_space
            self.history_size = 0
            self.weights_size = 0

        # Continuous action space: target weights in [0, 1]
        # Will be projected to simplex (sum=1, all>=0)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )

    def _flatten_obs(self, obs):
        """Flatten Dict observation to Box for SB3 compatibility."""
        if isinstance(obs, dict):
            history_flat = obs['history'].flatten().astype(np.float32)
            weights_flat = obs['weights'].flatten().astype(np.float32)
            return np.concatenate([history_flat, weights_flat])
        return obs

    def _project_to_simplex(self, action):
        """
        Project action to probability simplex (weights >= 0, sum = 1).

        Uses simple normalization. For negative values, clip to 0 first.
        """
        # Clip to non-negative
        action = np.maximum(action, 0.0)

        # Normalize to sum to 1
        action_sum = np.sum(action)
        if action_sum > 1e-8:
            action = action / action_sum
        else:
            # If all zeros, use equal weights
            action = np.ones(self.n_assets) / self.n_assets

        return action.astype(np.float32)

    def _compute_shaped_reward(self, info, raw_reward):
        """
        Compute shaped reward identical to tabular agents (reward_shaper.py).

         "one-step portfolio log-return minus λ·transaction_cost,
        with a small penalty on excess drawdown"

        Reward = log_return * scale - λ * cost * scale - drawdown_penalty

        This MUST match reward_shaper.py exactly for fair comparison.
        """
        # Get log return from info (base env computes this)
        log_return = info.get('log_return', raw_reward)

        # Get transaction cost from info
        transaction_cost = info.get('cost', 0.0)

        # Update peak value for drawdown calculation
        current_value = info.get('portfolio_value', 1.0)
        if current_value > self.peak_value:
            self.peak_value = current_value

        # Calculate current drawdown (same formula as tabular)
        current_drawdown = (self.peak_value - current_value) / (self.peak_value + 1e-8)

        # Base reward: LOG-RETURN  - SAME AS TABULAR
        shaped_reward = log_return * self.reward_scale

        # Subtract transaction cost penalty: λ * cost - SAME AS TABULAR
        shaped_reward -= self.cost_penalty * transaction_cost * self.reward_scale

        # Apply drawdown penalty if exceeds threshold - SAME AS TABULAR
        if current_drawdown > self.drawdown_threshold:
            excess_drawdown = current_drawdown - self.drawdown_threshold
            shaped_reward -= excess_drawdown * self.drawdown_penalty * self.reward_scale

        return shaped_reward

    def reset(self, **kwargs):
        """Reset environment and return flattened observation."""
        obs, info = self.env.reset(**kwargs)
        self.peak_value = 1.0  # Reset peak tracking
        return self._flatten_obs(obs), info

    def step(self, action):
        """
        Execute action with simplex projection and reward shaping.

        Args:
            action: Continuous weights in [0, 1]^n_assets

        Returns:
            Flattened observation, shaped reward, terminated, truncated, info
        """
        # Project action to simplex ( "projected to the simplex")
        projected_action = self._project_to_simplex(action)

        # Step base environment
        obs, raw_reward, terminated, truncated, info = self.env.step(projected_action)

        # Compute shaped reward (identical to tabular agents)
        shaped_reward = self._compute_shaped_reward(info, raw_reward)

        # Flatten observation
        flat_obs = self._flatten_obs(obs)

        return flat_obs, shaped_reward, terminated, truncated, info


def create_sb3_env(df, config, random_reset=True):
    """
    Create SB3-compatible environment with identical config as tabular agents.

    Args:
        df: Price dataframe
        config: Configuration dict (same as tabular training)
        random_reset: Whether to use random start positions

    Returns:
        SB3-compatible wrapped environment
    """
    from rl_portfolio_management.environments import PortfolioEnv

    # Create base environment (same as tabular)
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

    # Wrap for SB3 compatibility with identical reward shaping
    env = SB3PortfolioWrapper(
        env,
        reward_scale=config['reward_scale'],
        cost_penalty=config['cost_penalty'],
        drawdown_threshold=config['drawdown_threshold'],
        drawdown_penalty=config['drawdown_penalty']
    )

    return env
