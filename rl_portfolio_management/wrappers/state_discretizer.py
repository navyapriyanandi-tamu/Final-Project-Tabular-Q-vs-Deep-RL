"""
State Discretizer for Portfolio Environment.

Converts continuous observations into discrete states for tabular Q-learning.

- State space (S): recent per-asset returns/volatility, current portfolio weights,
  and a lightweight "regime" flag from rolling mean/vol.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class StateDiscretizer(gym.ObservationWrapper):
    """
    Wraps a portfolio environment to provide discrete state representations.

    Features extracted :
    1. Recent returns per asset (5 bins: strong_down/down/flat/up/strong_up)
    2. Volatility regime (3 bins: low/medium/high)
    3. Momentum (3 bins: bearish/neutral/bullish)
    4. Cash weight (3 bins: low/medium/high)

    Total states: 5^3 × 3 × 3 × 3 = 125 × 27 = 3,375 states

    Parameters:
    -----------
    env : gym.Env
        The portfolio environment to wrap
    """

    def __init__(self, env):
        super().__init__(env)

        # Get number of assets from unwrapped env
        self.n_assets = env.unwrapped.action_space.shape[0]
        self.n_tradeable = self.n_assets - 1  # Excluding cash

        # Return bins: 5 levels
        # strong down < -1%, down < -0.3%, flat, up > 0.3%, strong up > 1%
        self.return_bins = np.array([-0.01, -0.003, 0.003, 0.01])

        # Volatility bins: 3 levels (based on rolling std of returns)
        # low < 1%, medium < 3%, high >= 3%
        self.volatility_bins = np.array([0.01, 0.03])

        # Momentum bins: 3 levels (based on 5-period return)
        self.momentum_bins = np.array([-0.02, 0.02])

        # Weight bins: 3 levels for cash position
        self.weight_bins = np.array([0.33, 0.66])

        # State dimensions
        # Returns: 5 bins each for 3 assets = 5^3 = 125
        # Volatility: 3 bins (aggregate regime)
        # Momentum: 3 bins (aggregate)
        # Cash weight: 3 bins
        self.state_dims = (
            [5] * self.n_tradeable +  # Returns: 5 bins each
            [3] +                       # Volatility regime
            [3] +                       # Aggregate momentum
            [3]                         # Cash weight
        )
        self.n_states = int(np.prod(self.state_dims))
        self.observation_space = spaces.Discrete(self.n_states)

        # Store raw observation for debugging
        self._last_raw_obs = None

    def _extract_features(self, obs):
        """
        Extract discrete features from continuous observation.

        Parameters:
        -----------
        obs : dict
            Contains 'history' (price data) and 'weights' (portfolio weights)

        Returns:
        --------
        features : list of int
            List of discretized feature values
        """
        history = obs['history']
        weights = obs['weights']

        # Reshape history if flattened (MLP mode)
        if len(history.shape) == 1:
            n_features = 4  # open, high, low, close
            window_length = 50
            n_assets_history = len(history) // (window_length * n_features)
            history = history.reshape(n_assets_history, window_length, n_features)

        features = []
        momentums = []
        volatilities = []

        # ============================================
        # 1. RETURNS per asset (5 bins each)
        # ============================================
        for asset_idx in range(self.n_tradeable):
            close_prices = history[asset_idx, :, 0]

            # 1-period return
            if close_prices[-2] != 0:
                ret = (close_prices[-1] / close_prices[-2]) - 1
            else:
                ret = 0
            ret_bin = int(np.digitize(ret, self.return_bins))
            features.append(ret_bin)

            # 5-period return for momentum
            if close_prices[-6] != 0:
                mom = (close_prices[-1] / close_prices[-6]) - 1
            else:
                mom = 0
            momentums.append(mom)

            # Rolling volatility (std of last 10 returns)
            if len(close_prices) >= 11:
                returns = np.diff(close_prices[-11:]) / (close_prices[-11:-1] + 1e-8)
                vol = np.std(returns)
            else:
                vol = 0
            volatilities.append(vol)

        # ============================================
        # 2. VOLATILITY REGIME (3 bins)
        # ============================================
        avg_volatility = np.mean(volatilities)
        vol_bin = int(np.digitize(avg_volatility, self.volatility_bins))
        features.append(vol_bin)

        # ============================================
        # 3. AGGREGATE MOMENTUM (3 bins)
        # ============================================
        avg_momentum = np.mean(momentums)
        mom_bin = int(np.digitize(avg_momentum, self.momentum_bins))
        features.append(mom_bin)

        # ============================================
        # 4. CASH WEIGHT (3 bins)
        # ============================================
        cash_weight = weights[0]  # First weight is cash
        cash_bin = int(np.digitize(cash_weight, self.weight_bins))
        features.append(cash_bin)

        return features

    def _features_to_state(self, features):
        """Convert list of discrete features to single state integer."""
        state = 0
        multiplier = 1

        for feat, dim in zip(features, self.state_dims):
            # Clip to valid range
            feat = max(0, min(feat, dim - 1))
            state += feat * multiplier
            multiplier *= dim

        return state

    def _state_to_features(self, state):
        """Convert state integer back to feature list (for debugging)."""
        features = []
        for dim in self.state_dims:
            features.append(state % dim)
            state //= dim
        return features

    def observation(self, obs):
        """Convert continuous observation to discrete state."""
        self._last_raw_obs = obs
        features = self._extract_features(obs)
        state = self._features_to_state(features)
        return state

    def get_raw_observation(self):
        """Get the last raw (continuous) observation."""
        return self._last_raw_obs

    def get_n_states(self):
        """Return total number of discrete states."""
        return self.n_states

    def get_state_description(self, state):
        """
        Get human-readable description of a state (for debugging).

        Returns dict with:
        - returns: list of return bins per asset
        - volatility: 'low', 'medium', or 'high'
        - momentum: 'bearish', 'neutral', or 'bullish'
        - cash_weight: 'low', 'medium', or 'high'
        """
        features = self._state_to_features(state)

        returns = features[:self.n_tradeable]
        volatility = features[self.n_tradeable]
        momentum = features[self.n_tradeable + 1]
        cash_weight = features[self.n_tradeable + 2]

        return_names = ['strong_down', 'down', 'flat', 'up', 'strong_up']
        volatility_names = ['low', 'medium', 'high']
        momentum_names = ['bearish', 'neutral', 'bullish']
        weight_names = ['low', 'medium', 'high']

        return {
            'returns': [return_names[r] for r in returns],
            'volatility': volatility_names[volatility],
            'momentum': momentum_names[momentum],
            'cash_weight': weight_names[cash_weight]
        }
