"""
Double Q-Learning Agent for Portfolio Management.

Double Q-Learning reduces overestimation bias by maintaining two Q-tables
and decoupling action selection from action evaluation.


Key difference from standard Q-learning:
- Standard: Q(s,a) += α * (r + γ * max_a' Q(s',a') - Q(s,a))
- Double:   Q1(s,a) += α * (r + γ * Q2(s', argmax_a' Q1(s',a')) - Q1(s,a))
            (and vice versa, randomly choosing which to update)

This prevents the maximization bias that occurs when the same Q-values
are used both to select and evaluate actions.
"""

import numpy as np
from collections import defaultdict
import pickle


class DoubleQLearningAgent:
    """
    Double Q-Learning agent to reduce overestimation bias.

    Maintains two Q-tables (Q1 and Q2). On each update:
    1. Randomly choose which table to update (50/50)
    2. Use the OTHER table to evaluate the selected action

    Parameters:
    -----------
    n_states : int
        Number of discrete states
    n_actions : int
        Number of discrete actions
    learning_rate : float
        Initial learning rate α (default: 0.1)
    lr_min : float
        Minimum learning rate (default: 0.01)
    lr_decay : float
        Decay rate for learning rate (default: 0.9999)
    discount_factor : float
        Discount factor γ (default: 0.99)
    epsilon : float
        Initial exploration rate ε (default: 1.0)
    epsilon_min : float
        Minimum exploration rate (default: 0.01)
    epsilon_decay : float
        Decay rate for epsilon after each episode (default: 0.995)
    """

    def __init__(
        self,
        n_states,
        n_actions,
        learning_rate=0.1,
        lr_min=0.01,
        lr_decay=0.9999,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.lr_initial = learning_rate
        self.lr_min = lr_min
        self.lr_decay = lr_decay
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Initialize TWO Q-tables (the key difference from standard Q-learning)
        self.Q1 = defaultdict(lambda: np.zeros(n_actions))
        self.Q2 = defaultdict(lambda: np.zeros(n_actions))

        # Training statistics
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilon_values': [],
            'lr_values': [],
            'portfolio_values': [],
            'q1_updates': 0,
            'q2_updates': 0,
        }

    def choose_action(self, state, training=True):
        """
        Choose action using ε-greedy policy based on combined Q-values.

        For action selection, we use the sum of Q1 and Q2 (or average)
        to get a better estimate of the true Q-values.

        Parameters:
        -----------
        state : int
            Current discrete state
        training : bool
            If True, use ε-greedy; if False, use greedy (no exploration)

        Returns:
        --------
        action : int
            Chosen action
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best action based on combined Q-values
            combined_q = self.Q1[state] + self.Q2[state]
            return np.argmax(combined_q)

    def learn(self, state, action, reward, next_state, done):
        """
        Update Q-value using Double Q-learning update rule.

        Randomly choose which Q-table to update:
        - If updating Q1: use Q1 to select action, Q2 to evaluate
        - If updating Q2: use Q2 to select action, Q1 to evaluate

        This decoupling reduces overestimation bias.

        Parameters:
        -----------
        state : int
            Current state
        action : int
            Action taken
        reward : float
            Reward received
        next_state : int
            Next state
        done : bool
            Whether episode is finished
        """
        # Randomly choose which Q-table to update (50/50 probability)
        if np.random.random() < 0.5:
            # Update Q1 using Q2 for evaluation
            self._update_q1(state, action, reward, next_state, done)
            self.training_history['q1_updates'] += 1
        else:
            # Update Q2 using Q1 for evaluation
            self._update_q2(state, action, reward, next_state, done)
            self.training_history['q2_updates'] += 1

    def _update_q1(self, state, action, reward, next_state, done):
        """
        Update Q1 table.

        Q1(s,a) += α * (r + γ * Q2(s', argmax_a' Q1(s',a')) - Q1(s,a))

        - Action selection: argmax over Q1
        - Action evaluation: Q2
        """
        current_q = self.Q1[state][action]

        if done:
            target_q = reward
        else:
            # Select best action using Q1
            best_action = np.argmax(self.Q1[next_state])
            # Evaluate using Q2
            target_q = reward + self.gamma * self.Q2[next_state][best_action]

        # Update Q1
        self.Q1[state][action] = current_q + self.lr * (target_q - current_q)

    def _update_q2(self, state, action, reward, next_state, done):
        """
        Update Q2 table.

        Q2(s,a) += α * (r + γ * Q1(s', argmax_a' Q2(s',a')) - Q2(s,a))

        - Action selection: argmax over Q2
        - Action evaluation: Q1
        """
        current_q = self.Q2[state][action]

        if done:
            target_q = reward
        else:
            # Select best action using Q2
            best_action = np.argmax(self.Q2[next_state])
            # Evaluate using Q1
            target_q = reward + self.gamma * self.Q1[next_state][best_action]

        # Update Q2
        self.Q2[state][action] = current_q + self.lr * (target_q - current_q)

    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def decay_lr(self):
        """Decay learning rate after each episode."""
        self.lr = max(self.lr_min, self.lr * self.lr_decay)

    def train(self, env, n_episodes=1000, max_steps=256, verbose=True, print_every=100):
        """
        Train the agent on the environment.

        Parameters:
        -----------
        env : gym.Env
            The environment (should be wrapped with StateDiscretizer and DiscreteActionWrapper)
        n_episodes : int
            Number of training episodes
        max_steps : int
            Maximum steps per episode
        verbose : bool
            Whether to print progress
        print_every : int
            Print progress every N episodes

        Returns:
        --------
        training_history : dict
            Dictionary containing training statistics
        """
        for episode in range(n_episodes):
            state, info = env.reset()
            episode_reward = 0
            episode_length = 0

            for step in range(max_steps):
                # Choose action
                action = self.choose_action(state, training=True)

                # Take action
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Learn from experience
                self.learn(state, action, reward, next_state, done)

                # Update statistics
                episode_reward += reward
                episode_length += 1
                state = next_state

                if done:
                    break

            # Decay epsilon and learning rate
            self.decay_epsilon()
            self.decay_lr()

            # Record statistics
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            self.training_history['epsilon_values'].append(self.epsilon)
            self.training_history['lr_values'].append(self.lr)
            self.training_history['portfolio_values'].append(info.get('portfolio_value', 0))

            # Print progress
            if verbose and (episode + 1) % print_every == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-print_every:])
                avg_portfolio = np.mean(self.training_history['portfolio_values'][-print_every:])
                q1_updates = self.training_history['q1_updates']
                q2_updates = self.training_history['q2_updates']
                print(f"Episode {episode + 1}/{n_episodes} | "
                      f"Avg Reward: {avg_reward:.4f} | "
                      f"Avg Portfolio: {avg_portfolio:.4f} | "
                      f"ε: {self.epsilon:.4f} | "
                      f"Q1/Q2 updates: {q1_updates}/{q2_updates}")

        return self.training_history

    def evaluate(self, env, n_episodes=10, max_steps=256):
        """
        Evaluate the trained agent (no exploration).

        Parameters:
        -----------
        env : gym.Env
            The environment
        n_episodes : int
            Number of evaluation episodes
        max_steps : int
            Maximum steps per episode

        Returns:
        --------
        results : dict
            Evaluation statistics
        """
        total_rewards = []
        portfolio_values = []

        for episode in range(n_episodes):
            state, info = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                # Choose action (greedy, no exploration)
                action = self.choose_action(state, training=False)

                # Take action
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                state = next_state

                if done:
                    break

            total_rewards.append(episode_reward)
            portfolio_values.append(info.get('portfolio_value', 0))

        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_portfolio_value': np.mean(portfolio_values),
            'std_portfolio_value': np.std(portfolio_values),
            'all_rewards': total_rewards,
            'all_portfolio_values': portfolio_values
        }

    def get_policy(self):
        """
        Get the learned policy (best action for each visited state).

        Returns:
        --------
        policy : dict
            Mapping from state to best action
        """
        # Combine Q1 and Q2 for policy extraction
        all_states = set(self.Q1.keys()) | set(self.Q2.keys())
        policy = {}
        for state in all_states:
            combined_q = self.Q1[state] + self.Q2[state]
            policy[state] = np.argmax(combined_q)
        return policy

    def get_q_table_stats(self):
        """Get statistics about the Q-tables."""
        all_states = set(self.Q1.keys()) | set(self.Q2.keys())

        if len(all_states) == 0:
            return {'states_visited': 0}

        all_q1_values = []
        all_q2_values = []
        for state in self.Q1:
            all_q1_values.extend(self.Q1[state])
        for state in self.Q2:
            all_q2_values.extend(self.Q2[state])

        # Compute overestimation metric: compare max Q values
        overestimation_samples = []
        for state in all_states:
            q1_max = np.max(self.Q1[state])
            q2_max = np.max(self.Q2[state])
            combined_max = np.max(self.Q1[state] + self.Q2[state]) / 2
            # Positive means single table overestimates vs combined
            overestimation_samples.append((q1_max + q2_max) / 2 - combined_max)

        return {
            'states_visited': len(all_states),
            'coverage': len(all_states) / self.n_states * 100,
            'q1_states': len(self.Q1),
            'q2_states': len(self.Q2),
            'mean_q1_value': np.mean(all_q1_values) if all_q1_values else 0,
            'mean_q2_value': np.mean(all_q2_values) if all_q2_values else 0,
            'q1_updates': self.training_history['q1_updates'],
            'q2_updates': self.training_history['q2_updates'],
            'mean_overestimation': np.mean(overestimation_samples) if overestimation_samples else 0,
        }

    def save(self, filepath):
        """Save the agent to a file."""
        data = {
            'Q1': dict(self.Q1),
            'Q2': dict(self.Q2),
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'lr': self.lr,
            'lr_initial': self.lr_initial,
            'lr_min': self.lr_min,
            'lr_decay': self.lr_decay,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'training_history': self.training_history
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath):
        """Load the agent from a file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.Q1 = defaultdict(lambda: np.zeros(self.n_actions))
        self.Q1.update(data['Q1'])
        self.Q2 = defaultdict(lambda: np.zeros(self.n_actions))
        self.Q2.update(data['Q2'])
        self.n_states = data['n_states']
        self.n_actions = data['n_actions']
        self.lr = data['lr']
        self.lr_initial = data.get('lr_initial', self.lr)
        self.lr_min = data.get('lr_min', 0.01)
        self.lr_decay = data.get('lr_decay', 0.9999)
        self.gamma = data['gamma']
        self.epsilon = data['epsilon']
        self.epsilon_min = data['epsilon_min']
        self.epsilon_decay = data['epsilon_decay']
        self.training_history = data['training_history']

    def reset(self):
        """Reset for new episode (needed for evaluation compatibility)."""
        pass  # No episode-level state to reset
