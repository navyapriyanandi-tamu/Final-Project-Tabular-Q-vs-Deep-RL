"""
Tabular Q-Learning Agent for Portfolio Management.

Implements standard Q-learning with ε-greedy exploration and learning rate schedule.
"""

import numpy as np
from collections import defaultdict
import pickle


class QLearningAgent:
    """
    Tabular Q-Learning agent with learning rate schedule.

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

        # Initialize Q-table with zeros
        # Using defaultdict for memory efficiency (only stores visited states)
        self.Q = defaultdict(lambda: np.zeros(n_actions))

        # Training statistics
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilon_values': [],
            'lr_values': [],
            'portfolio_values': []
        }

    def choose_action(self, state, training=True):
        """
        Choose action using ε-greedy policy.

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
            # Exploit: best known action
            return np.argmax(self.Q[state])

    def learn(self, state, action, reward, next_state, done):
        """
        Update Q-value using Q-learning update rule.

        Q(s,a) = Q(s,a) + α * (r + γ * max Q(s',a') - Q(s,a))

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
        # Current Q-value
        current_q = self.Q[state][action]

        # Target Q-value
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.Q[next_state])

        # Update Q-value
        self.Q[state][action] = current_q + self.lr * (target_q - current_q)

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
                print(f"Episode {episode + 1}/{n_episodes} | "
                      f"Avg Reward: {avg_reward:.6f} | "
                      f"Avg Portfolio: {avg_portfolio:.4f} | "
                      f"Epsilon: {self.epsilon:.4f} | "
                      f"States visited: {len(self.Q)}")

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
        policy = {}
        for state in self.Q:
            policy[state] = np.argmax(self.Q[state])
        return policy

    def get_q_table_stats(self):
        """Get statistics about the Q-table."""
        if len(self.Q) == 0:
            return {'states_visited': 0}

        all_q_values = []
        for state in self.Q:
            all_q_values.extend(self.Q[state])

        return {
            'states_visited': len(self.Q),
            'coverage': len(self.Q) / self.n_states * 100,
            'mean_q_value': np.mean(all_q_values),
            'max_q_value': np.max(all_q_values),
            'min_q_value': np.min(all_q_values)
        }

    def save(self, filepath):
        """Save the agent to a file."""
        data = {
            'Q': dict(self.Q),
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

        self.Q = defaultdict(lambda: np.zeros(self.n_actions))
        self.Q.update(data['Q'])
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
