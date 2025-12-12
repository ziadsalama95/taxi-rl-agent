"""
Q-Learning Agent Module

Implements a tabular Q-Learning agent with epsilon-greedy exploration.
"""

import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class QLearningAgent:
    """
    Tabular Q-Learning Agent for discrete state and action spaces.
    
    Attributes:
        n_states: Number of states in the environment
        n_actions: Number of possible actions
        learning_rate: Learning rate (alpha) for Q-value updates
        discount_factor: Discount factor (gamma) for future rewards
        epsilon: Current exploration rate
        q_table: Q-value table of shape (n_states, n_actions)
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        """
        Initialize the Q-Learning agent.
        
        Args:
            n_states: Number of states in the environment
            n_actions: Number of possible actions
            learning_rate: Learning rate for Q-value updates
            discount_factor: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions))
        
        # Statistics
        self.training_steps = 0
        
        logger.info(
            f"Initialized Q-Learning Agent: "
            f"states={n_states}, actions={n_actions}, "
            f"lr={learning_rate}, gamma={discount_factor}"
        )
    
    def select_action(self, state: int, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether the agent is in training mode
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return int(np.argmax(self.q_table[state]))
    
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> float:
        """
        Update Q-value using the Q-Learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            
        Returns:
            TD error (for logging purposes)
        """
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error
        self.training_steps += 1
        
        return td_error
    
    def decay_epsilon(self) -> None:
        """Decay epsilon after each episode."""
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )
    
    def save(self, filepath: str) -> None:
        """Save the Q-table to a file."""
        np.save(filepath, self.q_table)
        logger.info(f"Saved Q-table to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load the Q-table from a file."""
        self.q_table = np.load(filepath)
        logger.info(f"Loaded Q-table from {filepath}")
    
    def get_policy(self) -> np.ndarray:
        """Get the current greedy policy."""
        return np.argmax(self.q_table, axis=1)
    
    def get_state_values(self) -> np.ndarray:
        """Get the value of each state (max Q-value)."""
        return np.max(self.q_table, axis=1)
    
    def __repr__(self) -> str:
        return (
            f"QLearningAgent(states={self.n_states}, actions={self.n_actions}, "
            f"epsilon={self.epsilon:.4f}, steps={self.training_steps})"
        )