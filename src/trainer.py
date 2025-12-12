"""
Trainer Module

Handles the training loop for the RL agent.
"""

import os
import time
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np
import logging

from .agent import QLearningAgent
from .environment import TaxiEnvironment
from .utils import MetricsLogger, create_directory

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for Q-Learning agent.
    
    Handles the complete training pipeline including:
    - Training loop
    - Metrics tracking
    - Model checkpointing
    - Logging
    """
    
    def __init__(
        self,
        agent: QLearningAgent,
        env: TaxiEnvironment,
        config: Dict
    ):
        """
        Initialize the trainer.
        
        Args:
            agent: Q-Learning agent to train
            env: Training environment
            config: Configuration dictionary
        """
        self.agent = agent
        self.env = env
        self.config = config
        
        # Training parameters
        self.n_episodes = config['training']['episodes']
        self.max_steps = config['training']['max_steps_per_episode']
        
        # Logging parameters
        self.log_interval = config['logging']['log_interval']
        self.save_interval = config['logging']['save_interval']
        self.metrics_window = config['logging']['metrics_window']
        
        # Paths
        self.models_dir = config['paths']['models_dir']
        self.logs_dir = config['paths']['logs_dir']
        
        # Create directories
        create_directory(self.models_dir)
        create_directory(self.logs_dir)
        
        # Metrics tracking
        self.metrics_logger = MetricsLogger(self.logs_dir)
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_successes: List[bool] = []
        
        # Rolling metrics
        self.recent_rewards = deque(maxlen=self.metrics_window)
        self.recent_lengths = deque(maxlen=self.metrics_window)
        self.recent_successes = deque(maxlen=self.metrics_window)
        
        logger.info(f"Initialized Trainer for {self.n_episodes} episodes")
    
    def train(self) -> Dict[str, List]:
        """Run the complete training loop."""
        logger.info("Starting training...")
        start_time = time.time()
        
        for episode in range(1, self.n_episodes + 1):
            episode_reward, episode_length, success = self._run_episode()
            
            # Track metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_successes.append(success)
            
            self.recent_rewards.append(episode_reward)
            self.recent_lengths.append(episode_length)
            self.recent_successes.append(success)
            
            # Decay exploration rate
            self.agent.decay_epsilon()
            
            # Logging
            if episode % self.log_interval == 0:
                self._log_progress(episode, start_time)
            
            # Save checkpoint
            if episode % self.save_interval == 0:
                self._save_checkpoint(episode)
        
        # Final save
        final_path = os.path.join(self.models_dir, "q_table_final.npy")
        self.agent.save(final_path)
        
        # Save metrics
        self.metrics_logger.save_metrics({
            'rewards': self.episode_rewards,
            'lengths': self.episode_lengths,
            'successes': self.episode_successes
        })
        
        # Plot results
        self.metrics_logger.plot_training_curves(
            self.episode_rewards,
            self.episode_lengths,
            self.metrics_window
        )
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        return {
            'rewards': self.episode_rewards,
            'lengths': self.episode_lengths,
            'successes': self.episode_successes
        }
    
    def _run_episode(self) -> Tuple[float, int, bool]:
        """Run a single training episode."""
        state, _ = self.env.reset()
        total_reward = 0.0
        
        for step in range(self.max_steps):
            action = self.agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            self.agent.update(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
            
            if done:
                success = reward == 20
                return total_reward, step + 1, success
        
        return total_reward, self.max_steps, False
    
    def _log_progress(self, episode: int, start_time: float) -> None:
        """Log training progress."""
        avg_reward = np.mean(self.recent_rewards)
        avg_length = np.mean(self.recent_lengths)
        success_rate = np.mean(self.recent_successes) * 100
        elapsed = time.time() - start_time
        
        logger.info(
            f"Episode {episode:>6}/{self.n_episodes} | "
            f"Avg Reward: {avg_reward:>7.2f} | "
            f"Avg Length: {avg_length:>6.1f} | "
            f"Success: {success_rate:>5.1f}% | "
            f"Epsilon: {self.agent.epsilon:.4f} | "
            f"Time: {elapsed:>6.1f}s"
        )
    
    def _save_checkpoint(self, episode: int) -> None:
        """Save a model checkpoint."""
        checkpoint_path = os.path.join(
            self.models_dir,
            f"q_table_episode_{episode}.npy"
        )
        self.agent.save(checkpoint_path)