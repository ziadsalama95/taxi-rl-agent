"""
Evaluator Module

Handles evaluation of trained agents.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from .agent import QLearningAgent
from .environment import TaxiEnvironment

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator class for testing trained agents.
    
    Provides:
    - Performance evaluation
    - Detailed episode analysis
    - Statistical summaries
    """
    
    def __init__(
        self,
        agent: QLearningAgent,
        env: TaxiEnvironment,
        n_episodes: int = 100
    ):
        """
        Initialize the evaluator.
        
        Args:
            agent: Trained Q-Learning agent
            env: Evaluation environment
            n_episodes: Number of evaluation episodes
        """
        self.agent = agent
        self.env = env
        self.n_episodes = n_episodes
        
        logger.info(f"Initialized Evaluator for {n_episodes} episodes")
    
    def evaluate(self, render: bool = False) -> Dict[str, float]:
        """Evaluate the agent's performance."""
        rewards = []
        lengths = []
        successes = []
        
        for episode in range(self.n_episodes):
            reward, length, success = self._run_episode(render)
            rewards.append(reward)
            lengths.append(length)
            successes.append(success)
        
        metrics = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'success_rate': np.mean(successes) * 100
        }
        
        self._print_summary(metrics)
        
        return metrics
    
    def _run_episode(self, render: bool = False) -> Tuple[float, int, bool]:
        """Run a single evaluation episode."""
        state, _ = self.env.reset()
        total_reward = 0.0
        
        for step in range(200):
            if render:
                self.env.render()
            
            action = self.agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            
            total_reward += reward
            state = next_state
            
            if terminated or truncated:
                success = reward == 20
                return total_reward, step + 1, success
        
        return total_reward, 200, False
    
    def _print_summary(self, metrics: Dict[str, float]) -> None:
        """Print evaluation summary."""
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"  Episodes Evaluated: {self.n_episodes}")
        print(f"  Success Rate:       {metrics['success_rate']:.1f}%")
        print(f"  Mean Reward:        {metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}")
        print(f"  Reward Range:       [{metrics['min_reward']:.0f}, {metrics['max_reward']:.0f}]")
        print(f"  Mean Episode Length:{metrics['mean_length']:.1f} +/- {metrics['std_length']:.1f}")
        print("=" * 50 + "\n")
    
    def run_demo(self, n_episodes: int = 3, delay: float = 0.5) -> None:
        """Run a visual demonstration of the agent."""
        import time
        
        print("\nRunning Demo Episodes...\n")
        
        demo_env = TaxiEnvironment(render_mode="human")
        
        for episode in range(n_episodes):
            print(f"\n--- Episode {episode + 1} ---")
            state, _ = demo_env.reset()
            total_reward = 0
            
            for step in range(100):
                demo_env.render()
                time.sleep(delay)
                
                action = self.agent.select_action(state, training=False)
                print(f"Step {step + 1}: {demo_env.get_action_name(action)}")
                
                next_state, reward, terminated, truncated, _ = demo_env.step(action)
                total_reward += reward
                state = next_state
                
                if terminated or truncated:
                    print(f"Episode finished! Total reward: {total_reward}")
                    break
        
        demo_env.close()