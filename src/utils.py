"""
Utility Module

Provides helper functions and classes for the project.
"""

import os
import yaml
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 4)


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    """Setup logging configuration."""
    create_directory(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_directory(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


class MetricsLogger:
    """Logger for training metrics and visualizations."""
    
    def __init__(self, log_dir: str):
        """Initialize the metrics logger."""
        self.log_dir = log_dir
        create_directory(log_dir)
    
    def save_metrics(self, metrics: Dict[str, List]) -> None:
        """Save metrics to CSV file."""
        import pandas as pd
        
        df = pd.DataFrame(metrics)
        filepath = os.path.join(self.log_dir, "training_metrics.csv")
        df.to_csv(filepath, index_label='episode')
        logging.info(f"Saved metrics to {filepath}")
    
    def plot_training_curves(
        self,
        rewards: List[float],
        lengths: List[int],
        window: int = 100
    ) -> None:
        """Plot and save training curves."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot 1: Episode Rewards
        ax1 = axes[0]
        ax1.plot(rewards, alpha=0.3, color='blue', label='Raw')
        ax1.plot(
            self._smooth(rewards, window),
            color='blue',
            linewidth=2,
            label=f'Smoothed (w={window})'
        )
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Episode Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Episode Lengths
        ax2 = axes[1]
        ax2.plot(lengths, alpha=0.3, color='green', label='Raw')
        ax2.plot(
            self._smooth(lengths, window),
            color='green',
            linewidth=2,
            label=f'Smoothed (w={window})'
        )
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_title('Episode Length')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Reward Distribution
        ax3 = axes[2]
        ax3.hist(rewards, bins=50, color='purple', alpha=0.7, edgecolor='black')
        ax3.axvline(
            np.mean(rewards),
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Mean: {np.mean(rewards):.2f}'
        )
        ax3.set_xlabel('Reward')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Reward Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.log_dir, "training_curves.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved training curves to {filepath}")
    
    @staticmethod
    def _smooth(data: List[float], window: int) -> np.ndarray:
        """Apply moving average smoothing."""
        if len(data) < window:
            return np.array(data)
        
        smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
        padding = np.full(window - 1, smoothed[0])
        return np.concatenate([padding, smoothed])


def print_banner() -> None:
    """Print project banner."""
    banner = """
    ============================================================
    
                    TAXI RL AGENT
    
        Reinforcement Learning for the Taxi-v3 Environment
    
    ============================================================
    """
    print(banner)


def print_config(config: Dict[str, Any]) -> None:
    """Print configuration summary."""
    print("\nConfiguration:")
    print("-" * 40)
    print(f"  Environment:    {config['environment']['name']}")
    print(f"  Episodes:       {config['training']['episodes']}")
    print(f"  Learning Rate:  {config['training']['learning_rate']}")
    print(f"  Discount:       {config['training']['discount_factor']}")
    print(f"  Epsilon Start:  {config['training']['epsilon_start']}")
    print(f"  Epsilon End:    {config['training']['epsilon_end']}")
    print(f"  Epsilon Decay:  {config['training']['epsilon_decay']}")
    print("-" * 40 + "\n")