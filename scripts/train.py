#!/usr/bin/env python3
"""
Training Script

Train the Q-Learning agent on the Taxi environment.

Usage:
    python scripts/train.py
    python scripts/train.py --episodes 20000
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agent import QLearningAgent
from src.environment import TaxiEnvironment
from src.trainer import Trainer
from src.utils import (
    setup_logging,
    load_config,
    print_banner,
    print_config
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Q-Learning agent on the Taxi environment"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=None,
        help='Number of training episodes (overrides config)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.episodes:
        config['training']['episodes'] = args.episodes
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    # Setup logging
    setup_logging(config['paths']['logs_dir'])
    
    # Print banner and config
    print_banner()
    print_config(config)
    
    # Initialize environment
    env = TaxiEnvironment()
    
    # Initialize agent
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=config['training']['learning_rate'],
        discount_factor=config['training']['discount_factor'],
        epsilon_start=config['training']['epsilon_start'],
        epsilon_end=config['training']['epsilon_end'],
        epsilon_decay=config['training']['epsilon_decay']
    )
    
    # Initialize trainer
    trainer = Trainer(agent, env, config)
    
    # Train
    print("Starting training...\n")
    metrics = trainer.train()
    
    # Summary
    print("\nTraining Complete!")
    print(f"   Final Success Rate: {sum(metrics['successes'][-100:]) / 100 * 100:.1f}%")
    print(f"   Final Avg Reward:   {sum(metrics['rewards'][-100:]) / 100:.2f}")
    print(f"   Model saved to:     models/q_table_final.npy")
    print(f"   Logs saved to:      logs/")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main()