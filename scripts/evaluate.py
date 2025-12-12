#!/usr/bin/env python3
"""
Evaluation Script

Evaluate a trained Q-Learning agent.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model models/q_table_final.npy --episodes 100
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agent import QLearningAgent
from src.environment import TaxiEnvironment
from src.evaluator import Evaluator
from src.utils import setup_logging, print_banner


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Q-Learning agent"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/q_table_final.npy',
        help='Path to trained model'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render episodes during evaluation'
    )
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logging
    setup_logging()
    
    # Print banner
    print_banner()
    
    print(f"Loading model from: {args.model}")
    print(f"Evaluating for {args.episodes} episodes\n")
    
    # Initialize environment
    render_mode = "human" if args.render else None
    env = TaxiEnvironment(render_mode=render_mode)
    
    # Initialize agent and load weights
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions
    )
    agent.load(args.model)
    
    # Set epsilon to 0 for pure exploitation
    agent.epsilon = 0.0
    
    # Initialize evaluator
    evaluator = Evaluator(agent, env, args.episodes)
    
    # Evaluate
    metrics = evaluator.evaluate(render=args.render)
    
    # Close environment
    env.close()
    
    return metrics


if __name__ == "__main__":
    main()