"""
Taxi RL Agent Package

A professional reinforcement learning project for training
an agent to solve the Taxi-v3 environment.
"""

from .agent import QLearningAgent
from .environment import TaxiEnvironment
from .trainer import Trainer
from .evaluator import Evaluator

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    "QLearningAgent",
    "TaxiEnvironment", 
    "Trainer",
    "Evaluator"
]