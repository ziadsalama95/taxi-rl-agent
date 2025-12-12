"""
Environment Wrapper Module

Provides a clean wrapper around the Gymnasium Taxi environment.
"""

import gymnasium as gym
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TaxiEnvironment:
    """
    Wrapper for the Gymnasium Taxi-v3 environment.
    
    The Taxi Problem:
    - 5x5 grid world
    - 4 designated locations (R, G, Y, B)
    - Task: Pick up passenger and deliver to destination
    
    State Space: 500 discrete states
        - 25 taxi positions (5x5 grid)
        - 5 passenger locations (4 locations + in taxi)
        - 4 destinations
        
    Action Space: 6 discrete actions
        - 0: Move South
        - 1: Move North
        - 2: Move East
        - 3: Move West
        - 4: Pickup passenger
        - 5: Drop off passenger
        
    Rewards:
        - +20: Successful delivery
        - -1: Each time step
        - -10: Illegal pickup/dropoff
    """
    
    ACTION_NAMES = {
        0: "South",
        1: "North",
        2: "East",
        3: "West",
        4: "Pickup",
        5: "Dropoff"
    }
    
    LOCATION_NAMES = ["Red", "Green", "Yellow", "Blue"]
    
    def __init__(self, render_mode: Optional[str] = None):
        """
        Initialize the Taxi environment.
        
        Args:
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        self.env = gym.make("Taxi-v3", render_mode=render_mode)
        self.render_mode = render_mode
        
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        
        logger.info(
            f"Initialized Taxi Environment: "
            f"states={self.n_states}, actions={self.n_actions}"
        )
    
    def reset(self) -> Tuple[int, Dict[str, Any]]:
        """Reset the environment."""
        state, info = self.env.reset()
        return int(state), info
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        next_state, reward, terminated, truncated, info = self.env.step(action)
        return int(next_state), float(reward), terminated, truncated, info
    
    def render(self) -> Optional[Any]:
        """Render the environment."""
        return self.env.render()
    
    def close(self) -> None:
        """Close the environment."""
        self.env.close()
    
    def decode_state(self, state: int) -> Tuple[int, int, int, int]:
        """Decode a state number into its components."""
        taxi_row = state // 100
        taxi_col = (state % 100) // 20
        passenger_loc = (state % 20) // 4
        destination = state % 4
        return taxi_row, taxi_col, passenger_loc, destination
    
    def get_state_description(self, state: int) -> str:
        """Get a human-readable description of a state."""
        taxi_row, taxi_col, pass_loc, dest = self.decode_state(state)
        
        if pass_loc == 4:
            passenger = "In Taxi"
        else:
            passenger = f"At {self.LOCATION_NAMES[pass_loc]}"
            
        destination = self.LOCATION_NAMES[dest]
        
        return (
            f"Taxi: ({taxi_row}, {taxi_col}) | "
            f"Passenger: {passenger} | "
            f"Destination: {destination}"
        )
    
    def get_action_name(self, action: int) -> str:
        """Get the name of an action."""
        return self.ACTION_NAMES.get(action, "Unknown")
    
    def sample_action(self) -> int:
        """Sample a random action."""
        return self.env.action_space.sample()
    
    def __repr__(self) -> str:
        return f"TaxiEnvironment(states={self.n_states}, actions={self.n_actions})"