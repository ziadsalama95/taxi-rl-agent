#!/usr/bin/env python3
"""
Interactive Play Script

Play against or watch the trained agent.

Usage:
    python scripts/play.py
    python scripts/play.py --model models/q_table_final.npy --mode watch
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agent import QLearningAgent
from src.environment import TaxiEnvironment
from src.utils import print_banner


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive play with trained agent"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/q_table_final.npy',
        help='Path to trained model'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['watch', 'compare'],
        default='watch',
        help='Play mode: watch agent or compare with random'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=5,
        help='Number of episodes to run'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.3,
        help='Delay between steps (seconds)'
    )
    return parser.parse_args()


def wait_for_key_or_timeout(timeout: float = 5.0) -> bool:
    """
    Wait for key press while keeping pygame responsive.
    
    Args:
        timeout: Maximum wait time in seconds
        
    Returns:
        True if should continue, False if user wants to quit
    """
    try:
        import pygame
        
        print("\n  Press SPACE for next episode, Q to quit, or wait 5 seconds...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
                        return True
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        return False
            
            pygame.display.flip()
            time.sleep(0.05)
        
        return True
        
    except Exception:
        # Fallback if pygame not available
        print("\n  Continuing in 3 seconds...")
        time.sleep(3)
        return True


def process_pygame_events() -> bool:
    """
    Process pygame events to keep window responsive.
    
    Returns:
        False if user wants to quit, True otherwise
    """
    try:
        import pygame
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    return False
        return True
        
    except Exception:
        return True


def watch_agent(agent: QLearningAgent, env: TaxiEnvironment, 
                n_episodes: int, delay: float) -> None:
    """Watch the trained agent play."""
    
    print("\nWatch Mode: Observing trained agent")
    print("=" * 50)
    print("Controls: Q/ESC = Quit, SPACE/ENTER = Next Episode")
    
    for episode in range(1, n_episodes + 1):
        print(f"\nEpisode {episode}/{n_episodes}")
        print("-" * 30)
        
        state, _ = env.reset()
        total_reward = 0
        should_quit = False
        
        for step in range(100):
            # Process events to keep window responsive
            if not process_pygame_events():
                should_quit = True
                break
            
            env.render()
            time.sleep(delay)
            
            action = agent.select_action(state, training=False)
            action_name = env.get_action_name(action)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            print(f"  Step {step+1:2d}: {action_name:8s} | Reward: {reward:+3.0f}")
            
            state = next_state
            
            if terminated or truncated:
                env.render()
                if reward == 20:
                    print(f"\n  SUCCESS! Passenger delivered!")
                else:
                    print(f"\n  Episode ended")
                print(f"  Total Reward: {total_reward}")
                break
        
        if should_quit:
            print("\nQuitting...")
            break
        
        # Wait for user input between episodes (except last one)
        if episode < n_episodes:
            if not wait_for_key_or_timeout(timeout=10.0):
                print("\nQuitting...")
                break
    
    print("\nDemo complete!")


def compare_with_random(agent: QLearningAgent, env: TaxiEnvironment,
                        n_episodes: int) -> None:
    """Compare trained agent with random agent."""
    
    print("\nCompare Mode: Trained Agent vs Random Agent")
    print("=" * 50)
    
    trained_rewards = []
    random_rewards = []
    
    for episode in range(n_episodes):
        # Trained agent
        state, _ = env.reset()
        trained_reward = 0
        for _ in range(200):
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            trained_reward += reward
            state = next_state
            if terminated or truncated:
                break
        trained_rewards.append(trained_reward)
        
        # Random agent
        state, _ = env.reset()
        random_reward = 0
        for _ in range(200):
            action = env.sample_action()
            next_state, reward, terminated, truncated, _ = env.step(action)
            random_reward += reward
            state = next_state
            if terminated or truncated:
                break
        random_rewards.append(random_reward)
    
    # Print comparison
    print("\nResults:")
    print("-" * 40)
    print(f"  {'Metric':<20} {'Trained':<12} {'Random':<12}")
    print("-" * 40)
    print(f"  {'Mean Reward':<20} {sum(trained_rewards)/n_episodes:<12.2f} {sum(random_rewards)/n_episodes:<12.2f}")
    print(f"  {'Max Reward':<20} {max(trained_rewards):<12.0f} {max(random_rewards):<12.0f}")
    print(f"  {'Min Reward':<20} {min(trained_rewards):<12.0f} {min(random_rewards):<12.0f}")
    print("-" * 40)
    
    improvement = ((sum(trained_rewards) - sum(random_rewards)) / 
                   abs(sum(random_rewards))) * 100
    print(f"\n  Trained agent is {improvement:.1f}% better than random!")


def main():
    """Main function."""
    args = parse_args()
    
    # Print banner
    print_banner()
    
    # Initialize environment
    env = TaxiEnvironment(render_mode="human" if args.mode == "watch" else None)
    
    # Initialize and load agent
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions
    )
    agent.load(args.model)
    agent.epsilon = 0.0
    
    print(f"Loaded model from: {args.model}")
    
    try:
        if args.mode == "watch":
            watch_agent(agent, env, args.episodes, args.delay)
        else:
            compare_with_random(agent, env, args.episodes)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    main()