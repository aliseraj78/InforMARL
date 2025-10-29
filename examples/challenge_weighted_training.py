#!/usr/bin/env python3
"""
Example script demonstrating challenge-based reward weighting for MARL training.

This script shows how to:
1. Create a challenge assessor with custom weights
2. Wrap existing scenarios with reward weighting
3. Train agents with challenge-weighted rewards
4. Compare performance with and without challenge weighting
"""

import sys
import os
import argparse
import numpy as np
from typing import Dict

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.challenge_assessment import (
    ChallengeAssessor, 
    ChallengeType, 
    RewardWeightingWrapper,
    create_challenge_weighted_scenario
)
from multiagent.custom_scenarios.navigation import Scenario as NavigationScenario
from multiagent.custom_scenarios.simple_adversary import Scenario as AdversaryScenario
from multiagent.custom_scenarios.simple_tag import Scenario as TagScenario


def create_navigation_experiment(args):
    """Create a navigation experiment with challenge weighting"""
    
    # Define custom challenge weights for navigation
    navigation_weights = {
        ChallengeType.DISTANCE_TO_GOAL: 1.5,      # High weight for goal distance
        ChallengeType.OBSTACLE_DENSITY: 2.0,      # Very high weight for obstacles
        ChallengeType.COLLISION_RISK: 1.8,        # High weight for collision risk
        ChallengeType.BOUNDARY_VIOLATION: 1.2,    # Moderate weight for boundary
        ChallengeType.ADVERSARY_PRESSURE: 0.0,    # No adversaries in navigation
        ChallengeType.COMMUNICATION_COMPLEXITY: 0.0,  # No communication
        ChallengeType.MULTI_AGENT_COORDINATION: 1.0,  # Moderate coordination
        ChallengeType.TASK_COMPLEXITY: 1.0        # Base task complexity
    }
    
    # Create challenge assessor
    assessor = ChallengeAssessor(
        challenge_weights=navigation_weights,
        min_challenge_threshold=0.1,
        max_challenge_threshold=8.0,
        reward_amplification_factor=2.5
    )
    
    # Create the original scenario
    original_scenario = NavigationScenario()
    original_scenario.make_world(args)
    
    # Create weighted scenario
    weighted_scenario = RewardWeightingWrapper(
        original_scenario, 
        assessor, 
        enable_adaptive_weights=True
    )
    
    return weighted_scenario


def create_adversary_experiment(args):
    """Create an adversary experiment with challenge weighting"""
    
    # Define custom challenge weights for adversary scenarios
    adversary_weights = {
        ChallengeType.DISTANCE_TO_GOAL: 1.0,      # Base goal distance weight
        ChallengeType.ADVERSARY_PRESSURE: 3.0,    # Very high weight for adversary pressure
        ChallengeType.COLLISION_RISK: 2.0,        # High collision risk
        ChallengeType.MULTI_AGENT_COORDINATION: 2.5,  # Very high coordination need
        ChallengeType.BOUNDARY_VIOLATION: 1.5,    # Moderate boundary violation
        ChallengeType.OBSTACLE_DENSITY: 0.0,      # No obstacles in adversary scenarios
        ChallengeType.COMMUNICATION_COMPLEXITY: 0.0,  # No communication
        ChallengeType.TASK_COMPLEXITY: 1.5        # High task complexity
    }
    
    # Create challenge assessor
    assessor = ChallengeAssessor(
        challenge_weights=adversary_weights,
        min_challenge_threshold=0.2,
        max_challenge_threshold=12.0,
        reward_amplification_factor=3.0
    )
    
    # Create the original scenario
    original_scenario = AdversaryScenario()
    original_scenario.make_world(args)
    
    # Create weighted scenario
    weighted_scenario = RewardWeightingWrapper(
        original_scenario, 
        assessor, 
        enable_adaptive_weights=True
    )
    
    return weighted_scenario


def create_tag_experiment(args):
    """Create a tag experiment with challenge weighting"""
    
    # Define custom challenge weights for tag scenarios
    tag_weights = {
        ChallengeType.ADVERSARY_PRESSURE: 2.5,    # High adversary pressure
        ChallengeType.COLLISION_RISK: 2.2,        # High collision risk
        ChallengeType.BOUNDARY_VIOLATION: 1.8,    # High boundary violation penalty
        ChallengeType.MULTI_AGENT_COORDINATION: 1.5,  # Moderate coordination
        ChallengeType.DISTANCE_TO_GOAL: 0.0,      # No explicit goals in tag
        ChallengeType.OBSTACLE_DENSITY: 0.0,      # No obstacles
        ChallengeType.COMMUNICATION_COMPLEXITY: 0.0,  # No communication
        ChallengeType.TASK_COMPLEXITY: 1.2        # Moderate task complexity
    }
    
    # Create challenge assessor
    assessor = ChallengeAssessor(
        challenge_weights=tag_weights,
        min_challenge_threshold=0.3,
        max_challenge_threshold=10.0,
        reward_amplification_factor=2.8
    )
    
    # Create the original scenario
    original_scenario = TagScenario()
    original_scenario.make_world(args)
    
    # Create weighted scenario
    weighted_scenario = RewardWeightingWrapper(
        original_scenario, 
        assessor, 
        enable_adaptive_weights=True
    )
    
    return weighted_scenario


def run_challenge_assessment_demo(scenario_name: str, args):
    """Run a demonstration of challenge assessment"""
    
    print(f"\n=== Challenge Assessment Demo: {scenario_name} ===")
    
    # Create scenarios
    if scenario_name == "navigation":
        weighted_scenario = create_navigation_experiment(args)
        original_scenario = weighted_scenario.scenario
    elif scenario_name == "adversary":
        weighted_scenario = create_adversary_experiment(args)
        original_scenario = weighted_scenario.scenario
    elif scenario_name == "tag":
        weighted_scenario = create_tag_experiment(args)
        original_scenario = weighted_scenario.scenario
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    
    # Get the world
    world = original_scenario.world
    
    # Run a few steps to demonstrate challenge assessment
    print(f"World initialized with {len(world.agents)} agents")
    print(f"Challenge weights: {weighted_scenario.challenge_assessor.challenge_weights}")
    
    # Simulate a few steps and show challenge metrics
    for step in range(5):
        print(f"\n--- Step {step + 1} ---")
        
        # Get challenge metrics for each agent
        for i, agent in enumerate(world.agents):
            # Assess challenge
            if scenario_name == "navigation":
                goal_entity = getattr(agent, 'goal_a', None) or getattr(agent, 'goal', None)
                challenge_metrics = weighted_scenario.challenge_assessor.assess_navigation_challenge(
                    agent, world, goal_entity
                )
            elif scenario_name in ["adversary", "tag"]:
                challenge_metrics = weighted_scenario.challenge_assessor.assess_adversary_challenge(
                    agent, world
                )
            
            # Calculate reward weights
            weight = weighted_scenario.challenge_assessor.calculate_reward_weight(challenge_metrics)
            
            # Get original and weighted rewards
            original_reward = weighted_scenario.original_reward(agent, world)
            weighted_reward = weighted_scenario.scenario.reward(agent, world)
            
            print(f"Agent {i} ({'adversary' if agent.adversary else 'good'}):")
            print(f"  Position: {agent.state.p_pos}")
            print(f"  Challenge metrics: {challenge_metrics}")
            print(f"  Total challenge: {challenge_metrics.get_total_challenge():.3f}")
            print(f"  Reward weight: {weight:.3f}")
            print(f"  Original reward: {original_reward:.3f}")
            print(f"  Weighted reward: {weighted_reward:.3f}")
            print(f"  Amplification: {weighted_reward/original_reward:.2f}x")
        
        # Advance world state
        world.step()


def compare_performance_with_without_weighting(scenario_name: str, args, num_episodes: int = 100):
    """Compare performance with and without challenge weighting"""
    
    print(f"\n=== Performance Comparison: {scenario_name} ===")
    print(f"Running {num_episodes} episodes for each configuration...")
    
    # Create scenarios
    if scenario_name == "navigation":
        weighted_scenario = create_navigation_experiment(args)
        original_scenario = weighted_scenario.scenario
    elif scenario_name == "adversary":
        weighted_scenario = create_adversary_experiment(args)
        original_scenario = weighted_scenario.scenario
    elif scenario_name == "tag":
        weighted_scenario = create_tag_experiment(args)
        original_scenario = weighted_scenario.scenario
    
    # Track performance metrics
    original_rewards = []
    weighted_rewards = []
    challenge_levels = []
    
    for episode in range(num_episodes):
        # Reset world
        original_scenario.reset_world(original_scenario.world)
        
        episode_original_reward = 0.0
        episode_weighted_reward = 0.0
        episode_challenge = 0.0
        
        # Run episode
        for step in range(50):  # 50 steps per episode
            for agent in original_scenario.world.agents:
                # Get original reward
                orig_reward = weighted_scenario.original_reward(agent, original_scenario.world)
                episode_original_reward += orig_reward
                
                # Get weighted reward
                weighted_reward = weighted_scenario.scenario.reward(agent, original_scenario.world)
                episode_weighted_reward += weighted_reward
                
                # Get challenge level
                if scenario_name == "navigation":
                    goal_entity = getattr(agent, 'goal_a', None) or getattr(agent, 'goal', None)
                    challenge_metrics = weighted_scenario.challenge_assessor.assess_navigation_challenge(
                        agent, original_scenario.world, goal_entity
                    )
                else:
                    challenge_metrics = weighted_scenario.challenge_assessor.assess_adversary_challenge(
                        agent, original_scenario.world
                    )
                
                episode_challenge += challenge_metrics.get_total_challenge()
            
            # Advance world
            original_scenario.world.step()
        
        # Store episode results
        original_rewards.append(episode_original_reward)
        weighted_rewards.append(episode_weighted_reward)
        challenge_levels.append(episode_challenge / (len(original_scenario.world.agents) * 50))
        
        if (episode + 1) % 20 == 0:
            print(f"Completed {episode + 1}/{num_episodes} episodes")
    
    # Calculate statistics
    print(f"\n--- Results ---")
    print(f"Original rewards - Mean: {np.mean(original_rewards):.3f}, Std: {np.std(original_rewards):.3f}")
    print(f"Weighted rewards - Mean: {np.mean(weighted_rewards):.3f}, Std: {np.std(weighted_rewards):.3f}")
    print(f"Average challenge level: {np.mean(challenge_levels):.3f}")
    print(f"Reward amplification factor: {np.mean(weighted_rewards) / np.mean(original_rewards):.2f}x")
    
    # Show challenge distribution
    high_challenge_episodes = sum(1 for c in challenge_levels if c > np.mean(challenge_levels))
    print(f"High challenge episodes: {high_challenge_episodes}/{num_episodes} ({high_challenge_episodes/num_episodes*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Challenge-weighted MARL training demo")
    parser.add_argument("--scenario", type=str, default="navigation", 
                       choices=["navigation", "adversary", "tag"],
                       help="Scenario to run")
    parser.add_argument("--num_good_agents", type=int, default=2, help="Number of good agents")
    parser.add_argument("--num_adversaries", type=int, default=1, help="Number of adversaries")
    parser.add_argument("--num_landmarks", type=int, default=2, help="Number of landmarks")
    parser.add_argument("--num_obstacles", type=int, default=3, help="Number of obstacles")
    parser.add_argument("--demo", action="store_true", help="Run challenge assessment demo")
    parser.add_argument("--compare", action="store_true", help="Compare performance with/without weighting")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes for comparison")
    
    args = parser.parse_args()
    
    print("Challenge-Weighted MARL Training Demo")
    print("=" * 50)
    
    if args.demo:
        run_challenge_assessment_demo(args.scenario, args)
    
    if args.compare:
        compare_performance_with_without_weighting(args.scenario, args, args.episodes)
    
    if not args.demo and not args.compare:
        print("Use --demo to run challenge assessment demo")
        print("Use --compare to compare performance with/without weighting")
        print("Use --help for more options")


if __name__ == "__main__":
    main() 