#!/usr/bin/env python3
"""
Training script for challenge-weighted MARL scenarios.

This script demonstrates how to train agents using challenge-based reward weighting
to give more importance to more challenging experiences.
"""

import argparse
import os
import sys
import numpy as np
import torch
from typing import Dict, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.challenge_assessment import ChallengeAssessor, ChallengeType, RewardWeightingWrapper
from multiagent.custom_scenarios.navigation_challenge_weighted import Scenario as ChallengeWeightedNavigation
from multiagent.custom_scenarios.navigation import Scenario as StandardNavigation
from multiagent.custom_scenarios.simple_adversary import Scenario as AdversaryScenario
from multiagent.custom_scenarios.simple_tag import Scenario as TagScenario


def create_challenge_weighted_environment(scenario_name: str, args, 
                                        enable_weighting: bool = True,
                                        challenge_weights: Dict = None,
                                        amplification_factor: float = 2.0):
    """Create an environment with optional challenge weighting"""
    
    if scenario_name == "navigation":
        if enable_weighting:
            # Use the built-in challenge-weighted scenario
            scenario = ChallengeWeightedNavigation(
                enable_challenge_weighting=True,
                challenge_weights=challenge_weights,
                reward_amplification_factor=amplification_factor
            )
        else:
            # Use standard navigation with wrapper
            scenario = StandardNavigation()
            if challenge_weights:
                assessor = ChallengeAssessor(
                    challenge_weights=challenge_weights,
                    reward_amplification_factor=amplification_factor
                )
                scenario = RewardWeightingWrapper(scenario, assessor)
        
        scenario.make_world(args)
        
    elif scenario_name == "adversary":
        scenario = AdversaryScenario()
        scenario.make_world(args)
        
        if enable_weighting:
            # Define adversary-specific weights
            adv_weights = challenge_weights or {
                ChallengeType.ADVERSARY_PRESSURE: 3.0,
                ChallengeType.COLLISION_RISK: 2.0,
                ChallengeType.MULTI_AGENT_COORDINATION: 2.5,
                ChallengeType.DISTANCE_TO_GOAL: 1.0,
                ChallengeType.BOUNDARY_VIOLATION: 1.5,
                ChallengeType.OBSTACLE_DENSITY: 0.0,
                ChallengeType.COMMUNICATION_COMPLEXITY: 0.0,
                ChallengeType.TASK_COMPLEXITY: 1.5
            }
            
            assessor = ChallengeAssessor(
                challenge_weights=adv_weights,
                reward_amplification_factor=amplification_factor
            )
            scenario = RewardWeightingWrapper(scenario, assessor)
    
    elif scenario_name == "tag":
        scenario = TagScenario()
        scenario.make_world(args)
        
        if enable_weighting:
            # Define tag-specific weights
            tag_weights = challenge_weights or {
                ChallengeType.ADVERSARY_PRESSURE: 2.5,
                ChallengeType.COLLISION_RISK: 2.2,
                ChallengeType.BOUNDARY_VIOLATION: 1.8,
                ChallengeType.MULTI_AGENT_COORDINATION: 1.5,
                ChallengeType.DISTANCE_TO_GOAL: 0.0,
                ChallengeType.OBSTACLE_DENSITY: 0.0,
                ChallengeType.COMMUNICATION_COMPLEXITY: 0.0,
                ChallengeType.TASK_COMPLEXITY: 1.2
            }
            
            assessor = ChallengeAssessor(
                challenge_weights=tag_weights,
                reward_amplification_factor=amplification_factor
            )
            scenario = RewardWeightingWrapper(scenario, assessor)
    
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    
    return scenario


def run_training_comparison(scenario_name: str, args, 
                          num_episodes: int = 1000,
                          enable_weighting: bool = True):
    """Run training comparison between weighted and unweighted scenarios"""
    
    print(f"\n=== Training Comparison: {scenario_name} ===")
    print(f"Challenge weighting: {'Enabled' if enable_weighting else 'Disabled'}")
    print(f"Episodes: {num_episodes}")
    
    # Create scenarios
    weighted_scenario = create_challenge_weighted_environment(
        scenario_name, args, enable_weighting=True
    )
    unweighted_scenario = create_challenge_weighted_environment(
        scenario_name, args, enable_weighting=False
    )
    
    # Training metrics
    weighted_rewards = []
    unweighted_rewards = []
    challenge_levels = []
    
    print("\nTraining with challenge weighting...")
    for episode in range(num_episodes // 2):
        # Reset world
        weighted_scenario.reset_world(weighted_scenario.world)
        
        episode_reward = 0.0
        episode_challenge = 0.0
        
        # Run episode
        for step in range(50):
            step_reward = 0.0
            step_challenge = 0.0
            
            for agent in weighted_scenario.world.agents:
                # Get reward
                reward = weighted_scenario.reward(agent, weighted_scenario.world)
                step_reward += reward
                
                # Get challenge level if available
                if hasattr(weighted_scenario, 'challenge_assessor'):
                    if scenario_name == "navigation":
                        goal_entity = getattr(agent, 'goal_a', None)
                        challenge_metrics = weighted_scenario.challenge_assessor.assess_navigation_challenge(
                            agent, weighted_scenario.world, goal_entity
                        )
                    else:
                        challenge_metrics = weighted_scenario.challenge_assessor.assess_adversary_challenge(
                            agent, weighted_scenario.world
                        )
                    step_challenge += challenge_metrics.get_total_challenge()
            
            episode_reward += step_reward
            episode_challenge += step_challenge
            
            # Advance world
            weighted_scenario.world.step()
        
        weighted_rewards.append(episode_reward)
        challenge_levels.append(episode_challenge / (len(weighted_scenario.world.agents) * 50))
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(weighted_rewards[-100:])
            avg_challenge = np.mean(challenge_levels[-100:])
            print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.3f}, Avg Challenge = {avg_challenge:.3f}")
    
    print("\nTraining without challenge weighting...")
    for episode in range(num_episodes // 2):
        # Reset world
        unweighted_scenario.reset_world(unweighted_scenario.world)
        
        episode_reward = 0.0
        
        # Run episode
        for step in range(50):
            step_reward = 0.0
            
            for agent in unweighted_scenario.world.agents:
                # Get reward
                reward = unweighted_scenario.reward(agent, unweighted_scenario.world)
                step_reward += reward
            
            episode_reward += step_reward
            
            # Advance world
            unweighted_scenario.world.step()
        
        unweighted_rewards.append(episode_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(unweighted_rewards[-100:])
            print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.3f}")
    
    # Calculate final statistics
    print(f"\n--- Final Results ---")
    print(f"Weighted training:")
    print(f"  Final 100 episodes - Mean: {np.mean(weighted_rewards[-100:]):.3f}, Std: {np.std(weighted_rewards[-100:]):.3f}")
    print(f"  Average challenge level: {np.mean(challenge_levels):.3f}")
    
    print(f"\nUnweighted training:")
    print(f"  Final 100 episodes - Mean: {np.mean(unweighted_rewards[-100:]):.3f}, Std: {np.std(unweighted_rewards[-100:]):.3f}")
    
    # Performance improvement
    weighted_final = np.mean(weighted_rewards[-100:])
    unweighted_final = np.mean(unweighted_rewards[-100:])
    improvement = ((weighted_final - unweighted_final) / abs(unweighted_final)) * 100
    
    print(f"\nPerformance improvement: {improvement:.1f}%")
    
    return {
        'weighted_rewards': weighted_rewards,
        'unweighted_rewards': unweighted_rewards,
        'challenge_levels': challenge_levels,
        'improvement': improvement
    }


def run_adaptive_weighting_demo(scenario_name: str, args, num_episodes: int = 500):
    """Demonstrate adaptive challenge weighting"""
    
    print(f"\n=== Adaptive Weighting Demo: {scenario_name} ===")
    
    # Create scenario with adaptive weighting
    scenario = create_challenge_weighted_environment(
        scenario_name, args, enable_weighting=True
    )
    
    # Track adaptive weights over time
    weight_history = []
    reward_history = []
    challenge_history = []
    
    print("Training with adaptive challenge weighting...")
    for episode in range(num_episodes):
        # Reset world
        scenario.reset_world(scenario.world)
        
        episode_reward = 0.0
        episode_challenge = 0.0
        
        # Run episode
        for step in range(50):
            step_reward = 0.0
            step_challenge = 0.0
            
            for agent in scenario.world.agents:
                # Get reward
                reward = scenario.reward(agent, scenario.world)
                step_reward += reward
                
                # Get challenge level
                if hasattr(scenario, 'challenge_assessor'):
                    if scenario_name == "navigation":
                        goal_entity = getattr(agent, 'goal_a', None)
                        challenge_metrics = scenario.challenge_assessor.assess_navigation_challenge(
                            agent, scenario.world, goal_entity
                        )
                    else:
                        challenge_metrics = scenario.challenge_assessor.assess_adversary_challenge(
                            agent, scenario.world
                        )
                    step_challenge += challenge_metrics.get_total_challenge()
            
            episode_reward += step_reward
            episode_challenge += step_challenge
            
            # Advance world
            scenario.world.step()
        
        # Store episode data
        reward_history.append(episode_reward)
        challenge_history.append(episode_challenge / (len(scenario.world.agents) * 50))
        
        # Get current adaptive weights
        if hasattr(scenario, 'challenge_assessor'):
            current_weights = scenario.challenge_assessor.get_adaptive_weights()
            weight_history.append(current_weights)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(reward_history[-100:])
            avg_challenge = np.mean(challenge_history[-100:])
            print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.3f}, Avg Challenge = {avg_challenge:.3f}")
            
            # Show weight changes
            if len(weight_history) > 1:
                current_weights = weight_history[-1]
                print(f"  Current weights: {current_weights}")
    
    print(f"\n--- Adaptive Weighting Results ---")
    print(f"Final 100 episodes - Mean reward: {np.mean(reward_history[-100:]):.3f}")
    print(f"Average challenge level: {np.mean(challenge_history):.3f}")
    
    # Show weight evolution
    if weight_history:
        print(f"\nWeight evolution:")
        for challenge_type in ChallengeType:
            weights = [w.get(challenge_type, 0.0) for w in weight_history]
            print(f"  {challenge_type.value}: {np.mean(weights):.2f} ± {np.std(weights):.2f}")
    
    return {
        'reward_history': reward_history,
        'challenge_history': challenge_history,
        'weight_history': weight_history
    }


def main():
    parser = argparse.ArgumentParser(description="Challenge-weighted MARL training")
    parser.add_argument("--scenario", type=str, default="navigation", 
                       choices=["navigation", "adversary", "tag"],
                       help="Scenario to train on")
    parser.add_argument("--num_good_agents", type=int, default=2, help="Number of good agents")
    parser.add_argument("--num_adversaries", type=int, default=1, help="Number of adversaries")
    parser.add_argument("--num_landmarks", type=int, default=3, help="Number of landmarks")
    parser.add_argument("--num_obstacles", type=int, default=3, help="Number of obstacles")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--amplification", type=float, default=2.0, help="Reward amplification factor")
    parser.add_argument("--compare", action="store_true", help="Compare weighted vs unweighted training")
    parser.add_argument("--adaptive", action="store_true", help="Run adaptive weighting demo")
    
    args = parser.parse_args()
    
    print("Challenge-Weighted MARL Training")
    print("=" * 50)
    print(f"Scenario: {args.scenario}")
    print(f"Episodes: {args.episodes}")
    print(f"Amplification factor: {args.amplification}")
    
    if args.compare:
        results = run_training_comparison(args.scenario, args, args.episodes)
        print(f"\nTraining comparison completed!")
        print(f"Performance improvement: {results['improvement']:.1f}%")
    
    if args.adaptive:
        results = run_adaptive_weighting_demo(args.scenario, args, args.episodes)
        print(f"\nAdaptive weighting demo completed!")
    
    if not args.compare and not args.adaptive:
        print("Use --compare to compare weighted vs unweighted training")
        print("Use --adaptive to run adaptive weighting demo")
        print("Use --help for more options")


if __name__ == "__main__":
    main() 