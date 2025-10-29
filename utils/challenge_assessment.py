import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ChallengeType(Enum):
    """Types of challenges that can be assessed"""
    DISTANCE_TO_GOAL = "distance_to_goal"
    ADVERSARY_PRESSURE = "adversary_pressure"
    OBSTACLE_DENSITY = "obstacle_density"
    COLLISION_RISK = "collision_risk"
    COMMUNICATION_COMPLEXITY = "communication_complexity"
    MULTI_AGENT_COORDINATION = "multi_agent_coordination"
    BOUNDARY_VIOLATION = "boundary_violation"
    TASK_COMPLEXITY = "task_complexity"


@dataclass
class ChallengeMetrics:
    """Container for challenge assessment metrics"""
    distance_to_goal: float = 0.0
    adversary_pressure: float = 0.0
    obstacle_density: float = 0.0
    collision_risk: float = 0.0
    communication_complexity: float = 0.0
    multi_agent_coordination: float = 0.0
    boundary_violation: float = 0.0
    task_complexity: float = 0.0
    
    def get_total_challenge(self, weights: Optional[Dict[ChallengeType, float]] = None) -> float:
        """Calculate total challenge score"""
        if weights is None:
            weights = {challenge_type: 1.0 for challenge_type in ChallengeType}
        
        total = 0.0
        for challenge_type, weight in weights.items():
            metric_value = getattr(self, challenge_type.value, 0.0)
            total += weight * metric_value
        
        return total


class ChallengeAssessor:
    """Assesses the challenge level of different scenarios and experiences"""
    
    def __init__(self, 
                 challenge_weights: Optional[Dict[ChallengeType, float]] = None,
                 min_challenge_threshold: float = 0.1,
                 max_challenge_threshold: float = 10.0,
                 reward_amplification_factor: float = 2.0):
        """
        Initialize the challenge assessor
        
        Args:
            challenge_weights: Weights for different challenge types
            min_challenge_threshold: Minimum challenge level to start weighting
            max_challenge_threshold: Maximum challenge level for normalization
            reward_amplification_factor: How much to amplify rewards for challenging experiences
        """
        self.challenge_weights = challenge_weights or {
            ChallengeType.DISTANCE_TO_GOAL: 1.0,
            ChallengeType.ADVERSARY_PRESSURE: 2.0,
            ChallengeType.OBSTACLE_DENSITY: 1.5,
            ChallengeType.COLLISION_RISK: 1.8,
            ChallengeType.COMMUNICATION_COMPLEXITY: 1.2,
            ChallengeType.MULTI_AGENT_COORDINATION: 1.5,
            ChallengeType.BOUNDARY_VIOLATION: 1.0,
            ChallengeType.TASK_COMPLEXITY: 1.3
        }
        self.min_challenge_threshold = min_challenge_threshold
        self.max_challenge_threshold = max_challenge_threshold
        self.reward_amplification_factor = reward_amplification_factor
        
        # Track challenge history for adaptive weighting
        self.challenge_history = []
        self.max_history_size = 1000
    
    def assess_navigation_challenge(self, agent, world, goal_entity=None) -> ChallengeMetrics:
        """Assess challenge for navigation scenarios"""
        metrics = ChallengeMetrics()
        
        # Distance to goal challenge
        if goal_entity:
            dist_to_goal = np.sqrt(np.sum(np.square(agent.state.p_pos - goal_entity.state.p_pos)))
            metrics.distance_to_goal = min(dist_to_goal / 2.0, 1.0)  # Normalize to [0, 1]
        
        # Obstacle density challenge
        if hasattr(world, 'obstacles') and world.obstacles:
            obstacle_dists = []
            for obstacle in world.obstacles:
                dist = np.sqrt(np.sum(np.square(agent.state.p_pos - obstacle.state.p_pos)))
                obstacle_dists.append(dist)
            
            if obstacle_dists:
                min_obstacle_dist = min(obstacle_dists)
                metrics.obstacle_density = max(0, 1.0 - min_obstacle_dist / 0.5)  # Higher when close to obstacles
        
        # Collision risk challenge
        collision_risk = 0.0
        for other_agent in world.agents:
            if other_agent != agent:
                dist = np.sqrt(np.sum(np.square(agent.state.p_pos - other_agent.state.p_pos)))
                if dist < 0.3:  # Close to other agents
                    collision_risk += (0.3 - dist) / 0.3
        metrics.collision_risk = min(collision_risk, 1.0)
        
        # Boundary violation challenge
        boundary_penalty = 0.0
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            if x > 0.9:
                boundary_penalty += (x - 0.9) * 10
        metrics.boundary_violation = min(boundary_penalty / 10.0, 1.0)
        
        return metrics
    
    def assess_adversary_challenge(self, agent, world) -> ChallengeMetrics:
        """Assess challenge for scenarios with adversaries"""
        metrics = ChallengeMetrics()
        
        # Adversary pressure challenge
        if hasattr(self, 'adversaries') and callable(self.adversaries):
            adversaries = self.adversaries(world)
            adversary_pressure = 0.0
            
            for adv in adversaries:
                dist = np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
                if dist < 0.5:  # Close to adversary
                    adversary_pressure += (0.5 - dist) / 0.5
            
            metrics.adversary_pressure = min(adversary_pressure, 1.0)
        
        # Distance to goal challenge
        if hasattr(agent, 'goal_a') and agent.goal_a:
            dist_to_goal = np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
            metrics.distance_to_goal = min(dist_to_goal / 2.0, 1.0)
        
        # Multi-agent coordination challenge
        if len(world.agents) > 1:
            # Assess coordination difficulty based on agent positions and goals
            coordination_difficulty = 0.0
            for other_agent in world.agents:
                if other_agent != agent:
                    dist = np.sqrt(np.sum(np.square(agent.state.p_pos - other_agent.state.p_pos)))
                    if dist < 0.3:  # Agents are close, coordination needed
                        coordination_difficulty += 0.5
            
            metrics.multi_agent_coordination = min(coordination_difficulty, 1.0)
        
        return metrics
    
    def assess_communication_challenge(self, agent, world) -> ChallengeMetrics:
        """Assess challenge for communication-based scenarios"""
        metrics = ChallengeMetrics()
        
        # Communication complexity challenge
        if hasattr(world, 'dim_c') and world.dim_c > 0:
            # Higher communication dimension = more complex
            metrics.communication_complexity = min(world.dim_c / 10.0, 1.0)
        
        # Task complexity challenge
        if hasattr(world, 'landmarks') and world.landmarks:
            # More landmarks = more complex task
            metrics.task_complexity = min(len(world.landmarks) / 5.0, 1.0)
        
        return metrics
    
    def calculate_reward_weight(self, challenge_metrics: ChallengeMetrics) -> float:
        """Calculate reward weight based on challenge level"""
        total_challenge = challenge_metrics.get_total_challenge(self.challenge_weights)
        
        # Normalize challenge to [0, 1]
        normalized_challenge = np.clip(
            (total_challenge - self.min_challenge_threshold) / 
            (self.max_challenge_threshold - self.min_challenge_threshold),
            0.0, 1.0
        )
        
        # Calculate weight using exponential amplification
        if normalized_challenge > 0:
            weight = 1.0 + (self.reward_amplification_factor - 1.0) * normalized_challenge
        else:
            weight = 1.0
        
        # Update challenge history
        self.challenge_history.append(total_challenge)
        if len(self.challenge_history) > self.max_history_size:
            self.challenge_history.pop(0)
        
        return weight
    
    def get_adaptive_weights(self) -> Dict[ChallengeType, float]:
        """Get adaptive weights based on challenge history"""
        if not self.challenge_history:
            return self.challenge_weights
        
        # Calculate average challenge level
        avg_challenge = np.mean(self.challenge_history)
        
        # Adjust weights based on average challenge
        adaptive_weights = {}
        for challenge_type, base_weight in self.challenge_weights.items():
            if avg_challenge > self.max_challenge_threshold * 0.7:
                # High challenge environment - increase weights for complex challenges
                if challenge_type in [ChallengeType.ADVERSARY_PRESSURE, 
                                    ChallengeType.MULTI_AGENT_COORDINATION,
                                    ChallengeType.COMMUNICATION_COMPLEXITY]:
                    adaptive_weights[challenge_type] = base_weight * 1.5
                else:
                    adaptive_weights[challenge_type] = base_weight
            elif avg_challenge < self.min_challenge_threshold * 2:
                # Low challenge environment - increase weights for basic challenges
                if challenge_type in [ChallengeType.DISTANCE_TO_GOAL, 
                                    ChallengeType.BOUNDARY_VIOLATION]:
                    adaptive_weights[challenge_type] = base_weight * 1.3
                else:
                    adaptive_weights[challenge_type] = base_weight
            else:
                adaptive_weights[challenge_type] = base_weight
        
        return adaptive_weights


class RewardWeightingWrapper:
    """Wrapper to add challenge-based reward weighting to existing scenarios"""
    
    def __init__(self, 
                 scenario,
                 challenge_assessor: ChallengeAssessor,
                 enable_adaptive_weights: bool = True):
        """
        Initialize the reward weighting wrapper
        
        Args:
            scenario: The original scenario object
            challenge_assessor: ChallengeAssessor instance
            enable_adaptive_weights: Whether to use adaptive weights
        """
        self.scenario = scenario
        self.challenge_assessor = challenge_assessor
        self.enable_adaptive_weights = enable_adaptive_weights
        
        # Store original reward method
        self.original_reward = scenario.reward
        self.original_agent_reward = getattr(scenario, 'agent_reward', None)
        self.original_adversary_reward = getattr(scenario, 'adversary_reward', None)
        
        # Override reward methods
        self._override_reward_methods()
    
    def _override_reward_methods(self):
        """Override the reward methods with weighted versions"""
        def weighted_reward(agent, world):
            # Get original reward
            original_reward = self.original_reward(agent, world)
            
            # Assess challenge
            if hasattr(self.scenario, 'adversaries') and callable(self.scenario.adversaries):
                challenge_metrics = self.challenge_assessor.assess_adversary_challenge(agent, world)
            elif 'navigation' in self.scenario.__class__.__name__.lower():
                goal_entity = getattr(agent, 'goal_a', None) or getattr(agent, 'goal', None)
                challenge_metrics = self.challenge_assessor.assess_navigation_challenge(agent, world, goal_entity)
            else:
                challenge_metrics = self.challenge_assessor.assess_communication_challenge(agent, world)
            
            # Calculate weight
            if self.enable_adaptive_weights:
                self.challenge_assessor.challenge_weights = self.challenge_assessor.get_adaptive_weights()
            
            weight = self.challenge_assessor.calculate_reward_weight(challenge_metrics)
            
            # Apply weight to reward
            weighted_reward = original_reward * weight
            
            return weighted_reward
        
        # Override the main reward method
        self.scenario.reward = weighted_reward
        
        # Also override specific reward methods if they exist
        if self.original_agent_reward:
            def weighted_agent_reward(agent, world):
                original_reward = self.original_agent_reward(agent, world)
                challenge_metrics = self.challenge_assessor.assess_adversary_challenge(agent, world)
                weight = self.challenge_assessor.calculate_reward_weight(challenge_metrics)
                return original_reward * weight
            
            self.scenario.agent_reward = weighted_agent_reward
        
        if self.original_adversary_reward:
            def weighted_adversary_reward(agent, world):
                original_reward = self.original_adversary_reward(agent, world)
                challenge_metrics = self.challenge_assessor.assess_adversary_challenge(agent, world)
                weight = self.challenge_assessor.calculate_reward_weight(challenge_metrics)
                return original_reward * weight
            
            self.scenario.adversary_reward = weighted_adversary_reward


def create_challenge_weighted_scenario(scenario_class, 
                                     challenge_weights: Optional[Dict[ChallengeType, float]] = None,
                                     enable_adaptive_weights: bool = True,
                                     **kwargs) -> RewardWeightingWrapper:
    """
    Factory function to create a challenge-weighted scenario
    
    Args:
        scenario_class: The scenario class to wrap
        challenge_weights: Custom challenge weights
        enable_adaptive_weights: Whether to use adaptive weights
        **kwargs: Arguments to pass to scenario constructor
    
    Returns:
        RewardWeightingWrapper instance
    """
    # Create the original scenario
    scenario = scenario_class(**kwargs)
    
    # Create challenge assessor
    assessor = ChallengeAssessor(challenge_weights=challenge_weights)
    
    # Create and return the wrapper
    return RewardWeightingWrapper(scenario, assessor, enable_adaptive_weights) 