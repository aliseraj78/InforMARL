import argparse
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.getcwd()))

from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from utils.challenge_assessment import ChallengeAssessor, ChallengeType


class Scenario(BaseScenario):
    """
    Navigation scenario with integrated challenge-based reward weighting.
    
    This scenario extends the basic navigation scenario by automatically
    weighting rewards based on the difficulty of the current situation.
    """
    
    def __init__(self, 
                 enable_challenge_weighting: bool = True,
                 challenge_weights: dict = None,
                 reward_amplification_factor: float = 2.0):
        """
        Initialize the scenario with challenge weighting
        
        Args:
            enable_challenge_weighting: Whether to enable challenge-based weighting
            challenge_weights: Custom weights for different challenge types
            reward_amplification_factor: How much to amplify rewards for challenging situations
        """
        super().__init__()
        
        self.enable_challenge_weighting = enable_challenge_weighting
        
        # Initialize challenge assessor if weighting is enabled
        if self.enable_challenge_weighting:
            # Default weights for navigation scenarios
            default_weights = {
                ChallengeType.DISTANCE_TO_GOAL: 1.5,      # High weight for goal distance
                ChallengeType.OBSTACLE_DENSITY: 2.0,      # Very high weight for obstacles
                ChallengeType.COLLISION_RISK: 1.8,        # High weight for collision risk
                ChallengeType.BOUNDARY_VIOLATION: 1.2,    # Moderate weight for boundary
                ChallengeType.ADVERSARY_PRESSURE: 0.0,    # No adversaries in navigation
                ChallengeType.COMMUNICATION_COMPLEXITY: 0.0,  # No communication
                ChallengeType.MULTI_AGENT_COORDINATION: 1.0,  # Moderate coordination
                ChallengeType.TASK_COMPLEXITY: 1.0        # Base task complexity
            }
            
            # Override with custom weights if provided
            if challenge_weights:
                default_weights.update(challenge_weights)
            
            self.challenge_assessor = ChallengeAssessor(
                challenge_weights=default_weights,
                min_challenge_threshold=0.1,
                max_challenge_threshold=8.0,
                reward_amplification_factor=reward_amplification_factor
            )
        
        # Scenario parameters
        self.min_dist_thresh = 0.1
        self.goal_rew = 10
        self.collision_rew = 10
        self.use_dones = True
        self.obs_type = "global"
        self.use_comm = False
        self.max_edge_dist = 0.5
        self.num_nbd_entities = 5

    def make_world(self, args: argparse.Namespace) -> World:
        """Create the world with agents, landmarks, and obstacles"""
        world = World()
        
        # Set world properties
        world.dim_c = 2
        world.dim_p = 2
        
        # Get parameters from args
        num_agents = args.num_agents if hasattr(args, 'num_agents') else 3
        num_landmarks = args.num_landmarks if hasattr(args, 'num_landmarks') else 3
        num_obstacles = args.num_obstacles if hasattr(args, 'num_obstacles') else 3
        
        # Add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent {i}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
            agent.accel = 4.0
            agent.max_speed = 1.3
            agent.id = i
        
        # Add landmarks (goals)
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f"landmark {i}"
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.1
            landmark.boundary = False
            landmark.id = i
        
        # Add obstacles
        world.obstacles = [Landmark() for i in range(num_obstacles)]
        for i, obstacle in enumerate(world.obstacles):
            obstacle.name = f"obstacle {i}"
            obstacle.collide = True
            obstacle.movable = False
            obstacle.size = 0.1
            obstacle.boundary = False
            obstacle.id = i
        
        # Set initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world: World) -> None:
        """Reset the world state"""
        # Assign colors
        world.assign_agent_colors()
        world.assign_landmark_colors()
        
        # Set random initial states for agents
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        
        # Set random positions for landmarks
        for landmark in world.landmarks:
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        
        # Set random positions for obstacles
        for obstacle in world.obstacles:
            obstacle.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            obstacle.state.p_vel = np.zeros(world.dim_p)
        
        # Assign goals to agents (one-to-one mapping)
        for i, agent in enumerate(world.agents):
            goal_idx = i % len(world.landmarks)
            agent.goal_a = world.landmarks[goal_idx]

    def assess_challenge(self, agent: Agent, world: World) -> float:
        """Assess the challenge level for the current agent state"""
        if not self.enable_challenge_weighting:
            return 1.0  # No weighting
        
        # Get challenge metrics
        goal_entity = getattr(agent, 'goal_a', None) or getattr(agent, 'goal', None)
        challenge_metrics = self.challenge_assessor.assess_navigation_challenge(
            agent, world, goal_entity
        )
        
        # Calculate reward weight
        weight = self.challenge_assessor.calculate_reward_weight(challenge_metrics)
        
        return weight

    def reward(self, agent: Agent, world: World) -> float:
        """Calculate reward with optional challenge weighting"""
        # Calculate base reward
        base_reward = self._calculate_base_reward(agent, world)
        
        # Apply challenge weighting if enabled
        if self.enable_challenge_weighting:
            challenge_weight = self.assess_challenge(agent, world)
            weighted_reward = base_reward * challenge_weight
            
            # Store challenge information for debugging
            if hasattr(agent, 'last_challenge_weight'):
                agent.last_challenge_weight = challenge_weight
            if hasattr(agent, 'last_base_reward'):
                agent.last_base_reward = base_reward
            
            return weighted_reward
        else:
            return base_reward

    def _calculate_base_reward(self, agent: Agent, world: World) -> float:
        """Calculate the base reward without challenge weighting"""
        rew = 0
        
        # Get agent's goal
        agents_goal = world.get_entity(entity_type="landmark", id=agent.id)
        if agents_goal is None:
            # Fallback to assigned goal
            agents_goal = getattr(agent, 'goal_a', None)
        
        if agents_goal:
            # Distance-based reward
            dist_to_goal = np.sqrt(
                np.sum(np.square(agent.state.p_pos - agents_goal.state.p_pos))
            )
            
            if dist_to_goal < self.min_dist_thresh:
                rew += self.goal_rew
            else:
                rew -= dist_to_goal
        
        # Collision penalties
        if agent.collide:
            # Agent-agent collisions
            for a in world.agents:
                if a.id == agent.id:
                    continue
                if self.is_collision(a, agent):
                    rew -= self.collision_rew
            
            # Agent-obstacle collisions
            if self.is_obstacle_collision(
                pos=agent.state.p_pos, entity_size=agent.size, world=world
            ):
                rew -= self.collision_rew
        
        return rew

    def is_collision(self, agent1: Agent, agent2: Agent) -> bool:
        """Check if two agents are colliding"""
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return dist < dist_min

    def is_obstacle_collision(self, pos, entity_size: float, world: World) -> bool:
        """Check if an entity is colliding with any obstacle"""
        for obstacle in world.obstacles:
            delta_pos = pos - obstacle.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            dist_min = entity_size + obstacle.size
            if dist < dist_min:
                return True
        return False

    def done(self, agent: Agent, world: World) -> bool:
        """Check if the episode is done for this agent"""
        if self.use_dones:
            landmark = world.get_entity("landmark", agent.id)
            if landmark is None:
                landmark = getattr(agent, 'goal_a', None)
            
            if landmark:
                dist = np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos)))
                if dist < self.min_dist_thresh:
                    return True
            return False
        else:
            # Episode ends only when world length is reached
            if hasattr(world, 'current_time_step') and hasattr(world, 'world_length'):
                return world.current_time_step >= world.world_length
            return False

    def observation(self, agent: Agent, world: World) -> np.ndarray:
        """Get observation for the agent"""
        # Get agent's own information
        agent_vel, agent_pos = agent.state.p_vel, agent.state.p_pos
        
        # Get goal position
        agents_goal = world.get_entity("landmark", agent.id)
        if agents_goal is None:
            agents_goal = getattr(agent, 'goal_a', None)
        
        goal_pos = np.zeros(world.dim_p)
        if agents_goal:
            goal_pos = agents_goal.state.p_pos - agent_pos
        
        # Get other entities' positions
        other_agents_pos, obstacle_pos = [], []
        
        if self.obs_type in ["global", "nbd"]:
            # Other agents
            for other in world.agents:
                if other is agent:
                    continue
                other_agents_pos.append(np.array(other.state.p_pos - agent_pos))
            
            # Obstacles
            for obstacle in world.obstacles:
                obstacle_pos.append(np.array(obstacle.state.p_pos - agent_pos))
        
        # Combine all positions
        other_pos = np.array(other_agents_pos + obstacle_pos)
        
        # Apply neighborhood filtering if needed
        if self.obs_type == "nbd" and len(other_pos) > 0:
            # Calculate distances
            dist_mag = []
            for pos in other_pos:
                dist_mag.append(np.linalg.norm(pos))
            
            # Sort by distance
            if dist_mag:
                dist_mag_sort, dist_sort_idx = np.sort(dist_mag), np.argsort(dist_mag)
                other_pos = other_pos[dist_sort_idx, :]
                
                # Filter by max edge distance
                filter_mask = np.array(dist_mag_sort) < self.max_edge_dist
                filter_mask = np.expand_dims(filter_mask, axis=1)
                filter_mask = np.repeat(filter_mask, axis=1, repeats=other_pos.shape[1])
                other_pos = other_pos * filter_mask
                
                # Limit number of entities
                other_pos = other_pos[:self.num_nbd_entities, :]
        
        # Flatten and concatenate
        other_pos = other_pos.flatten()
        
        # Add communication if enabled
        if self.use_comm:
            comm = []
            for other in world.agents:
                if other is agent:
                    continue
                comm.append(other.state.c)
            comm_array = np.array(comm).flatten()
            return np.concatenate([agent_vel, agent_pos, goal_pos, other_pos, comm_array])
        else:
            return np.concatenate([agent_vel, agent_pos, goal_pos, other_pos])

    def info_callback(self, agent: Agent, world: World) -> dict:
        """Get additional information for the agent"""
        info = {}
        
        if self.enable_challenge_weighting:
            # Add challenge information
            challenge_weight = self.assess_challenge(agent, world)
            info['challenge_weight'] = challenge_weight
            info['base_reward'] = getattr(agent, 'last_base_reward', 0.0)
            info['weighted_reward'] = getattr(agent, 'last_challenge_weight', 1.0) * getattr(agent, 'last_base_reward', 0.0)
        
        return info

    class Args:
        """Default arguments for the scenario"""
        def __init__(self):
            self.num_agents = 3
            self.num_landmarks = 3
            self.num_obstacles = 3 