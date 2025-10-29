"""
    Code for creating a multiagent environment with challenge-based reward weighting.
    This extends the original MPE environment to support challenge weighting parameters.
    
    Can be called by using, for example:
        env = MPEEnv('navigation_graph_challenge_weighted')
    After producing the env object, can be used similarly to an OpenAI gym
    environment.

    A policy using this environment must output actions in the form of a list
    for all agents. Each element of the list should be a numpy array,
    of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
    communication actions in this array. See environment.py for more details.
"""

import argparse
from typing import Dict
import numpy as np

from multiagent.custom_scenarios import load
from utils.challenge_assessment import ChallengeAssessor, ChallengeType


def MPEEnv(args: argparse.Namespace):
    """
    Creates a MultiAgentEnv object as env with optional challenge weighting.
    This can be used similar to a gym environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        args.scenario_name  :   name of the scenario from ./scenarios/ to be
                            Returns (without the .py extension)
        args.enable_challenge_weighting : whether to enable challenge weighting
        args.challenge_amplification_factor : how much to amplify rewards
        args.challenge_min_threshold : minimum challenge level to start weighting
        args.challenge_max_threshold : maximum challenge level for normalization
        benchmark       :   whether you want to produce benchmarking data
                        (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    """

    # load scenario from script
    scenario = load(args.scenario_name + ".py").Scenario()
    
    # Apply challenge weighting if enabled
    if hasattr(args, 'enable_challenge_weighting') and args.enable_challenge_weighting:
        # Create challenge weights from arguments
        challenge_weights = {}
        if hasattr(args, 'challenge_distance_weight'):
            challenge_weights[ChallengeType.DISTANCE_TO_GOAL] = args.challenge_distance_weight
        if hasattr(args, 'challenge_obstacle_weight'):
            challenge_weights[ChallengeType.OBSTACLE_DENSITY] = args.challenge_obstacle_weight
        if hasattr(args, 'challenge_collision_weight'):
            challenge_weights[ChallengeType.COLLISION_RISK] = args.challenge_collision_weight
        if hasattr(args, 'challenge_coordination_weight'):
            challenge_weights[ChallengeType.MULTI_AGENT_COORDINATION] = args.challenge_coordination_weight
        
        # Set default weights for missing challenge types
        default_weights = {
            ChallengeType.BOUNDARY_VIOLATION: 1.0,
            ChallengeType.ADVERSARY_PRESSURE: 0.0,
            ChallengeType.COMMUNICATION_COMPLEXITY: 0.0,
            ChallengeType.TASK_COMPLEXITY: 1.0
        }
        
        for challenge_type, default_weight in default_weights.items():
            if challenge_type not in challenge_weights:
                challenge_weights[challenge_type] = default_weight
        
        # Initialize scenario with challenge weighting
        if hasattr(scenario, '__init__'):
            # Check if scenario supports challenge weighting parameters
            if 'enable_challenge_weighting' in scenario.__init__.__code__.co_varnames:
                scenario = load(args.scenario_name + ".py").Scenario(
                    enable_challenge_weighting=True,
                    challenge_weights=challenge_weights,
                    reward_amplification_factor=getattr(args, 'challenge_amplification_factor', 2.0),
                    challenge_min_threshold=getattr(args, 'challenge_min_threshold', 0.1),
                    challenge_max_threshold=getattr(args, 'challenge_max_threshold', 8.0)
                )
            else:
                # Apply wrapper approach for scenarios that don't support challenge weighting
                from utils.challenge_assessment import RewardWeightingWrapper
                assessor = ChallengeAssessor(
                    challenge_weights=challenge_weights,
                    min_challenge_threshold=getattr(args, 'challenge_min_threshold', 0.1),
                    max_challenge_threshold=getattr(args, 'challenge_max_threshold', 8.0),
                    reward_amplification_factor=getattr(args, 'challenge_amplification_factor', 2.0)
                )
                scenario = RewardWeightingWrapper(scenario, assessor)
    
    # create world
    world = scenario.make_world(args=args)
    
    if args.algorithm_name in ["mappo", "rmappo"]:
        from multiagent.environment import MultiAgentPPOEnv as MultiAgentEnv
    else:
        from multiagent.environment import MultiAgentOffPolicyEnv as MultiAgentEnv
    
    # Create environment with appropriate callbacks
    if hasattr(scenario, 'scenario'):  # Wrapped scenario
        actual_scenario = scenario.scenario
        reward_callback = scenario.scenario.reward
    else:
        actual_scenario = scenario
        reward_callback = scenario.reward
    
    env = MultiAgentEnv(
        world=world,
        reset_callback=actual_scenario.reset_world,
        reward_callback=reward_callback,
        observation_callback=actual_scenario.observation,
        info_callback=actual_scenario.info_callback
        if hasattr(actual_scenario, "info_callback")
        else None,
        scenario_name=args.scenario_name,
    )
    return env


def GraphMPEEnv(args):
    """
    Same as MPEEnv but for graph environment with challenge weighting support
    """

    # load scenario from script
    assert "graph" in args.scenario_name, "Only use graph env for graph scenarios"
    scenario = load(args.scenario_name + ".py").Scenario()
    
    # Apply challenge weighting if enabled
    if hasattr(args, 'enable_challenge_weighting') and args.enable_challenge_weighting:
        # Create challenge weights from arguments
        challenge_weights = {}
        if hasattr(args, 'challenge_distance_weight'):
            challenge_weights[ChallengeType.DISTANCE_TO_GOAL] = args.challenge_distance_weight
        if hasattr(args, 'challenge_obstacle_weight'):
            challenge_weights[ChallengeType.OBSTACLE_DENSITY] = args.challenge_obstacle_weight
        if hasattr(args, 'challenge_collision_weight'):
            challenge_weights[ChallengeType.COLLISION_RISK] = args.challenge_collision_weight
        if hasattr(args, 'challenge_coordination_weight'):
            challenge_weights[ChallengeType.MULTI_AGENT_COORDINATION] = args.challenge_coordination_weight
        
        # Set default weights for missing challenge types
        default_weights = {
            ChallengeType.BOUNDARY_VIOLATION: 1.0,
            ChallengeType.ADVERSARY_PRESSURE: 0.0,
            ChallengeType.COMMUNICATION_COMPLEXITY: 0.0,
            ChallengeType.TASK_COMPLEXITY: 1.0
        }
        
        for challenge_type, default_weight in default_weights.items():
            if challenge_type not in challenge_weights:
                challenge_weights[challenge_type] = default_weight
        
        # Initialize scenario with challenge weighting
        if hasattr(scenario, '__init__'):
            # Check if scenario supports challenge weighting parameters
            if 'enable_challenge_weighting' in scenario.__init__.__code__.co_varnames:
                scenario = load(args.scenario_name + ".py").Scenario(
                    enable_challenge_weighting=True,
                    challenge_weights=challenge_weights,
                    reward_amplification_factor=getattr(args, 'challenge_amplification_factor', 2.0),
                    challenge_min_threshold=getattr(args, 'challenge_min_threshold', 0.1),
                    challenge_max_threshold=getattr(args, 'challenge_max_threshold', 8.0)
                )
            else:
                # Apply wrapper approach for scenarios that don't support challenge weighting
                from utils.challenge_assessment import RewardWeightingWrapper
                assessor = ChallengeAssessor(
                    challenge_weights=challenge_weights,
                    min_challenge_threshold=getattr(args, 'challenge_min_threshold', 0.1),
                    max_challenge_threshold=getattr(args, 'challenge_max_threshold', 8.0),
                    reward_amplification_factor=getattr(args, 'challenge_amplification_factor', 2.0)
                )
                scenario = RewardWeightingWrapper(scenario, assessor)
    
    # create world
    world = scenario.make_world(args=args)
    from multiagent.environment import MultiAgentGraphEnv

    # Create environment with appropriate callbacks
    if hasattr(scenario, 'scenario'):  # Wrapped scenario
        actual_scenario = scenario.scenario
        reward_callback = scenario.scenario.reward
    else:
        actual_scenario = scenario
        reward_callback = scenario.reward

    # create multiagent environment
    env = MultiAgentGraphEnv(
        world=world,
        reset_callback=actual_scenario.reset_world,
        reward_callback=reward_callback,
        observation_callback=actual_scenario.observation,
        graph_observation_callback=getattr(actual_scenario, 'graph_observation', None),
        update_graph=getattr(actual_scenario, 'update_graph', None),
        id_callback=getattr(actual_scenario, 'get_id', None),
        info_callback=actual_scenario.info_callback,
        scenario_name=args.scenario_name,
    )

    return env


def GPGMPEEnv(args):
    """
    MPE env but compatible with the GPG baseline code
    """
    # load scenario from script
    scenario = load("navigation_gpg.py").Scenario()
    # create world
    world = scenario.make_world(args=args)
    from multiagent.environment import MultiAgentGPGEnv

    env = MultiAgentGPGEnv(
        world=world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        info_callback=scenario.info_callback
        if hasattr(scenario, "info_callback")
        else None,
        scenario_name="navigation_gpg",
    )
    return env


def CADRLMPEEnv(args, phase):
    """
    MPE env but compatible with the CADRL baseline code
    """
    # load scenario from script
    scenario = load("navigation_cadrl.py").Scenario()
    # create world
    world = scenario.make_world(args=args)
    from multiagent.environment import MultiAgentCADRLEnv

    env = MultiAgentCADRLEnv(
        world=world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        info_callback=scenario.info_callback
        if hasattr(scenario, "info_callback")
        else None,
        scenario_name="navigation_cadrl",
        phase=phase,
    )
    return env


def AttentionMPEEnv(args):
    """
    MPE env but compatible with the Attention baseline code
    """
    # load scenario from script
    scenario = load("navigation_attention.py").Scenario()
    # create world
    world = scenario.make_world(args=args)
    from multiagent.environment import MultiAgentAttentionEnv

    env = MultiAgentAttentionEnv(
        world=world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        info_callback=scenario.info_callback
        if hasattr(scenario, "info_callback")
        else None,
        scenario_name="navigation_attention",
    )
    return env


def DGNMPEEnv(args):
    """
    MPE env but compatible with the DGN baseline code
    """
    # load scenario from script
    scenario = load("navigation_dgn.py").Scenario()
    # create world
    world = scenario.make_world(args=args)
    from multiagent.environment import MultiAgentDGNEnv

    env = MultiAgentDGNEnv(
        world=world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        info_callback=scenario.info_callback
        if hasattr(scenario, "info_callback")
        else None,
        scenario_name="navigation_dgn",
    )
    return env


def make_parallel_env(args: argparse.Namespace):
    """
    Creates a parallel environment for training with challenge weighting support
    """
    def get_env_fn(rank: int):
        def init_env():
            if args.env_name == "MPE":
                env = MPEEnv(args)
            elif args.env_name == "GraphMPE":
                env = GraphMPEEnv(args)
            else:
                print(f"Can not support the {args.env_name} environment")
                raise NotImplementedError
            env.seed(args.seed + rank * 1000)
            return env

        return init_env

    if args.n_rollout_threads == 1:
        if args.env_name == "GraphMPE":
            from onpolicy.envs.env_wrappers import GraphDummyVecEnv
            return GraphDummyVecEnv([get_env_fn(0)])
        else:
            from onpolicy.envs.env_wrappers import DummyVecEnv
            return DummyVecEnv([get_env_fn(0)])
    else:
        if args.env_name == "GraphMPE":
            from onpolicy.envs.env_wrappers import GraphSubprocVecEnv
            return GraphSubprocVecEnv(
                [get_env_fn(i) for i in range(args.n_rollout_threads)]
            )
        else:
            from onpolicy.envs.env_wrappers import SubprocVecEnv
            return SubprocVecEnv([get_env_fn(i) for i in range(args.n_rollout_threads)]) 