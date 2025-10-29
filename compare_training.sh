#!/bin/bash

# Comparison script: Run training with and without challenge weighting
# This will help you see the difference in performance

echo "InforMARL Training Comparison: Standard vs Challenge-Weighted"
echo "============================================================="
echo ""

# Create output directories
mkdir -p comparison_results

echo "Step 1: Running standard training (without challenge weighting)..."
echo "=================================================================="

# Run standard training
python -u onpolicy/scripts/train_mpe.py --project_name "informarl_standard" ^
--env_name "GraphMPE" ^
--algorithm_name "rmappo" ^
--seed 0 ^
--experiment_name "informarl_standard" ^
--scenario_name "navigation_graph" ^
--num_agents 3 ^
--collision_rew 5 ^
--n_training_threads 1 ^
--n_rollout_threads 5 ^
--num_mini_batch 1 ^
--episode_length 25 ^
--num_env_steps 1000000 ^
--ppo_epoch 10 ^
--use_ReLU ^
--gain 0.01 ^
--lr 7e-4 ^
--critic_lr 7e-4 ^
--user_name "marl" ^
--use_cent_obs "False" ^
--graph_feat_type "relative" ^
--auto_mini_batch_size ^
--target_mini_batch_size 128 > comparison_results/standard_training.log 2>&1

echo "Standard training completed!"
echo ""

echo "Step 2: Running challenge-weighted training..."
echo "============================================="

# Run challenge-weighted training
python -u onpolicy/scripts/train_mpe_challenge_weighted.py --project_name "informarl_challenge_weighted" ^
--env_name "GraphMPE" ^
--algorithm_name "rmappo" ^
--seed 0 ^
--experiment_name "informarl_challenge_weighted" ^
--scenario_name "navigation_graph_challenge_weighted" ^
--num_agents 3 ^
--collision_rew 5 ^
--n_training_threads 1 ^
--n_rollout_threads 5 ^
--num_mini_batch 1 ^
--episode_length 25 ^
--num_env_steps 1000000 ^
--ppo_epoch 10 ^
--use_ReLU ^
--gain 0.01 ^
--lr 7e-4 ^
--critic_lr 7e-4 ^
--user_name "marl" ^
--use_cent_obs "False" ^
--graph_feat_type "relative" ^
--auto_mini_batch_size ^
--target_mini_batch_size 128 ^
--enable_challenge_weighting "True" ^
--challenge_amplification_factor 2.0 ^
--challenge_min_threshold 0.1 ^
--challenge_max_threshold 8.0 ^
--challenge_distance_weight 1.5 ^
--challenge_obstacle_weight 2.0 ^
--challenge_collision_weight 1.8 ^
--challenge_coordination_weight 1.5 > comparison_results/challenge_weighted_training.log 2>&1

echo "Challenge-weighted training completed!"
echo ""

echo "Step 3: Analyzing results..."
echo "============================"

# Create a simple analysis script
python -c "
import os
import re

def extract_final_rewards(log_file):
    if not os.path.exists(log_file):
        return 'Log file not found'
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Look for reward patterns
    reward_patterns = re.findall(r'reward.*?([-+]?\d*\.\d+)', content, re.IGNORECASE)
    if reward_patterns:
        # Get the last few rewards
        recent_rewards = reward_patterns[-10:]
        return f'Recent rewards: {recent_rewards}'
    else:
        return 'No reward data found'

print('Standard Training Results:')
print(extract_final_rewards('comparison_results/standard_training.log'))
print()
print('Challenge-Weighted Training Results:')
print(extract_final_rewards('comparison_results/challenge_weighted_training.log'))
print()
print('Check the log files in comparison_results/ for detailed information.')
"

echo ""
echo "Comparison completed!"
echo "Check the log files in comparison_results/ for detailed results."
echo "The challenge-weighted version should show better learning from difficult situations." 