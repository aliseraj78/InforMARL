#!/bin/bash

# Challenge-weighted training script for InforMARL (without wandb)
# This script runs training with challenge-based reward weighting enabled

echo "Starting Challenge-Weighted InforMARL Training (No wandb)..."
echo "=========================================================="
echo "Challenge weighting will give more importance to difficult experiences"
echo "This should improve learning from challenging situations"
echo ""

# Run the challenge-weighted training with wandb disabled
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
--num_env_steps 2000000 ^
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
--challenge_coordination_weight 1.5 ^
--use_wandb "False"

echo ""
echo "Challenge-weighted training completed!"
echo "Check the logs for challenge weighting information and performance metrics." 