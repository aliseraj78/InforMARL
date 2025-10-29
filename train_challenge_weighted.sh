#!/bin/bash

# Challenge-weighted training script for InforMARL
# This script runs the original training command with challenge-based reward weighting

echo "Starting Challenge-Weighted InforMARL Training..."
echo "================================================"

# Original training command with challenge weighting modifications
python -u onpolicy/scripts/train_mpe.py --project_name "informarl_challenge_weighted" ^
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
--challenge_max_threshold 8.0

echo "Challenge-weighted training completed!" 