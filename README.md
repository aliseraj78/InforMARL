# InforMARL Training Guide

This repository contains the InforMARL project for Multi-Agent Reinforcement Learning.

## Prerequisites

- Python 3.10.4

## Installation

1. Create a virtual environment (recommended):

```bash
python -m venv env
```

2. Activate the virtual environment:

   - Windows:
     ```bash
     .\env\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source env/bin/activate
     ```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running Training

To run the challenge-weighted training script, use the following command:

```bash
python -u .\onpolicy\scripts\train_mpe_challenge_weighted.py --project_name "informarl" --env_name "GraphMPE" --algorithm_name "rmappo" --seed 0 --experiment_name "informarl" --scenario_name "navigation_graph" --num_agents 3 --collision_rew 5 --n_training_threads 1 --n_rollout_threads 5 --num_mini_batch 1 --episode_length 25 --num_env_steps 2000000 --ppo_epoch 10 --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --user_name "marl" --use_cent_obs "False" --graph_feat_type "relative" --auto_mini_batch_size --target_mini_batch_size 128
```

### Key Parameters

- `--num_agents`: Number of agents (default: 3)
- `--episode_length`: Length of each episode (default: 25)
- `--num_env_steps`: Total number of environment steps (default: 2000000)
- `--lr`: Learning rate for actor (default: 7e-4)
- `--critic_lr`: Learning rate for critic (default: 7e-4)
- `--scenario_name`: Scenario to run (default: navigation_graph)
- `--algorithm_name`: Algorithm to use (default: rmappo)
- `--graph_feat_type`: Graph feature type (default: relative)

## Project Structure

- `onpolicy/`: On-policy algorithms implementation
- `multiagent/`: Multi-agent environment definitions
- `scripts/`: Training and evaluation scripts
- `utils/`: Utility functions
- `baselines/`: Baseline algorithms

## License

See LICENSE file for details.
