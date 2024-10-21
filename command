python -u onpolicy/scripts/train_mpe.py --use_valuenorm --use_popart ^
--project_name "informarl" ^
--env_name "GraphMPE" ^
--algorithm_name "rmappo" ^
--seed 0 ^
--experiment_name "informarl" ^
--scenario_name "navigation_graph" ^
--num_agents 3 ^
--collision_rew 5 ^
--n_training_threads 1 --n_rollout_threads 20 ^
--num_mini_batch 1 ^
--episode_length 25 ^
--num_env_steps 200000 ^
--ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 ^
--user_name "marl" ^
--use_cent_obs "False" ^
--graph_feat_type "relative" ^
--auto_mini_batch_size --target_mini_batch_size 20


6a726c699b6673b42e85883280fe61090af39a6d

6a726c699b6673b42e85883280fe61090af39a6d



