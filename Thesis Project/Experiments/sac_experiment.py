#  PPO Conflict Situation-1.3 Experiment Discussed in the Thesis Documentation:



if __name__ == "__main__":
    
    
    
    """
        SAC Conflict Situation-1.3 Experiment Discussed in the Thesis Documentation:

    
    run_config = air.RunConfig(
        storage_path=storage_path,
        stop=stop_criteria,
        checkpoint_config=train.CheckpointConfig(
                checkpoint_frequency=2, checkpoint_at_end=True),  # to checkpoint trials every N iterations,
        #callbacks=[ExtractEpisodeReturnCallback()]  # Ensure metric is extracted
        callbacks=[
            ExtractEpisodeReturnCallback(),  # First, extract episode_return_mean
            WandbLoggerCallback(      # Ensure metric is extracted
                project="ppo-experiment-Conflict_situation-1.3",
                api_key="api_key",
                log_config=True  # Log the full config
            )
        ],
        success_metric={
            f"{EVALUATION_RESULTS}/{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": (
                args.stop_reward
            ),
        },
    
    )
    
    
    """
    
    import gymnasium as gym
    
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    from ray.rllib.utils.test_utils import (
        add_rllib_example_script_args,
        run_rllib_example_script_experiment,
    )
    from ray.tune.registry import get_trainable_cls
    
    
    import ray
    import os
    from ray.rllib.algorithms.sac import SACConfig, SAC
    from ray.tune.schedulers.pb2 import PB2
    from ray.tune.logger import Logger
    # Set a custom temporary directory and Plasma store directory
    from ray import train, tune
    from ray import air
    
    
    # Import CustomEnvClass
    import CustomEnvClass
    import logging
    import torch
    import numpy as np
    from ray.rllib.env.env_context import EnvContext
    
    # Dynamically load all attributes from CustomEnvClass
    for name in dir(CustomEnvClass):
        if not name.startswith("__"):  # Skip system names
            globals()[name] = getattr(CustomEnvClass, name)
    
    # Configure logging for debugging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    
    from ray.tune.logger import Logger
    from ray.rllib.policy.policy import PolicySpec
    
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from ray.air.integrations.wandb import WandbLoggerCallback
    from ray.rllib.utils.metrics import *
    
    ray.init(
        _temp_dir="/workspace/Ray_Results/tmp",  # Temporary directory for Ray's process
       # object_store_memory=2 * 1024 * 1024 * 1024,  # 2GB object store memory limit
        #_plasma_directory="/nfs1/obiorah/workspace/Ray_Results/plasma",  # Plasma store directory
        #_redis_max_memory=4 * 1024 * 1024 * 1024,  # Set 4GB memory limit for Redis
        #logging_level="INFO",  # Set logging level
        #log_to_driver=True,  # Log output to driver
        ignore_reinit_error=True  # Prevent errors when reinitializing
    )
    
    
    # Define the custom logger creator function
    def logger_creator(config):
        # Define your custom log directory
        logdir = "/workspace/Ray_Results/logs"
        
        # Ensure the directory exists
        if not os.path.exists(logdir):
            os.makedirs(logdir, exist_ok=True)
        
        # Create the UnifiedLogger instance
        return Logger(config=config, logdir=logdir)
    
    
    
    #storage_path = "file:///nfs1/obiorah/workspace/Ray_Results"
    storage_path = "/workspace/Ray_Results"
    checkpoint_path = "/workspace/Ray_Results/experiment_checkpoints"
    
    def policy_mapping_fn(agent_id, episode, **kwargs):
        agent_idx = int(agent_id[-1])  # 0 (player1) or 1 (player2)
        return "learning_policy" if episode.id_ % 2 == agent_idx else "random_policy"
    
    
    
    # Environment configuration
    simulation_env_config = {
        "agent_count": 2,
        "env_name": "Conflict_situation-1.3",
        "max_speed": 1,
        "scale": 1.32,
        "dt": 0.01488343,
        "goal_radius": 0.01,
        "timesteps_per_episode": 5000,    #  horizon range 320 to 5000
        "agent_size": (0.25, 0.25), 
        "goal_size": (0.2, 0.2)
    }
    
    # Initialize environment
    env_config = EnvContext(simulation_env_config, worker_index=0)
    env = ContinuousPathfindingEnv(env_config)
    
    shared_obs_space = list(env.observation_spaces.values())[0]
    shared_act_space = list(env.action_spaces.values())[0]
    
    #wrapped_env = FlattenObservation(env)
    
    policy_config = {
        #"lr": 0.0003,
        #"entropy_coeff": 0.01,
        #"vf_loss_coeff": 1.0,
        #"clip_param": 0.2,
        #"kl_coeff": 0.2,
        #"vf_clip_param": 10.0,
        #"use_critic": True,
        "num_gpus": 2 if torch.cuda.is_available() else 0,  # Use GPU if available
    
    }
    
    
    from ray.tune.logger import pretty_print
    
    class CustomLogger:
        def __init__(self, config):
            # Initialize the logger with your config
            self.config = config
            self.logdir =  "/nfs1/obiorah/workspace/Ray_Results/logs"
    
    
        def on_result(self, result):
            # Log the result in your preferred format
            print(pretty_print(result))  # You can replace this with your custom logic
    
        def update_config(self, config):
            # Update the config if necessary
            pass
    
    # Use the custom logger
    logger_creator_nt = lambda config: CustomLogger(config)
    
    
    from ray.tune import Callback
    
    class ExtractEpisodeReturnCallback(Callback):
        def on_trial_result(self, iteration, trials, trial, result, **kwargs):
            # Ensure "episode_return_mean" exists in env_runners
            if "evaluation" in result and "env_runners" in result["evaluation"] and "episode_return_mean" in result["evaluation"]["env_runners"]:
                # Move it to the top level
                result["episode_return_mean"] = result["evaluation"]["env_runners"]["episode_return_mean"]
    
    from ray.air.integrations.wandb import WandbLoggerCallback
    
    class CustomWandbLoggerCallback(WandbLoggerCallback):
        def log_trial_result(self, iteration, trials, trial, result):
            # Extract episode_return_mean from env_runners
            if "env_runners" in result and "episode_return_mean" in result["env_runners"]:
                result["episode_return_mean"] = result["env_runners"]["episode_return_mean"]
    
            # Now log to Wandb
            super().log_trial_result(iteration, trials, trial, result)
    
    time_attr=simulation_env_config["timesteps_per_episode"]
    
    pb2_scheduler = PB2(
        time_attr=time_attr,
        #metric=f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}",
        #mode="max",
        perturbation_interval=50000,
        # Copy bottom % with top % weights.
        quantile_fraction=0.25,
        hyperparam_bounds={
            "actor_lr": [1e-5, 1e-3],
            "critic_lr": [1e-6, 1e-4],
            "alpha_lr": [1e-6, 1e-3],
            "gamma": [0.95, 0.99],
            "n_step": [1, 3],
            "initial_alpha": [1.0, 1.5],
            "tau": [0.001, 0.1],
            "target_entropy": [-10, -1],
            "train_batch_size": [128, 512],
            "target_network_update_freq": [1, 4],
            #"timesteps_per_episode": [200, 4000],
        },
    )
    
    
    SAC_Config = (
        SACConfig()
        #.update_from_dict(config)
        .api_stack(
            #enable_rl_module_and_learner=True,
            #enable_env_runner_and_connector_v2=True,
        )    
        .environment(
            env=ContinuousPathfindingEnv,
            env_config=env_config,
        )
    
        .env_runners(
            rollout_fragment_length="auto",
            num_env_runners=6,
        )
     
        .reporting(
            metrics_num_episodes_for_smoothing=5,
            min_sample_timesteps_per_iteration=1000,
        )
        .evaluation(
            evaluation_duration="auto",
            evaluation_interval=1,
            evaluation_num_env_runners=1,
            evaluation_parallel_to_training=True,
            evaluation_config={
                "explore": False,
            },
        )
        
        .training(
            initial_alpha=tune.choice([1.0, 1.5]),
            actor_lr=tune.uniform(1e-5, 1e-3),
            critic_lr=tune.uniform(1e-6, 1e-4),
            alpha_lr=tune.uniform(1e-6, 1e-3),
            target_entropy=tune.choice([-10, -5, -1, "auto"]),
            n_step=tune.choice([1, 3, (1, 3)]),
            tau=tune.uniform(0.001, 0.1),
            train_batch_size=tune.choice([128, 256, 512]),
            target_network_update_freq=tune.choice([1, 2, 4]),
            #timesteps_per_episode=tune.uniform([200, 4000]),
    
            
            lr=None,
            replay_buffer_config={
                "type": "MultiAgentPrioritizedEpisodeReplayBuffer",
                "capacity": 100000,
                "alpha": 0.6,
                "beta": 0.4,
            },
            num_steps_sampled_before_learning_starts=256,
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                "post_fcnet_hiddens": [],
                "post_fcnet_activation": None,
                "post_fcnet_weights_initializer": "orthogonal_",
                "post_fcnet_weights_initializer_config": {"gain": 0.01},
            },
        
    
        )
        .evaluation(
            evaluation_duration="auto",
            evaluation_interval=1,
            evaluation_num_env_runners=1,
            evaluation_parallel_to_training=True,
            evaluation_config={
                "explore": False,
            },
        )
    
    
        .multi_agent(
            policies={
                #"policy_1": (PPOTorchPolicy, wrapped_env.observation_space, wrapped_env.action_space, policy_config),
                #"policy_2": (PPOTorchPolicy, wrapped_env.observation_space, wrapped_env.action_space, policy_config),
                
                "default_policy": PolicySpec( 
                    policy_class=None,  # Use default policy class
                    observation_space=shared_obs_space,  # Auto-detect from env
                    action_space=shared_act_space,  # Auto-detect from env
                    config=policy_config
                ),
                "learning_policy": PolicySpec( 
                    policy_class=None,  # Use default policy class
                    observation_space=shared_obs_space,  # Auto-detect from env
                    action_space=shared_act_space,  # Auto-detect from env
                    config=policy_config
                )
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "learning_policy",
            policies_to_train=["learning_policy"]  # Reference the defined policy
        )
    
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(rl_module_specs={
                "learning_policy": RLModuleSpec(),
                #"random_policy": RLModuleSpec(),
            }),
        )
        .learners(
            num_learners=1,  # Use all 8 learners (one per GPU)
            num_gpus_per_learner=0.25,  # Each learner gets one GPU
            num_cpus_per_learner=0,  
        )
        .framework("torch")
        .debugging(
            logger_creator=logger_creator_nt
        )
        .reporting(
            #metrics_num_episodes_for_smoothing=5,
            #min_sample_timesteps_per_iteration=time_attr*0.5,
    
        )
    )
    
    
    
    
    
    # Train for n iterations with high LR.
    #config.training(lr=0.001)
    # Build the Algorithm with the custom logger creator
    #algo = config.build(logger_creator=logger_creator_nt)
    
    #print(algo_high_lr.train())
    env_name=simulation_env_config["env_name"]
    storage_path = "/workspace/Ray_Results"
    stop_reward = 10000
    stop_criteria = {
        "training_iteration": 500,  # Stop after 200 iterations
        NUM_ENV_STEPS_SAMPLED_LIFETIME: time_attr,
        #"episode_reward_mean": 5000,  # Stop if mean reward reaches 500
        #"timesteps_total": 1_0  # Stop after 1M timesteps
        #f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": -800.0,
        #f"{ENV_RUNNER_RESULTS}/{NUM_ENV_STEPS_SAMPLED_LIFETIME}": 100000,
        #f"{EVALUATION_RESULTS}/{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": (
        #    stop_reward
        #),
        #f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": (
        #    stop_reward
        #),
    }
    
    run_config = air.RunConfig(
        name="SAC_Trial_Experiment_pb2_" + env_name,
        storage_path=storage_path,
        stop=stop_criteria,
        checkpoint_config=train.CheckpointConfig(
                checkpoint_frequency=20,  # Save every 5 iterations intune with evaluation interval of 5
                checkpoint_at_end=True,   # Save final checkpoint
                checkpoint_score_attribute="evaluation/env_runners/episode_reward_mean",  # Sort by evaluation reward
                checkpoint_score_order="max"  # Keep the ones with the highest reward
            ),
        #callbacks=[ExtractEpisodeReturnCallback()]  # Ensure metric is extracted
    
        callbacks=[
            ExtractEpisodeReturnCallback(),  # First, extract episode_return_mean
            WandbLoggerCallback(      # Ensure metric is extracted
                project="SAC-experiment-Conflict_situation-1.3",
                api_key="api_key",
                log_config=True  # Log the full config
            )
        ],
         
    )
    
    
    final_run_config = air.RunConfig(
        name="SAC_Final_Experiment_pb2_" + env_name + "with_best_config",
        storage_path=storage_path,
        stop=stop_criteria,
        checkpoint_config=train.CheckpointConfig(
                checkpoint_frequency=20,  # Save every 5 iterations intune with evaluation interval of 5
                checkpoint_at_end=True,   # Save final checkpoint
                checkpoint_score_attribute="evaluation/env_runners/episode_reward_mean",  # Sort by evaluation reward
                checkpoint_score_order="max"  # Keep the ones with the highest reward
            ),
        #callbacks=[ExtractEpisodeReturnCallback()]  # Ensure metric is extracted
    
        callbacks=[
            ExtractEpisodeReturnCallback(),  # First, extract episode_return_mean
            WandbLoggerCallback(      # Ensure metric is extracted
                project="SAC-experiment-Conflict_situation-1.3",
                api_key="api_key",
                log_config=True  # Log the full config
            )
        ],
        
    )
    
    
    
    
    
    
    tuner = tune.Tuner(
        SAC,
        param_space=SAC_Config.to_dict(),
        tune_config=tune.TuneConfig(
            #metric="env_runners/episode_return_mean",  # Metric to track
            #metric="module_episode_returns_mean.learning_policy", 
            #metric=f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}", # Metric to track
            #metric=f"{EVALUATION_RESULTS}/{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}", # Metric to track
            mode="max",  # Maximizing the mean reward
            #scheduler=pb2_scheduler,
            num_samples=100,  # Number of trials
        ),
    
        run_config=run_config,
    )
    
    result_grid = tuner.fit()
    best_result = result_grid.get_best_result(
        metric="evaluation/env_runners/episode_return_mean", mode="max"
    )
    print(
        f"Finished running hyperparameter search for (env={env_name})"
    )
    #print(f"Best result for {env_name}: {best_result}")
    #print(f"Best config for {env_name}: {best_result.metrics['config']}")
    print(f"Best env_runners for {env_name} environment: {best_result.metrics['evaluation']['env_runners']}")  
    
    
    # Final run with the best configuration found.
    tuner = tune.Tuner(
        SAC,
        param_space=best_result.config,
        run_config=final_run_config,
    )
    print(f"Running final experiment with best configuration for (env={env_name})...")
    
    final_result_grid = tuner.fit()
    
    print("."*100)
    print(f"Running final experiment with best configuration for (env={env_name})...")
    print(
        "Results from running the best configs can be found in the "
        "`sac_experiment_pb2_" + env_name + "_best` directories."
    )
    
    
    ray.shutdown()
    
    
    

