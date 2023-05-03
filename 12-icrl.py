import datetime
import importlib
import json
import os
import pickle
import sys
import time
import random
import gym
import numpy as np
import yaml
import torch
import tools
import wandb
import common
import sys; sys.path += ["baselines"]

from baselines.common.cns_evaluation import evaluate_icrl_policy
from baselines.constraint_models.constraint_net.constraint_net import ConstraintNet
from baselines.stable_baselines3 import PPOLagrangian
from baselines.stable_baselines3.common.vec_env import DummyVecEnv, VecCostWrapper
from baselines.utils.data_utils import read_args, load_config
from baselines.utils.env_utils import sample_from_agent
from baselines.utils.model_utils import load_ppo_config

tools.utils.nowarnings()

# def null_cost(x, *args):
#     # Zero cost everywhere
#     return np.zeros(x.shape[:1])


def train(config):
    configuration, seed = load_config(args)
    configuration["seed"] = seed
    configuration = tools.data.Configuration(tools.utils.convert_lambdas(configuration))
    state_action_space = tools.environments.get_state_action_space(
        configuration["env_type"], configuration["env_id"])
    configuration.update({"state_action_space": state_action_space})
    config_name = os.path.splitext(os.path.basename(args.c))[0]
    logdir = "%s(%s)-%s-%s-(%.2f,%d)" % ("ICRL",
        "PPOLag", config_name.split("-")[-1], tools.utils.timestamp(), 
        0, configuration["seed"])
    logger = tools.data.Logger(project="ICL", 
        window=configuration["window"], logdir=logdir)
    configuration.update({"logger": logger})
    wandb.run.log_code()
    # wandb.run.log_code(root=args.c, include_fn=lambda path: path.endswith(".json"))
    yaml_artifact = wandb.Artifact('config-yaml', type='yaml')
    yaml_artifact.add_file(args.c)
    wandb.log_artifact(yaml_artifact)

    # Create manual cost function
    if configuration["cost_condition"] != "":
        manual_cost = common.create_manual_cost_function(configuration)
        manualcostvalues, manualcostmap = \
            manual_cost.outputs(configuration["state_action_space"])
        manualcostvalues = np.array(manualcostvalues).squeeze()
        configuration["logger"].update({
            "expert_cost": manualcostmap.fig,
        })
        configuration.update({
            "manualcostvalues": manualcostvalues,
        })

    # Create cost function
    cost = tools.functions.CostFunction(configuration, i=configuration["i"], h=64, o=1)
    configuration.update({"cost": cost})
    costvalues, costmap = cost.outputs(configuration["state_action_space"], invert=True)
    costvalues = np.array(costvalues).squeeze()
    configuration["logger"].update({"cost": costmap.fig})
    if configuration["cost_condition"] != "":
        configuration["logger"].update({"cost_comparison": \
            configuration["cost_comparison"](manualcostvalues, costvalues)})

    # Expert dataset accrual
    expert_dataset = tools.base.TrajectoryDataset.load()
    expert_acr, expert_acrplot = tools.functions.NormalizedAccrual()({
        "state_reduction": configuration["state_reduction"],
        "dataset": expert_dataset,
        "spaces": configuration["state_action_space"],
        "normalize_func": configuration["normalize_func"],
    })
    expert_acr = np.array(expert_acr).squeeze()
    configuration["logger"].update({
        "expert_accrual": expert_acrplot.fig
    })
    configuration.update({
        "expert_acr": expert_acr,
    })

    # Set specs
    train_env = configuration["env"]
    test_env = configuration["test_env"]
    sampling_env = configuration["sampling_env"]
    is_discrete = isinstance(train_env.action_space, gym.spaces.Discrete)
    obs_dim = train_env.observation_space.shape[0]
    acs_dim = train_env.action_space.n if is_discrete else train_env.action_space.shape[0]
    action_low, action_high = None, None
    if isinstance(train_env.action_space, gym.spaces.Box):
        action_low, action_high = train_env.action_space.low, train_env.action_space.high

    # Load expert data
    expert_data = torch.load("data.pt")
    expert_obs = []
    expert_acs = []
    for S, A in expert_data:
        for s in S:
            expert_obs += [s]
        for a in A:
            expert_acs += [a]
    expert_obs = np.array(expert_obs)
    expert_acs = np.array(expert_acs)

    # Initialize constraint net, true constraint net
    cn_parameters = {
        'cost': configuration["cost"],
        'obs_dim': obs_dim,
        'acs_dim': acs_dim,
        'batch_size': configuration['PPO']["batch_size"],
        'expert_obs': expert_obs,  # select obs at a time step t
        'expert_acs': expert_acs,  # select acs at a time step t
        'is_discrete': is_discrete,
        'regularizer_coeff': configuration['CN']['cn_reg_coeff'],
        'no_importance_sampling': configuration['CN']['no_importance_sampling'],
        'per_step_importance_sampling': configuration['CN']['per_step_importance_sampling'],
        'clip_obs': configuration['CN']['clip_obs'],
        'initial_obs_mean': None,
        'initial_obs_var': None,
        'action_low': action_low,
        'action_high': action_high,
        'target_kl_old_new': configuration['CN']['cn_target_kl_old_new'],
        'target_kl_new_old': configuration['CN']['cn_target_kl_new_old'],
        'train_gail_lambda': configuration['CN']['train_gail_lambda'],
        'eps': configuration['CN']['cn_eps'],
        'device': configuration['t'].device,
    }

    constraint_net = ConstraintNet(**cn_parameters)

    # Pass constraint net cost function to cost wrapper (train env)
    train_env = VecCostWrapper(DummyVecEnv([lambda: configuration["env"]]))
    train_env.set_cost_function(constraint_net.cost_function)
    test_env = VecCostWrapper(DummyVecEnv([lambda: configuration["test_env"]]))
    test_env.set_cost_function(constraint_net.cost_function)

    # Init ppo agent
    ppo_parameters = load_ppo_config(configuration, train_env, seed, None)
    create_nominal_agent = lambda: PPOLagrangian(logger, **ppo_parameters)
    nominal_agent = create_nominal_agent()

    class ICRLPolicy(tools.base.Policy):
        def act(self, s):
            return nominal_agent.policy.forward(torch.as_tensor([s]).to(configuration['t'].device))[0].detach().view(-1).cpu().numpy()
    icrl_policy = ICRLPolicy()

    # Train
    timesteps = 0
    best_true_reward, best_true_cost, best_forward_kl, best_reverse_kl = -np.inf, np.inf, np.inf, np.inf
    for itr in range(configuration['running']['n_iters']):
        current_progress_remaining = 1 - float(itr) / float(configuration['running']['n_iters'])

        # Update agent
        nominal_agent.learn(
            total_timesteps=configuration['PPO']['forward_timesteps'],
            cost_function="cost",  # Cost should come from cost wrapper
            callback=[],
        )
        timesteps += nominal_agent.num_timesteps

        # Sample nominal trajectories
        observations, actions, lengths = sample_from_agent(
            agent=nominal_agent,
            env=sampling_env,
            configuration=configuration,
            rollouts=int(configuration['running']['sample_rollouts']),
            store_by_game=False,
        )
        sample_obs = observations # these are unnormalized
        sample_acts = actions

        # Update constraint net
        mean, var = None, None
        backward_metrics = constraint_net.train_nn(iterations=configuration['CN']['backward_iters'],
                                                   nominal_obs=sample_obs,
                                                   nominal_acs=sample_acts,
                                                   episode_lengths=lengths,
                                                   obs_mean=mean,
                                                   obs_var=var,
                                                   current_progress_remaining=current_progress_remaining)

        # Pass updated cost_function to cost wrapper (train_env, eval_env, but not sampling_env)
        train_env.set_cost_function(constraint_net.cost_function)
        test_env.set_cost_function(constraint_net.cost_function)

        # Evaluate:
        # reward on true environment
        mean_reward, std_reward, mean_nc_reward, std_nc_reward, record_infos, costs = \
            evaluate_icrl_policy(nominal_agent, test_env,
                                 record_info_names=[],
                                 n_eval_episodes=configuration['running']['n_eval_episodes'],
                                 deterministic=False)

        # Update best metrics
        if mean_nc_reward > best_true_reward:
            best_true_reward = mean_nc_reward

        # Collect metrics
        metrics = {
            "run_iter": itr,
            "timesteps": timesteps,
            "true/mean_nc_reward": mean_nc_reward,
            "true/std_nc_reward": std_nc_reward,
            "true/mean_reward": mean_reward,
            "true/std_reward": std_reward,
            "best_true/best_reward": best_true_reward
        }
        metrics.update(backward_metrics)
        logger.update(metrics)

        if itr%configuration["plot_interval"]==0:
            costvalues, costmap = cost.outputs(configuration["state_action_space"], invert=True)
            costvalues = np.array(costvalues).squeeze()
            configuration["logger"].update({"cost": costmap.fig})
            if configuration["cost_condition"] != "":
                configuration["logger"].update({"cost_comparison": \
                    configuration["cost_comparison"](manualcostvalues, costvalues)})

            dataset = configuration["env"].trajectory_dataset(icrl_policy, 
                configuration["expert_episodes"], cost=configuration["cost"])
            acr, acrplot = tools.functions.NormalizedAccrual()({
                "state_reduction": configuration["state_reduction"],
                "dataset": dataset,
                "spaces": configuration["state_action_space"],
                "normalize_func": configuration["normalize_func"],
            })
            acr = np.array(acr).squeeze()
            configuration["logger"].update({
                "accrual": acrplot.fig,
                "accrual_comparison": configuration["accrual_comparison"](expert_acr, acr),
            })

            dataset = configuration["env"].trajectory_dataset(icrl_policy, 
                configuration["expert_episodes"])
            acr, acrplot = tools.functions.NormalizedAccrual()({
                "state_reduction": configuration["state_reduction"],
                "dataset": dataset,
                "spaces": configuration["state_action_space"],
                "normalize_func": configuration["normalize_func"],
            })
            acr = np.array(acr).squeeze()
            configuration["logger"].update({
                "accrual_no_cost": acrplot.fig,
                "accrual_comparison_no_cost": configuration["accrual_comparison"](expert_acr, acr),
            })


    costvalues, costmap = cost.outputs(configuration["state_action_space"], invert=True)
    costvalues = np.array(costvalues).squeeze()
    configuration["logger"].update({"cost": costmap.fig})
    if configuration["cost_condition"] != "":
        configuration["logger"].update({"cost_comparison": \
            configuration["cost_comparison"](manualcostvalues, costvalues)})

    dataset = configuration["env"].trajectory_dataset(icrl_policy, 
        configuration["expert_episodes"], cost=configuration["cost"])
    acr, acrplot = tools.functions.NormalizedAccrual()({
        "state_reduction": configuration["state_reduction"],
        "dataset": dataset,
        "spaces": configuration["state_action_space"],
        "normalize_func": configuration["normalize_func"],
    })
    acr = np.array(acr).squeeze()
    configuration["accruals"] = acr
    configuration["expert_accruals"] = expert_acr
    configuration["logger"].update({
        "accrual": acrplot.fig,
        "accrual_comparison": configuration["accrual_comparison"](expert_acr, acr),
    })

    dataset = configuration["env"].trajectory_dataset(icrl_policy, 
        configuration["expert_episodes"])
    configuration.update({"agent_dataset": dataset})
    acr, acrplot = tools.functions.NormalizedAccrual()({
        "state_reduction": configuration["state_reduction"],
        "dataset": dataset,
        "spaces": configuration["state_action_space"],
        "normalize_func": configuration["normalize_func"],
    })
    acr = np.array(acr).squeeze()
    configuration["accruals_no_cost"] = acr
    configuration["logger"].update({
        "accrual_no_cost": acrplot.fig,
        "accrual_comparison_no_cost": configuration["accrual_comparison"](expert_acr, acr),
    })

    common.finish(configuration)

if __name__ == "__main__":
    args = read_args()
    train(args)
