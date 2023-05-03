import os
import sys
import torch
import tools
import gym
import numpy as np
import sys; sys.path += ["baselines"]

from baselines.constraint_models.constraint_net.gail_net import GailDiscriminator, GailCallback
from baselines.stable_baselines3 import PPO
from baselines.stable_baselines3.common.utils import get_schedule_fn
from baselines.utils.data_utils import read_args, load_config
from baselines.utils.model_utils import load_ppo_config

tools.utils.nowarnings()

def train(args):
    configuration, seed = load_config(args)
    configuration["seed"] = seed
    configuration = tools.data.Configuration(tools.utils.convert_lambdas(configuration))
    state_action_space = tools.environments.get_state_action_space(
        configuration["env_type"], configuration["env_id"])
    configuration.update({"state_action_space": state_action_space})
    config_name = os.path.splitext(os.path.basename(args.c))[0]
    logdir = "%s(%s)-%s-%s-(%.2f,%d)" % ("GAIL",
        "GC", config_name.split("-")[-1], tools.utils.timestamp(), 
        0, configuration["seed"])
    logger = tools.data.Logger(window=configuration["window"], logdir=logdir)
    configuration.update({"logger": logger})

    # Create manual cost function
    if configuration["cost_condition"] != "":
        manual_cost = tools.common.create_manual_cost_function(configuration)
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

    discriminator = GailDiscriminator(
        obs_dim,
        acs_dim,
        configuration["cost"],
        configuration['PPO']['batch_size'],
        expert_obs,
        expert_acs,
        is_discrete,
        obs_select_dim=None,
        acs_select_dim=None,
        clip_obs=configuration['DISC']['clip_obs'],
        initial_obs_mean=None,
        initial_obs_var=None,
        action_low=action_low,
        action_high=action_high,
        num_spurious_features=None,
        freeze_weights=False,
        eps=float(configuration['DISC']['disc_eps']),
        device=configuration['t'].device,
    )

    # true_cost_function = get_true_cost_function(configuration['env']['eval_env_id'])

    # costShapingCallback = CostShapingCallback(obs_dim,
    #                                           acs_dim,
    #                                           use_nn_for_shaping=configuration['DISC']['use_cost_net'])
    # all_callbacks = [costShapingCallback]

    # Define and train model
    ppo_parameters = load_ppo_config(config=configuration, train_env=train_env, seed=seed, log_file=None)
    model = PPO(logger, **ppo_parameters)

    class GAILPolicy(tools.base.Policy):
        def act(self, s):
            return model.policy.forward(torch.as_tensor([s]).to(configuration['t'].device))[0].detach().view(-1).cpu().numpy()
    policy = GAILPolicy()

    gail_update = GailCallback(logger, configuration, policy, configuration['plot_interval'],
                                discriminator=discriminator,
                                learn_cost=configuration['DISC']['learn_cost'],
                                plot_disc=False)
    all_callbacks = [gail_update]

    # Train
    try:
        model.learn(total_timesteps=int(configuration['PPO']['timesteps']),
                    callback=all_callbacks)
    except:
        pass

    costvalues, costmap = cost.outputs(configuration["state_action_space"], invert=True)
    costvalues = np.array(costvalues).squeeze()
    configuration["logger"].update({"cost": costmap.fig})
    if configuration["cost_condition"] != "":
        configuration["logger"].update({"cost_comparison": \
            configuration["cost_comparison"](manualcostvalues, costvalues)})

    dataset = configuration["env"].trajectory_dataset(policy, 
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

    dataset = configuration["env"].trajectory_dataset(policy, 
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

    tools.common.finish(configuration)

if __name__ == "__main__":
    args = read_args()
    train(args)
