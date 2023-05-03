import tools
import numpy as np
import torch

# Get configuration
configuration = tools.common.get_configuration(method_name="icl-mix-improved")

# Create manual cost function
if configuration["cost_condition"] != "":
    manual_cost = tools.common.create_manual_cost_function(configuration)
    manualcostvalues, manualcostmap = \
        manual_cost.outputs(configuration["state_action_space"])
    manualcostvalues = np.array(manualcostvalues).squeeze()
    configuration["logger"].update({
        "expert_cost": manualcostmap.fig,
    })

# Create cost function
cost = tools.functions.CostFunction(configuration, i=configuration["i"], h=64, o=1)
configuration.update({"cost": cost})
costvalues, costmap = cost.outputs(configuration["state_action_space"])
costvalues = np.array(costvalues).squeeze()
configuration["logger"].update({"cost": costmap.fig})
if configuration["cost_condition"] != "":
    configuration["logger"].update({"cost_comparison": \
        configuration["cost_comparison"](manualcostvalues, costvalues)})

# Expert dataset accrual + train flow
expert_dataset = tools.base.TrajectoryDataset.load()
eS = configuration["vector_state_reduction"](expert_dataset.S)
eA = configuration["vector_action_reduction"](expert_dataset.A)
eSA = configuration["vector_input_format"](eS, eA).view(-1, configuration["i"])[
        torch.nonzero(expert_dataset.M.view(-1)).view(-1)]
flow = tools.functions.create_flow(configuration, eSA, "realnvp", configuration["i"])
for flowepoch in range(configuration["flow_epochs"]):
    configuration["logger"].update(flow.train())
ep = -flow.log_probs(eSA)
em, es = ep.mean(), ep.std()
configuration.update({"flow": flow, "expert_nll": (em.item(), es.item())})
configuration["logger"].update({"expert_nll": (em.item(), es.item())})
expert_acr, expert_acrplot = tools.functions.NormalizedAccrual()({
    "state_reduction": configuration["state_reduction"],
    "dataset": expert_dataset,
    "spaces": configuration["state_action_space"],
    "normalize_func": configuration["normalize_func"],
})
expert_acr = np.array(expert_acr).squeeze()
configuration["logger"].update({"expert_accrual": expert_acrplot.fig})

# Alternating process
for outer_epoch in range(configuration["outer_epochs"]):

    # Constrained PPO
    algorithm = {
        "CPPO": tools.algorithms.CPPO,
        "PPOLag": tools.algorithms.PPOLag,
    }[configuration["forward_crl"]](configuration)
    for epoch in range(configuration["ppo_epochs"]):
        metrics = algorithm.train()
        configuration["logger"].update(metrics)
    dataset = configuration["env"].trajectory_dataset(algorithm.policy, 
        configuration["expert_episodes"], weights=configuration["past_pi_weights"],
        p=configuration["past_pi_dissimilarities"], config=configuration)
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

    # Cost adjustment
    adjustment = tools.algorithms.CostAdjustment(configuration)
    for inner_epoch in range(configuration["updates_per_epoch"]):
        metrics = adjustment.train()
        configuration["logger"].update(metrics)
    costvalues, costmap = cost.outputs(configuration["state_action_space"])
    costvalues = np.array(costvalues).squeeze()
    configuration["logger"].update({"cost": costmap.fig})
    if configuration["cost_condition"] != "":
        configuration["logger"].update({"cost_comparison": \
            configuration["cost_comparison"](manualcostvalues, costvalues)})

# Constrained PPO
algorithm = {
    "CPPO": tools.algorithms.CPPO,
    "PPOLag": tools.algorithms.PPOLag,
}[configuration["forward_crl"]](configuration)
for epoch in range(configuration["ppo_epochs"]):
    metrics = algorithm.train(no_mix=True)
    configuration["logger"].update(metrics)
dataset = configuration["env"].trajectory_dataset(algorithm.policy, 
    configuration["expert_episodes"], cost=configuration["cost"], config=configuration)
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
    "ppo_accrual": acrplot.fig,
    "ppo_accrual_comparison": configuration["accrual_comparison"](expert_acr, acr),
})

# Constrained PPO no cost
dataset = configuration["env"].trajectory_dataset(algorithm.policy, 
    configuration["expert_episodes"], config=configuration)
acr, acrplot = tools.functions.NormalizedAccrual()({
    "state_reduction": configuration["state_reduction"],
    "dataset": dataset,
    "spaces": configuration["state_action_space"],
    "normalize_func": configuration["normalize_func"],
})
acr = np.array(acr).squeeze()
configuration["accruals_no_cost"] = acr
configuration["logger"].update({
    "ppo_accrual_no_cost": acrplot.fig,
    "ppo_accrual_comparison_no_cost": configuration["accrual_comparison"](expert_acr, acr),
})

# Finally
tools.common.finish(configuration)
