import tools
import gc
import numpy as np
tools.utils.nowarnings()

# Get configuration
configuration = tools.utils.common.get_configuration(method_name="expert")

# Create manual cost function
assert(configuration["cost_condition"] != "")
manual_cost = tools.utils.common.create_manual_cost_function(configuration)
configuration.update({"cost": manual_cost})
_, manualcostmap = \
    manual_cost.outputs(configuration["state_action_space"])
configuration["logger"].update({
    "expert_cost": manualcostmap.fig,
})

# Constrained PPO
algorithm = {
    "CPPO": tools.algorithms.CPPO,
    "PPOLag": tools.algorithms.PPOLag,
}[configuration["forward_crl"]](configuration)
for epoch in range(configuration["ppo_epochs"]):
    metrics = algorithm.train(no_mix=True)
    configuration["logger"].update(metrics)

# Finally, save dataset
dataset = configuration["env"].trajectory_dataset(algorithm.policy, 
    configuration["expert_episodes"], cost=manual_cost, only_success=True, config=configuration,
    is_torch_policy=configuration["is_torch_policy"])
dataset.save()
acr, acrplot = tools.functions.NormalizedAccrual()({
    "state_reduction": configuration["state_reduction"],
    "dataset": dataset,
    "spaces": configuration["state_action_space"],
    "normalize_func": configuration["normalize_func"],
})
acr = np.array(acr).squeeze()
configuration["accruals"] = acr
configuration["expert_accruals"] = acr
configuration["logger"].update({"expert_accrual": acrplot.fig})

# Finally
del configuration.data["cost"]
gc.collect()
tools.utils.common.finish(configuration)
