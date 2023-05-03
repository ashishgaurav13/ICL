import tools
import wandb
from cppo import CPPO
from ppolag2 import PPOLag2
import common
import gc
import numpy as np
tools.utils.nowarnings()

# Get configuration
configuration = common.get_configuration(method_name="expert")

# Create manual cost function
assert(configuration["cost_condition"] != "")
manual_cost = common.create_manual_cost_function(configuration)
configuration.update({"cost": manual_cost})
_, manualcostmap = \
    manual_cost.outputs(configuration["state_action_space"])
configuration["logger"].update({
    "expert_cost": manualcostmap.fig,
})

# Constrained PPO
algorithm = {
    "CPPO": CPPO,
    "PPOLag2": PPOLag2
}[configuration["forward_crl"]](configuration)
for epoch in range(configuration["ppo_epochs"]):
    metrics = algorithm.train(no_mix=True)
    configuration["logger"].update(metrics)

# Finally, save dataset
dataset = configuration["env"].trajectory_dataset(algorithm.policy, 
    configuration["expert_episodes"], cost=manual_cost, only_success=True, config=configuration)
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
common.finish(configuration)
