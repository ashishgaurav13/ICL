import tools
import argparse
import os
import torch

def get_configuration(method_name=None, config_name=None):
    """
    Get arguments from command line, then load configuration.
    """
    tools.utils.nowarnings()
    assert(method_name != None)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, required=True)
    parser.add_argument("-beta", type=float, default=-1.) # If -1, do not change it
    parser.add_argument("-seed", type=int, default=1)
    args = parser.parse_args()
    update_params = {"seed": args.seed}
    if args.beta != -1.: update_params["beta"] = args.beta
    configuration = tools.data.Configuration.from_json(args.c, update_params)
    state_action_space = tools.environments.get_state_action_space(
        configuration["env_type"], configuration["env_id"])
    configuration.update({"state_action_space": state_action_space})
    config_name = os.path.splitext(os.path.basename(args.c))[0]
    configuration.update({"config_name": config_name})
    logdir = "%s(%s)-%s-%s-(%.2f,%d)" % (method_name,
        configuration["forward_crl"], config_name, tools.utils.timestamp(), 
        configuration["beta"], configuration["seed"])
    logger = tools.data.Logger(window=configuration["window"], logdir=logdir)
    configuration.update({"logger": logger})
    if configuration["forward_crl"] in ["CPPO"]:
        configuration["is_torch_policy"] = True
    elif configuration["forward_crl"] in ["PPOLag"]:
        configuration["is_torch_policy"] = False
    return configuration

def create_manual_cost_function(configuration):
    class ManualCostFunction(tools.base.Function):
        beta = configuration["beta"]
        discount_factor = configuration["discount_factor"]
        def __call__(self, sa):
            s, a = sa
            if configuration["cost_condition"](s, a):
                return 1.
            return 0.
    manual_cost = ManualCostFunction()
    return manual_cost

def finish(configuration):
    if "accruals" in configuration.data.keys():
        anc = []
        if "accruals_no_cost" in configuration.data.keys():
            anc += [configuration["accruals_no_cost"]]
        torch.save([configuration["accruals"], configuration["expert_accruals"], *anc], "%s-acr.pt" % \
                   configuration["logger"].logdir)
    if "cost" in configuration.data.keys():
        configuration["cost"].save("%s-cost.pt" % configuration["logger"].logdir)
    if "agent_dataset" in configuration.data.keys():
        configuration["agent_dataset"].save("%s-agent-dataset.pt" % configuration["logger"].logdir)

def make_table(d, cols):
    l = []
    for env in d.keys():
        firstenv = True
        for algo in d[env].keys():
            firstalgo = True
            for beta in d[env][algo].keys():
                firstbeta = True
                for seed in d[env][algo][beta].keys():
                    firstseed = True
                    s = ""
                    if firstenv:
                        s += "<td>%s</td>" % env
                        firstenv = False
                    else:
                        s += "<td/>"
                    if firstalgo:
                        s += "<td>%s</td>" % algo
                        firstalgo = False
                    else:
                        s += "<td/>"
                    if firstbeta:
                        s += "<td>%s</td>" % beta
                        firstbeta = False
                    else:
                        s += "<td/>"
                    if firstseed:
                        s += "<td>%s</td>" % seed
                        firstseed = False
                    else:
                        s += "<td/>"
                    for item in d[env][algo][beta][seed]:
                        s += "<td>%s</td>" % item
                    l += ["<tr>%s</tr>" % s]
    ll = ""
    for item in cols:
        ll += "<th>%s</th>" % item
    ll = "<tr>%s</tr>" % ll
    return "<table padding=2>%s%s</table>" % (ll, "".join(l))
