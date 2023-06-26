import tools
import tqdm
import numpy as np

class StepbystepPolicy(tools.base.Policy):
    def act(self, s):
        return None

def play_episode(env, policy):
    S, A = [], []
    S.append(env.reset())
    done = False
    while not done:
        assert(S[-1] != None)
        action = policy.act(S[-1])
        step_data = env.step(action)
        A.append(np.array([step_data["info"]["action"]]))
        assert(A[-1] != None)
        S.append(step_data["next_state"])
        done = step_data["done"]
    return S, A

data = []
env = tools.environments.create('driving', 'exiD', normalize_states=False, normalize_actions=False, time_limit=1000)
pbar = tqdm.trange(200)
count = 0
for _ in pbar:
    policy = StepbystepPolicy()
    S, A = play_episode(env, policy)
    data += [[S[:-1], A]]
    # print(data[-1])
    count += 1
    pbar.set_description("%d/%d" % (count, 200))
    pbar.refresh()
expert_dataset = tools.base.TrajectoryDataset(data)
print(len(expert_dataset))
expert_dataset.save()