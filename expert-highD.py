import tools
import tqdm
import numpy as np

data = []
env = tools.environments.HighDSampleEnvironmentWrapper(timelimit=5000, sequential=True)
pbar = tqdm.tqdm(env.possibleegoids)
total = 0
count = 0
use_real = True
for _ in pbar:
    class StepbystepPolicy(tools.base.Policy):
        def act(self, s):
            if not hasattr(self, 'acc'):
                # print("Playing with eid=%d" % env.possibleegoids[(env.eid-1)%len(env.possibleegoids)])
                eid = (env.eid-1) % len(env.possibleegoids)
                self.acc = np.sqrt(env.agents[eid][6]**2+env.agents[eid][7]**2)
                self.cpts = env.agents[eid][1]
                self.v = np.sqrt(env.agents[eid][4]**2+env.agents[eid][5]**2)
                self.delta = 0
                self.distances = []
                self.t = 0
            self.distances += [s[-1]]
            curr_acc = self.acc[self.t]
            if use_real:
                ox, oy, scale = env.env.canvas.ox, env.env.canvas.oy, env.env.canvas.scale
                x = ox + float(self.cpts[self.t][0]) * scale
                y = oy + float(self.cpts[self.t][1]) * scale
                env.env.agents[env.env.ego_id].f['x'] = self.cpts[self.t][0]
                env.env.agents[env.env.ego_id].f['y'] = self.cpts[self.t][1]
                env.env.agents[env.env.ego_id].f['v'] = self.v[self.t]
                env.env.agents[env.env.ego_id].f['acc'] = curr_acc
                env.env.agents[env.env.ego_id].items[0].items[0].update(x = x, y = y)
            else:
                if len(self.distances) >= 2 and self.distances[-2] != -1 and self.distances[-1] != -1:
                    if self.distances[-1] > self.distances[-2]:
                        self.delta += 0.1
                    elif self.distances[-1] < self.distances[-2]:
                        self.delta -= 0.1
                else:
                    self.delta = 0.
            # if self.t > 0:
            #     print(s[0], s[1], self.cpts[self.t-1], self.cpts[self.t], self.cpts.shape[0])
            # else:
            #     print(s[0], s[1], self.cpts[self.t], self.cpts.shape[0])
            if (self.t+1) < len(self.acc): self.t = self.t+1
            if use_real:
                return np.array([curr_acc, 0, np.nan])
            else:
                return np.array([curr_acc+self.delta, 0])
    policy = StepbystepPolicy()
    S, A, R, Info = env.play_episode(policy, info=True)
    data += [[[s[[0, 1, 2, -1]] for s in S], [a[0:2] for a in A]]]
    # print(Info)
    if 'mode' in Info.keys() and Info['mode'] == 'success':
        count += 1
    total += 1
    pbar.set_description("%d/%d" % (count, total))
    pbar.refresh()
expert_dataset = tools.base.TrajectoryDataset(data)
print(len(expert_dataset))
expert_dataset.save()
