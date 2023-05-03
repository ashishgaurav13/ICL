import os
from itertools import accumulate
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import stable_baselines3.common.callbacks as callbacks
import torch as th
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.utils import update_learning_rate
from torch import nn
from tqdm import tqdm
import tools


# =====================================================================================
# GAIL Discriminator
# =====================================================================================


class GailDiscriminator(nn.Module):
    def __init__(
            self,
            obs_dim: int,
            acs_dim: int,
            cost,
            batch_size,
            expert_obs: np.ndarray,
            expert_acs: np.ndarray,
            is_discrete: bool,
            obs_select_dim: Optional[Tuple[int, ...]] = None,
            acs_select_dim: Optional[Tuple[int, ...]] = None,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            clip_obs: Optional[float] = 10.,
            initial_obs_mean: Optional[np.ndarray] = None,
            initial_obs_var: Optional[np.ndarray] = None,
            action_low: Optional[float] = None,
            action_high: Optional[float] = None,
            num_spurious_features: Optional[float] = None,
            freeze_weights: Optional[bool] = False,
            eps: float = 1e-5,
            device: str = "cpu"
    ):
        super(GailDiscriminator, self).__init__()

        self.obs_dim = obs_dim
        self.acs_dim = acs_dim
        self.obs_select_dim = obs_select_dim
        self.acs_select_dim = acs_select_dim

        self._define_input_dims()

        self.num_spurious_features = num_spurious_features

        if self.num_spurious_features is not None:
            # We are not expecting use of spurious features with selected dimensions
            assert (self.obs_select_dim is None and
                    self.acs_select_dim is None)
            assert (self.num_spurious_features > 0)
            self.input_dims += self.num_spurious_features

        self.expert_obs = expert_obs
        self.expert_acs = expert_acs

        self.batch_size = batch_size
        self.is_discrete = is_discrete
        self.clip_obs = clip_obs
        self.device = device
        self.eps = eps
        self.cost = cost

        self.freeze_weights = freeze_weights

        self.current_obs_mean = initial_obs_mean
        self.current_obs_var = initial_obs_var
        self.action_low = action_low
        self.action_high = action_high

        self.current_progress_remaining = 1.
        self.iters = 0
        self.outer_iters = 0

        self._build()

    def _define_input_dims(self) -> None:
        self.select_dim = []
        if self.obs_select_dim is None:
            self.select_dim += [i for i in range(self.obs_dim)]
        elif self.obs_select_dim[0] != -1:
            self.select_dim += self.obs_select_dim
        if self.acs_select_dim is None:
            self.select_dim += [i for i in range(self.acs_dim)]
        elif self.acs_select_dim[0] != -1:
            self.select_dim += self.acs_select_dim
        assert len(self.select_dim) > 0, ""

        self.input_dims = len(self.select_dim)

    def _build(self) -> None:

        # Create network and add sigmoid at the end
        self.network = self.cost.Cost
        print(self.device)
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        self.network.to(self.device)

        # Build optimizer
        self.optimizer = self.cost.Opt

        # Define loss function
        self.loss_fn = nn.BCELoss()

    def forward(self, x: th.tensor) -> th.tensor:
        return self.network(x)

    def flatten(self, x: np.ndarray):
        if len(x.shape) > 2:
            d0, d1 = x.shape[:2]
            return np.reshape(x, (d0 * d1, -1)), (d0, d1)
        # TODO: This if condition is specifically tailored towards
        # plotting and may result in errors otherwise
        # Find a better way to fix this
        elif len(x.shape) == 2:
            return x, (x.shape[0], 1)
        else:
            raise NotImplementedError

    def predict(self, x):
        with th.no_grad():
            out = self.forward(x)
        return out.detach().cpu().numpy()

    def reward_function(self, obs: np.ndarray, acs: np.ndarray, apply_log=True) -> np.ndarray:
        assert obs.shape[-1] == self.obs_dim, ""
        if not self.is_discrete:
            assert acs.shape[-1] == self.acs_dim, ""

        x, orig_shape = self.prepare_nominal_data(obs, acs)
        reward = np.reshape(self.predict(x), orig_shape)
        if apply_log:
            # return np.squeeze(np.log(reward + self.eps))
            return np.log(reward + self.eps)
        else:
            # return np.squeeze(reward)
            return reward

    def call_forward(self, x: np.ndarray):
        with th.no_grad():
            out = self.__call__(th.tensor(x, dtype=th.float32).to(self.device))
        return out

    def train(
            self,
            iterations: np.ndarray,
            nominal_obs: np.ndarray,
            nominal_acs: np.ndarray,
            obs_mean: Optional[np.ndarray] = None,
            obs_var: Optional[np.ndarray] = None,
            current_progress_remaining: float = 1,
    ) -> Dict[str, Any]:

        # Update learning rate
        # if not self.freeze_weights:
        #     self._update_learning_rate(current_progress_remaining)

        # Update normalization stats
        self.current_obs_mean = obs_mean
        self.current_obs_var = obs_var

        # Prepare data
        nominal_data, _ = self.prepare_nominal_data(nominal_obs, nominal_acs)
        expert_data = self.prepare_expert_data(self.expert_obs, self.expert_acs)

        for itr in tqdm(range(iterations)):
            for batch_indices in self.get(min(nominal_data.shape[0], expert_data.shape[0])):
                # Get batch data
                nominal_batch = nominal_data[batch_indices]
                expert_batch = expert_data[batch_indices]

                # Make predictions
                nominal_preds = self.__call__(nominal_batch)
                expert_preds = self.__call__(expert_batch)

                # Calculate loss
                nominal_loss = self.loss_fn(nominal_preds, th.zeros(*nominal_preds.size()).to(self.device))
                expert_loss = self.loss_fn(expert_preds, th.ones(*expert_preds.size()).to(self.device))
                loss = nominal_loss + expert_loss

                # Update
                if not self.freeze_weights:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        self.iters += iterations
        self.outer_iters += 1
        disc_metrics = {"discriminator/disc_loss": loss.item(),
                        "discriminator/expert_loss": expert_loss.item(),
                        "discriminator/nominal_loss": nominal_loss.item(),
                        "discriminator/mean_nominal_preds": nominal_preds.mean().item(),
                        "discriminator/mean_expert_preds": expert_preds.mean().item(),
                        "discriminator/iters": self.outer_iters,
                        "discriminator/total_iters": self.iters}
        return disc_metrics

    def select_appropriate_dims(self, x: Union[np.ndarray, th.tensor]) -> Union[np.ndarray, th.tensor]:
        return x[..., self.select_dim]

    def add_spurious_features(
            self,
            x,
            y,  # string indicating nominal or expert
            num_features,  # number of spurious bits to add
    ):
        new_shape = list(x.shape)
        new_shape[-1] += num_features
        if y == 'expert':
            z = np.zeros(new_shape)
            z[..., :-num_features] = x
        elif y == 'nominal':
            z = np.ones(new_shape)
            z[..., :-num_features] = x
        return z

    def prepare_nominal_data(
            self,
            obs: np.ndarray,
            acs: np.ndarray,
    ) -> th.tensor:

        # We are expecting obs to have shape [batch_size, n_envs, obs_dim]
        obs, orig_shape = self.flatten(obs)
        acs, _ = self.flatten(acs)
        # No need to normalize as nominal obs are already normalized
        # obs = self.normalize_obs(obs, self.current_obs_mean, self.current_obs_var, self.clip_obs)
        # acs = self.reshape_actions(acs)
        #        acs = self.clip_actions(acs, self.action_low, self.action_high)

        obs = th.tensor(np.expand_dims(obs, axis=0), dtype=th.float32).to(self.device)
        acs = th.tensor(np.expand_dims(acs, axis=0), dtype=th.float32).to(self.device)
        if len(acs.shape) == 2:
            acs = acs.unsqueeze(-1)
        obs = self.cost.vector_state_reduction(obs)
        acs = self.cost.vector_action_reduction(acs)
        concat = self.cost.vector_input_format(obs, acs)
        concat = concat.squeeze(0)
        # concat = self.select_appropriate_dims(np.concatenate([obs, acs], axis=-1))
        # if self.num_spurious_features is not None:
        #     concat = self.add_spurious_features(concat, 'nominal', self.num_spurious_features)
        return concat, orig_shape

    def prepare_expert_data(
            self,
            obs: np.ndarray,
            acs: np.ndarray,
    ) -> th.tensor:

        obs = self.normalize_obs(obs, self.current_obs_mean, self.current_obs_var, self.clip_obs)
        # acs = self.reshape_actions(acs)
        #        acs = self.clip_actions(acs, self.action_low, self.action_high)
        obs = th.tensor(np.expand_dims(obs, axis=0), dtype=th.float32).to(self.device)
        acs = th.tensor(np.expand_dims(acs, axis=0), dtype=th.float32).to(self.device)
        if len(acs.shape) == 2:
            acs = acs.unsqueeze(-1)
        obs = self.cost.vector_state_reduction(obs)
        acs = self.cost.vector_action_reduction(acs)
        concat = self.cost.vector_input_format(obs, acs)
        concat = concat.squeeze(0)
        # concat = np.concatenate([obs, acs], axis=-1)
        # concat = self.select_appropriate_dims(np.concatenate([obs, acs], axis=-1))
        # if self.num_spurious_features is not None:
        #     concat = self.add_spurious_features(concat, 'expert', self.num_spurious_features)
        return concat

    def normalize_obs(self, obs: np.ndarray, mean: Optional[float] = None, var: Optional[float] = None,
                      clip_obs: Optional[float] = None) -> np.ndarray:
        #        if mean is not None and var is not None:
        #            mean, var = mean[None], var[None]
        #            obs = (obs - mean) / np.sqrt(var + self.eps)
        #        if clip_obs is not None:
        #            obs = np.clip(obs, -self.clip_obs, self.clip_obs)
        #
        return obs

    def reshape_actions(self, acs):
        if self.is_discrete:
            acs_ = acs.astype(int)
            if len(acs.shape) > 1:
                acs_ = np.squeeze(acs_, axis=-1)
            acs = np.zeros([acs.shape[0], self.acs_dim])
            acs[np.arange(acs_.shape[0]), acs_] = 1.

        return acs

    def clip_actions(self, acs: np.ndarray, low: Optional[float] = None, high: Optional[float] = None) -> np.ndarray:
        if high is not None and low is not None:
            acs = np.clip(acs, low, high)

        return acs

    def get(self, size: int) -> np.ndarray:
        indices = np.random.permutation(size)

        batch_size = self.batch_size
        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = size

        start_idx = 0
        while start_idx < size:
            batch_indices = indices[start_idx:start_idx + batch_size]
            yield batch_indices
            start_idx += batch_size

    def _update_learning_rate(self, current_progress_remaining) -> None:
        self.current_progress_remaining = current_progress_remaining
        update_learning_rate(self.optimizer, self.lr_schedule(current_progress_remaining))


# =====================================================================================
# GAIL CALLBACK
# =====================================================================================

class GailCallback(callbacks.BaseCallback):
    """At the end of rollouts (but before policy update), this callback
       (1) Updates the discriminator using the rollouts.
       (2) Relabels rewards using the updated discriminator and inserts them into buffer.
       (3) Calls buffer's 'compute_returns_and_advantage() function to recompute
           advatnage using updated buffer.
       (4) Makes a plot of GAIL discriminator if possible."""

    def __init__(
            self,
            gail_logger,
            configuration,
            policy,
            plot_interval,
            discriminator,
            learn_cost: bool,
            # true_cost_function,
            # save_dir: str,
            plot_disc,
            update_freq: int = 1,
            verbose: int = 1,
    ):
        super(GailCallback, self).__init__(verbose)
        self.discriminator = discriminator
        self.update_freq = update_freq
        self.learn_cost = learn_cost
        # self.true_cost_function = true_cost_function
        # self.plot_save_dir = save_dir
        self.plot_disc = plot_disc
        self.gail_logger = gail_logger
        self.configuration = configuration
        self.policy = policy
        self.plot_interval = plot_interval

    def _init_callback(self):
        self.disc_itr = 0
        # if self.plot_disc:
        #     del_and_make(os.path.join(self.plot_save_dir, "discriminator"))
        #     self.plot_folder = os.path.join(self.plot_save_dir, "discriminator")
            # self.discriminator.plot_expert(os.path.join(self.plot_folder,'expert.png'))

    def _on_step(self):
        pass

    def _on_rollout_end(self):
        # Get data from buffer
        obs = self.model.rollout_buffer.observations.copy()
        acs = self.model.rollout_buffer.actions.copy()
        # unnormalized_obs = self.training_env.unnormalize_obs(obs)
        # obs = unnormalized_obs

        if self.disc_itr % self.update_freq == 0:  # Get the current mean and variance
            # self.discriminator.current_obs_mean = self.training_env.obs_rms.mean
            # self.discriminator.current_obs_var = self.training_env.obs_rms.var
            # update discriminator
            metrics = self.discriminator.train(1, obs, acs)
            self.gail_logger.update(metrics)
            # if self.plot_disc:
            #     save_name = os.path.join(self.plot_folder, str(self.disc_itr) + '.png')
            #     self.discriminator.plot_HC(save_name, nominal_obs=unnormalized_obs)

        # Log cost on true cost function
        obs_dim = obs.shape[-1] # unnormalized_obs.shape[-1]
        acs_dim = acs.shape[-1]
        # tmp1 = self.true_cost_function(
        #     unnormalized_obs.reshape((-1, obs_dim)),
        #     acs.reshape((-1, acs_dim))
        # )
        # tmp2 = unnormalized_obs.reshape((-1, obs_dim))
        # true_average_cost = np.mean(self.true_cost_function(
        #     unnormalized_obs.reshape((-1, obs_dim)),
        #     acs.reshape((-1, acs_dim))
        # ))
        # self.logger.record('eval/mean_cost', true_average_cost)

        # self.logger.write(metrics, {k: None for k in metrics.keys()}, step=self.num_timesteps)
        rewards = self.discriminator.reward_function(obs, acs)
        current_reward_shape = self.model.rollout_buffer.rewards.shape
        assert (rewards.shape == current_reward_shape)
        if self.learn_cost:
            self.model.rollout_buffer.rewards += rewards
        else:
            self.model.rollout_buffer.rewards = rewards
        # recompute returns and advantages
        last_values, dones = (self.model.extras['last_values'],
                              self.model.extras['dones'])
        self.model.rollout_buffer.compute_returns_and_advantage(last_values, dones)
        self.disc_itr += 1  # Use for plotting freq

        if self.disc_itr % self.plot_interval == 0:

            configuration = self.configuration
            cost = configuration["cost"]
            expert_acr = configuration["expert_acr"]
            costvalues, costmap = cost.outputs(configuration["state_action_space"], invert=True)
            costvalues = np.array(costvalues).squeeze()
            configuration["logger"].update({"cost": costmap.fig})
            if configuration["cost_condition"] != "":
                manualcostvalues = configuration["manualcostvalues"]
                configuration["logger"].update({"cost_comparison": \
                    configuration["cost_comparison"](manualcostvalues, costvalues)})

            dataset = configuration["env"].trajectory_dataset(self.policy, 
                configuration["expert_episodes"], cost=configuration["cost"], 
                is_torch_policy=configuration["is_torch_policy"])
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

            dataset = configuration["env"].trajectory_dataset(self.policy, 
                configuration["expert_episodes"], is_torch_policy=configuration["is_torch_policy"])
            configuration.update({"agent_dataset": dataset})
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