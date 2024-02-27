import copy
import os
from itertools import accumulate
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.utils import update_learning_rate
from torch.distributions.categorical import Categorical
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F

from utils.data_utils import idx2vector

class ConstraintGFlowNet(nn.Module):
    def __init__(
            self,
            obs_dim: int,
            acs_dim: int,
            hidden_sizes: Tuple[int, ...],
            batch_size: int,
            lr_schedule: Callable[[float], float],
            expert_obs: np.ndarray,
            expert_acs: np.ndarray,
            is_discrete: bool,
            task: str = 'ICRL',
            regularizer_coeff: float = 0.,
            obs_select_dim: Optional[Tuple[int, ...]] = None,
            acs_select_dim: Optional[Tuple[int, ...]] = None,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            no_importance_sampling: bool = True,
            per_step_importance_sampling: bool = False,
            clip_obs: Optional[float] = 10.,
            initial_obs_mean: Optional[np.ndarray] = None,
            initial_obs_var: Optional[np.ndarray] = None,
            action_low: Optional[float] = None,
            action_high: Optional[float] = None,
            target_kl_old_new: float = -1,
            target_kl_new_old: float = -1,
            train_gail_lambda: Optional[bool] = False,
            eps: float = 1e-5,
            recon_obs: bool = False,
            env = None,
            env_configs: dict = None,
            device: str = "cpu",
            log_file= None,
            gflownet_sizes : Tuple[int, ...] = [64, 64],
            gflownet_lr: float = 0.001,
            generated_weight: float = 0.001,
            generated_mode: str = 'traj'
    ):
        super(ConstraintGFlowNet, self).__init__()
        self.task = task
        self.obs_dim = obs_dim
        self.acs_dim = acs_dim
        self.obs_select_dim = obs_select_dim
        self.acs_select_dim = acs_select_dim
        self._define_input_dims()
        self.expert_obs = expert_obs
        self.expert_acs = expert_acs
        self.hidden_sizes = hidden_sizes
        self.batch_size = batch_size
        self.is_discrete = is_discrete
        self.regularizer_coeff = regularizer_coeff
        self.importance_sampling = not no_importance_sampling
        self.per_step_importance_sampling = per_step_importance_sampling
        self.clip_obs = clip_obs
        self.device = device
        self.eps = eps
        self.env = env
        self.recon_obs = recon_obs
        self.env_configs = env_configs
        self.train_gail_lambda = train_gail_lambda
        self.gflownet_sizes = gflownet_sizes
        self.gflownet_lr = gflownet_lr
        self.generated_weight = generated_weight
        self.generated_mode = generated_mode

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_class = optimizer_class
        self.lr_schedule = lr_schedule

        self.current_obs_mean = initial_obs_mean
        self.current_obs_var = initial_obs_var
        self.action_low = action_low
        self.action_high = action_high

        self.target_kl_old_new = target_kl_old_new
        self.target_kl_new_old = target_kl_new_old

        self.current_progress_remaining = 1.
        self.log_file = log_file

        self._build()

    def _define_input_dims(self) -> None:
        self.input_obs_dim = []
        self.input_acs_dim = []
        if self.obs_select_dim is None:
            self.input_obs_dim += [i for i in range(self.obs_dim)]
        elif self.obs_select_dim[0] != -1:
            self.input_obs_dim += self.obs_select_dim
        obs_len = len(self.input_obs_dim)
        if self.acs_select_dim is None:
            self.input_acs_dim += [i for i in range(self.acs_dim)]
        elif self.acs_select_dim[0] != -1:
            self.input_acs_dim += self.acs_select_dim
        self.select_dim = self.input_obs_dim + [i + obs_len for i in self.input_acs_dim]
        self.input_dims = len(self.select_dim)
        assert self.input_dims > 0, ""

    def _build(self) -> None:

        # Create network and add sigmoid at the end
        gflownet = create_mlp(input_dim=self.input_dims,
                              output_dim=2*self.acs_dim,
                              net_arch=self.gflownet_sizes,
                              activation_fn = nn.LeakyReLU)
        self.gflownet = nn.Sequential(*gflownet)
        self.gflownet.to(self.device)
        self.logZ = nn.Parameter(th.ones(1)).to(self.device)

        self.network = nn.Sequential(
            *create_mlp(self.input_dims, 1, self.hidden_sizes),
            nn.Sigmoid()
        )
        self.network.to(self.device)

        # Build optimizer
        if self.optimizer_class is not None:
            self.optimizer = self.optimizer_class(self.parameters(), lr=self.lr_schedule(1), **self.optimizer_kwargs)
            self.gflow_optimizer= th.optim.Adam(self.gflownet.parameters(), lr=self.gflownet_lr)
        else:
            self.optimizer = None
            self.gflow_optimizer = None
        if self.train_gail_lambda:
            self.criterion = nn.BCELoss()

    def forward(self, x: th.tensor) -> th.tensor:
        return self.network(x)

    def gflow_forward(self, x: th.tensor) -> th.tensor:
        logits = self.gflownet(x)
        P_F = logits[..., :self.acs_dim]
        P_B = logits[..., self.acs_dim:]
        return P_F, P_B

    def cost_function(self, obs: np.ndarray, acs: np.ndarray, force_mode: str = None) -> np.ndarray:
        assert self.recon_obs or obs.shape[-1] == self.obs_dim, ""
        if not self.is_discrete:
            assert acs.shape[-1] == self.acs_dim, ""

        x = self.prepare_data(obs, acs)
        with th.no_grad():
            out = self.__call__(x)
        cost = 1 - out.detach().cpu().numpy()
        return cost.squeeze(axis=-1)

    def call_forward(self, x: np.ndarray):
        with th.no_grad():
            out = self.__call__(th.tensor(x, dtype=th.float32).to(self.device))
        return out

    def train_gridworld_nn(
            self,
            iterations: np.ndarray,
            nominal_obs: np.ndarray,
            nominal_acs: np.ndarray,
            episode_lengths: np.ndarray,
            obs_mean: Optional[np.ndarray] = None,
            obs_var: Optional[np.ndarray] = None,
            env_configs: Dict = None,
            current_progress_remaining: float = 1,
    ) -> Dict[str, Any]:
        # Update learning rate
        self._update_learning_rate(current_progress_remaining)

        # Update normalization stats
        self.current_obs_mean = obs_mean
        self.current_obs_var = obs_var
        # Prepare data
        nominal_data_games = [self.prepare_data(nominal_obs[i], nominal_acs[i])
                              for i in range(len(nominal_obs))]
        expert_data_games = [self.prepare_data(self.expert_obs[i], self.expert_acs[i])
                             for i in range(len(self.expert_obs))]
        early_stop_itr = iterations
        # loss = th.tensor(np.inf)
        for itr in tqdm(range(iterations)):
            for gid in range(min(len(nominal_data_games), len(expert_data_games))):
                # train gflownet
                nominal_data = nominal_data_games[gid]
                expert_data = expert_data_games[gid]

                clip = min(len(nominal_data),len(expert_data))
                nominal_data = nominal_data[0:clip, ]
                expert_data = expert_data[0:clip, ]

                # calulate distance between nominal and expert
                # euclidean_distance = (th.sqrt(th.sum((nominal_data - expert_data) ** 2)))
                reward_list = [0]* (clip-1)
                reward_list.append(1)

                P_F_s, P_B_s = self.gflow_forward(nominal_data[0])
                total_P_F = 0
                total_P_B = 0
                gflow_loss=0
                for i in range(1, clip):
                    cat = Categorical(logits=P_F_s)
                    action = cat.sample()
                    total_P_F = total_P_F + cat.log_prob(F.softmax(action.float(), dim=-1))

                    new_state = nominal_data[i]
                    reward = reward_list[i]
                    if reward != 0:
                        break
                    P_F_s, P_B_s = self.gflow_forward(new_state)
                    total_P_B = total_P_B + Categorical(logits=P_B_s).log_prob(F.softmax(action.float(), dim=-1))
                    loss = th.mean((self.logZ + total_P_F - th.log(
                        th.tensor(reward).float()).clip(-20) - total_P_B).pow(2))
                    gflow_loss = gflow_loss + loss

                self.gflow_optimizer.zero_grad()
                gflow_loss.backward()
                # print(density_loss)
                self.gflow_optimizer.step()

            # generate trajectories
            if self.generated_mode == 'traj':
                num_generate_trajs = min(len(nominal_data_games), len(expert_data_games))
                all_generated_orig_obs, all_generated_obs, all_generated_acs = [], [], [],
                for i in range(num_generate_trajs):
                    # Avoid double reset, as VecEnv are reset automatically
                    if i == 0:
                        obs = self.env.reset()
                    done = False
                    origin_obs_game = []
                    obs_game = []
                    acs_game = []

                    while not done:
                        with th.no_grad():
                            action = self.select_action(obs)
                        origin_obs_game.append(self.env.get_original_obs())
                        obs_game.append(obs)
                        acs_game.append(action)
                        obs, _, done, _ = self.env.step(action)

                    origin_obs_game = np.squeeze(np.array(origin_obs_game), axis=1)
                    obs_game = np.squeeze(np.array(obs_game), axis=1)
                    acs_game = np.squeeze(np.array(acs_game), axis=1)

                    all_generated_orig_obs.append(origin_obs_game)
                    all_generated_obs.append(obs_game)
                    all_generated_acs.append(acs_game)

                # Prepare generated data
                generated_data_games = [self.prepare_data(all_generated_orig_obs[i], all_generated_acs[i])
                                      for i in range(len(all_generated_orig_obs))]

            for gid in range(min(len(nominal_data_games), len(expert_data_games))):
                nominal_data = nominal_data_games[gid]
                expert_data = expert_data_games[gid]
                generated_data = generated_data_games[gid]

                # Save current network predictions if using importance sampling
                if self.importance_sampling:
                    with th.no_grad():
                        start_preds = self.forward(nominal_data).detach()
                        generated_start_preds = self.forward(generated_data).detach()

                # Compute IS weights
                if self.importance_sampling:
                    with th.no_grad():
                        current_preds = self.forward(nominal_data).detach()
                        generated_current_preds = self.forward(generated_data).detach()
                    is_weights, kl_old_new, kl_new_old = self.compute_is_weights(start_preds.clone(),
                                                                                 current_preds.clone(),
                                                                                 episode_lengths)
                    generated_is_weights, generated_kl_old_new, generated_kl_new_old = \
                        self.compute_is_weights(generated_start_preds.clone(),
                                                generated_current_preds.clone(),
                                                episode_lengths)
                    # Break if kl is very large
                    if ((self.target_kl_old_new != -1 and kl_old_new > self.target_kl_old_new) or
                            (self.target_kl_new_old != -1 and kl_new_old > self.target_kl_new_old)):
                            # (self.target_kl_old_new != -1 and generated_kl_old_new > self.target_kl_old_new) or
                            # (self.target_kl_new_old != -1 and generated_kl_new_old > self.target_kl_new_old)):
                        early_stop_itr = itr
                        break
                else:
                    is_weights = th.ones(nominal_data.shape[0]).to(self.device)
                    generated_is_weights = th.ones(generated_data.shape[0]).to(self.device)

                nominal_preds_all = []
                generated_preds_all = []
                expert_preds_all = []
                # Do a complete pass on the game data
                for nom_batch_indices, generated_batch_indices, exp_batch_indices in self.get_three(nominal_data.shape[0],
                                                                           generated_data.shape[0],
                                                                           expert_data.shape[0]):
                    # Get batch data
                    nominal_batch = nominal_data[nom_batch_indices]
                    generated_batch = generated_data[generated_batch_indices]
                    expert_batch = expert_data[exp_batch_indices]
                    is_batch = is_weights[nom_batch_indices][..., None]
                    generated_is_batch = generated_is_weights[generated_batch_indices][..., None]

                    # Make predictions
                    nominal_preds = self.__call__(nominal_batch)
                    nominal_preds_all.append(nominal_preds)
                    generated_preds = self.__call__(generated_batch)
                    generated_preds_all.append(generated_preds)
                    expert_preds = self.__call__(expert_batch)
                    expert_preds_all.append(expert_preds)

                # Calculate loss
                expert_preds_all = th.concat(expert_preds_all)
                nominal_preds_all = th.concat(nominal_preds_all)
                generated_preds_all = th.concat(generated_preds_all)
                if self.train_gail_lambda:
                    nominal_preds_prod = nominal_preds_all.prod(dim=0)
                    nominal_loss = self.criterion(nominal_preds_prod,
                                                  th.zeros(*nominal_preds_prod.size()).to(self.device))
                    generated_preds_prod = generated_preds_all.prod(dim=0)
                    generated_loss = self.criterion(generated_preds_prod,
                                                  th.zeros(*generated_preds_prod.size()).to(self.device))
                    expert_preds_prod = expert_preds_all.prod(dim=0)
                    expert_loss = self.criterion(expert_preds_prod, th.ones(*expert_preds_prod.size()).to(self.device))
                    regularizer_loss = th.tensor(0)
                    loss = nominal_loss + self.generated_weight * generated_loss + expert_loss
                    # expert_loss = th.sum(th.log(expert_preds_all + self.eps))
                    # nominal_loss = th.sum(th.log(nominal_preds_all + self.eps))
                    # regularizer_loss = th.tensor(0)
                    # loss = (-expert_loss + nominal_loss) + regularizer_loss
                else:
                    expert_loss = th.sum(th.log(expert_preds_all + self.eps))
                    nominal_loss = th.sum(is_batch * th.log(nominal_preds_all + self.eps))
                    generated_loss = th.sum(generated_is_batch * th.log(generated_preds_all + self.eps))
                    regularizer_loss = self.regularizer_coeff * (th.mean(1 - expert_preds_all)
                                                                 + th.mean(1 - nominal_preds_all)
                                                                 + self.generated_weight * th.mean(1 - generated_preds_all))
                    loss = (-expert_loss + nominal_loss + self.generated_weight * generated_loss) + regularizer_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        bw_metrics = {"backward/cn_loss": loss.item(),
                      "backward/expert_loss": expert_loss.item(),
                      "backward/unweighted_nominal_loss": th.mean(th.log(nominal_preds + self.eps)).item(),
                      "backward/nominal_loss": nominal_loss.item(),
                      "backward/regularizer_loss": regularizer_loss.item(),
                      "backward/is_mean": th.mean(is_weights).detach().item(),
                      "backward/is_max": th.max(is_weights).detach().item(),
                      "backward/is_min": th.min(is_weights).detach().item(),
                      "backward/nominal_preds_max": th.max(nominal_preds).item(),
                      "backward/nominal_preds_min": th.min(nominal_preds).item(),
                      "backward/nominal_preds_mean": th.mean(nominal_preds).item(),
                      "backward/expert_preds_max": th.max(expert_preds).item(),
                      "backward/expert_preds_min": th.min(expert_preds).item(),
                      "backward/expert_preds_mean": th.mean(expert_preds).item(), }
        if self.importance_sampling:
            stop_metrics = {"backward/kl_old_new": kl_old_new.item(),
                            "backward/kl_new_old": kl_new_old.item(),
                            "backward/early_stop_itr": early_stop_itr}
            bw_metrics.update(stop_metrics)

        return bw_metrics

    def train_nn(
            self,
            iterations: np.ndarray,
            nominal_obs: np.ndarray,
            nominal_acs: np.ndarray,
            episode_lengths: np.ndarray,
            obs_mean: Optional[np.ndarray] = None,
            obs_var: Optional[np.ndarray] = None,
            current_progress_remaining: float = 1,
    ) -> Dict[str, Any]:

        # Update learning rate
        self._update_learning_rate(current_progress_remaining)

        # Update normalization stats
        self.current_obs_mean = obs_mean
        self.current_obs_var = obs_var

        # Prepare data
        nominal_data = self.prepare_data(nominal_obs, nominal_acs)
        expert_data = self.prepare_data(self.expert_obs, self.expert_acs)

        # Save current network predictions if using importance sampling
        if self.importance_sampling:
            with th.no_grad():
                start_preds = self.forward(nominal_data).detach()

        early_stop_itr = iterations
        loss = th.tensor(np.inf)
        for itr in tqdm(range(iterations)):
            # Compute IS weights
            if self.importance_sampling:
                with th.no_grad():
                    current_preds = self.forward(nominal_data).detach()
                is_weights, kl_old_new, kl_new_old = self.compute_is_weights(start_preds.clone(), current_preds.clone(),
                                                                             episode_lengths)
                # Break if kl is very large
                if ((self.target_kl_old_new != -1 and kl_old_new > self.target_kl_old_new) or
                        (self.target_kl_new_old != -1 and kl_new_old > self.target_kl_new_old)):
                    early_stop_itr = itr
                    break
            else:
                is_weights = th.ones(nominal_data.shape[0]).to(self.device)

            # Do a complete pass on data
            for nom_batch_indices, exp_batch_indices in self.get(nominal_data.shape[0], expert_data.shape[0]):
                # Get batch data
                nominal_batch = nominal_data[nom_batch_indices]
                expert_batch = expert_data[exp_batch_indices]
                is_batch = is_weights[nom_batch_indices][..., None]

                # Make predictions
                nominal_preds = self.__call__(nominal_batch)
                expert_preds = self.__call__(expert_batch)

                # Calculate loss
                if self.train_gail_lambda:
                    nominal_loss = self.criterion(nominal_preds, th.zeros(*nominal_preds.size()).to(self.device))
                    expert_loss = self.criterion(expert_preds, th.ones(*expert_preds.size()).to(self.device))
                    regularizer_loss = th.tensor(0)
                    loss = nominal_loss + expert_loss
                else:
                    expert_loss = th.mean(th.log(expert_preds + self.eps))
                    nominal_loss = th.mean(is_batch * th.log(nominal_preds + self.eps))
                    regularizer_loss = self.regularizer_coeff * (th.mean(1 - expert_preds) + th.mean(1 - nominal_preds))
                    loss = (-expert_loss + nominal_loss) + regularizer_loss

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        bw_metrics = {"backward/cn_loss": loss.item(),
                      "backward/expert_loss": expert_loss.item(),
                      "backward/unweighted_nominal_loss": th.mean(th.log(nominal_preds + self.eps)).item(),
                      "backward/nominal_loss": nominal_loss.item(),
                      "backward/regularizer_loss": regularizer_loss.item(),
                      "backward/is_mean": th.mean(is_weights).detach().item(),
                      "backward/is_max": th.max(is_weights).detach().item(),
                      "backward/is_min": th.min(is_weights).detach().item(),
                      "backward/nominal_preds_max": th.max(nominal_preds).item(),
                      "backward/nominal_preds_min": th.min(nominal_preds).item(),
                      "backward/nominal_preds_mean": th.mean(nominal_preds).item(),
                      "backward/expert_preds_max": th.max(expert_preds).item(),
                      "backward/expert_preds_min": th.min(expert_preds).item(),
                      "backward/expert_preds_mean": th.mean(expert_preds).item(), }
        if self.importance_sampling:
            stop_metrics = {"backward/kl_old_new": kl_old_new.item(),
                            "backward/kl_new_old": kl_new_old.item(),
                            "backward/early_stop_itr": early_stop_itr}
            bw_metrics.update(stop_metrics)

        return bw_metrics

    def get_three(self, nom_size: int, generated_size: int, exp_size: int) -> np.ndarray:
        if self.batch_size is None:
            yield np.arange(nom_size), np.arange(generated_size), np.arange(exp_size)
        else:
            size = min(nom_size, generated_size, exp_size)
            expert_indices = np.random.permutation(exp_size)
            nom_indices = np.random.permutation(nom_size)
            generated_indices = np.random.permutation(generated_size)
            start_idx = 0
            while start_idx < size:
                batch_expert_indices = expert_indices[start_idx:start_idx + self.batch_size]
                batch_nom_indices = nom_indices[start_idx:start_idx + self.batch_size]
                batch_generated_indices = generated_indices[start_idx:start_idx + self.batch_size]
                yield batch_nom_indices, batch_generated_indices, batch_expert_indices
                start_idx += self.batch_size

    def select_action(self, state):
        with th.no_grad():
            recon_state = self.prepare_obs_data(state)
            P_F_s, P_B_s = self.gflow_forward(recon_state)
            cat = Categorical(logits=P_F_s)
            action = cat.sample()
        return action.unsqueeze(0).cpu().data.numpy()

    def compute_is_weights(self, preds_old: th.Tensor, preds_new: th.Tensor, episode_lengths: np.ndarray) -> th.tensor:
        with th.no_grad():
            n_episodes = len(episode_lengths)
            cumulative = [0] + list(accumulate(episode_lengths))

            ratio = (preds_new + self.eps) / (preds_old + self.eps)
            prod = [th.prod(ratio[cumulative[j]:cumulative[j + 1]]) for j in range(n_episodes)]
            prod = th.tensor(prod)
            normed = n_episodes * prod / (th.sum(prod) + self.eps)

            if self.per_step_importance_sampling:
                is_weights = th.tensor(ratio / th.mean(ratio))
            else:
                is_weights = []
                for length, weight in zip(episode_lengths, normed):
                    is_weights += [weight] * length
                is_weights = th.tensor(is_weights)

            # Compute KL(old, current)
            kl_old_new = th.mean(-th.log(prod + self.eps))
            # Compute KL(current, old)
            prod_mean = th.mean(prod)
            kl_new_old = th.mean((prod - prod_mean) * th.log(prod + self.eps) / (prod_mean + self.eps))

        return is_weights.to(self.device), kl_old_new, kl_new_old

    def prepare_data(
            self,
            obs: np.ndarray,
            acs: np.ndarray,
    ) -> th.tensor:

        if self.recon_obs:
            obs = idx2vector(obs, height=self.env_configs['map_height'], width=self.env_configs['map_width'])
        else:
            obs = copy.copy(obs)

        obs = self.normalize_obs(obs, self.current_obs_mean, self.current_obs_var, self.clip_obs)
        acs = self.reshape_actions(acs)
        acs = self.clip_actions(acs, self.action_low, self.action_high)
        concat = self.select_appropriate_dims(np.concatenate([obs, acs], axis=-1))
        del obs, acs

        return th.tensor(concat, dtype=th.float32).to(self.device)

    def prepare_obs_data(
            self,
            obs: np.ndarray,
    ) -> th.tensor:

        if self.recon_obs:
            obs = idx2vector(obs, height=self.env_configs['map_height'], width=self.env_configs['map_width'])
        else:
            obs = copy.copy(obs)

        obs = self.normalize_obs(obs, self.current_obs_mean, self.current_obs_var, self.clip_obs)

        return th.tensor(obs, dtype=th.float32).to(self.device)

    def select_appropriate_dims(self, x: Union[np.ndarray, th.tensor]) -> Union[np.ndarray, th.tensor]:
        return x[..., self.select_dim]

    def normalize_obs(self, obs: np.ndarray, mean: Optional[float] = None, var: Optional[float] = None,
                      clip_obs: Optional[float] = None) -> np.ndarray:
        if mean is not None and var is not None:
            mean, var = mean[None], var[None]
            obs = (obs - mean) / np.sqrt(var + self.eps)
        if clip_obs is not None:
            obs = np.clip(obs, -self.clip_obs, self.clip_obs)

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

    def get(self, nom_size: int, exp_size: int) -> np.ndarray:
        if self.batch_size is None:
            yield np.arange(nom_size), np.arange(exp_size)
        else:
            size = min(nom_size, exp_size)
            expert_indices = np.random.permutation(exp_size)
            nom_indices = np.random.permutation(nom_size)
            start_idx = 0
            while start_idx < size:
                batch_expert_indices = expert_indices[start_idx:start_idx + self.batch_size]
                batch_nom_indices = nom_indices[start_idx:start_idx + self.batch_size]
                yield batch_nom_indices, batch_expert_indices
                start_idx += self.batch_size

            # size = min(nom_size, exp_size)
            # indices = np.random.permutation(size)
            # batch_size = self.batch_size
            # # Return everything, don't create minibatches
            # if batch_size is None:
            #     batch_size = size
            #
            # start_idx = 0
            # while start_idx < size:
            #     batch_indices = indices[start_idx:start_idx + batch_size]
            #     yield batch_indices, batch_indices
            #     start_idx += batch_size

    def _update_learning_rate(self, current_progress_remaining) -> None:
        self.current_progress_remaining = current_progress_remaining
        update_learning_rate(self.optimizer, self.lr_schedule(current_progress_remaining))
        print("Learning rate is {0}.".format(self.lr_schedule(current_progress_remaining)),
              file=self.log_file,
              flush=True)

    def save(self, save_path):
        state_dict = dict(
            cn_network=self.network.state_dict(),
            cn_optimizer=self.optimizer.state_dict(),
            obs_dim=self.obs_dim,
            acs_dim=self.acs_dim,
            is_discrete=self.is_discrete,
            obs_select_dim=self.obs_select_dim,
            acs_select_dim=self.acs_select_dim,
            clip_obs=self.clip_obs,
            obs_mean=self.current_obs_mean,
            obs_var=self.current_obs_var,
            action_low=self.action_low,
            action_high=self.action_high,
            device=self.device,
            hidden_sizes=self.hidden_sizes
        )
        th.save(state_dict, save_path)

    def _load(self, load_path):
        state_dict = th.load(load_path)
        if "cn_network" in state_dict:
            self.network.load_state_dict(dict["cn_network"])
        if "cn_optimizer" in state_dict and self.optimizer is not None:
            self.optimizer.load_state_dict(dict["cn_optimizer"])

    # Provide basic functionality to load this class
    @classmethod
    def load(
            cls,
            load_path: str,
            obs_dim: Optional[int] = None,
            acs_dim: Optional[int] = None,
            is_discrete: bool = None,
            obs_select_dim: Optional[Tuple[int, ...]] = None,
            acs_select_dim: Optional[Tuple[int, ...]] = None,
            clip_obs: Optional[float] = None,
            obs_mean: Optional[np.ndarray] = None,
            obs_var: Optional[np.ndarray] = None,
            action_low: Optional[float] = None,
            action_high: Optional[float] = None,
            device: str = "auto"
    ):

        state_dict = th.load(load_path)
        # If value isn't specified, then get from state_dict
        if obs_dim is None:
            obs_dim = state_dict["obs_dim"]
        if acs_dim is None:
            acs_dim = state_dict["acs_dim"]
        if is_discrete is None:
            is_discrete = state_dict["is_discrete"]
        if obs_select_dim is None:
            obs_select_dim = state_dict["obs_select_dim"]
        if acs_select_dim is None:
            acs_select_dim = state_dict["acs_select_dim"]
        if clip_obs is None:
            clip_obs = state_dict["clip_obs"]
        if obs_mean is None:
            obs_mean = state_dict["obs_mean"]
        if obs_var is None:
            obs_var = state_dict["obs_var"]
        if action_low is None:
            action_low = state_dict["action_low"]
        if action_high is None:
            action_high = state_dict["action_high"]
        if device is None:
            device = state_dict["device"]

        # Create network
        hidden_sizes = state_dict["hidden_sizes"]
        constraint_net = cls(
            obs_dim=obs_dim,
            acs_dim=acs_dim,
            hidden_sizes=hidden_sizes,
            batch_size=None,
            lr_schedule=None,
            expert_obs=None,
            expert_acs=None,
            optimizer_class=None,
            is_discrete=is_discrete,
            obs_select_dim=obs_select_dim,
            acs_select_dim=acs_select_dim,
            clip_obs=clip_obs,
            initial_obs_mean=obs_mean,
            initial_obs_var=obs_var,
            action_low=action_low,
            action_high=action_high,
            device=device
        )
        constraint_net.network.load_state_dict(state_dict["cn_network"])

        return constraint_net
