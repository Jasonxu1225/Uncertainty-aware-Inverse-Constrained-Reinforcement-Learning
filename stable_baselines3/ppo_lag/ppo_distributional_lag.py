from typing import Any, Callable, Dict, Optional, Type, Union

import numpy as np
import torch as th
import torch
import gym
from gym import spaces
from torch.nn import functional as F
import random

from stable_baselines3.common import logger
from stable_baselines3.common.dual_variable import DualVariable, PIDLagrangian
from stable_baselines3.common.on_policy_algorithm import \
    OnPolicyWithCostAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

# using distributional cost directly in L-CLIP
class PPODistributionalLagrangian(OnPolicyWithCostAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) augmented with a Lagrangian (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        algo_type: str = 'lagrangian',         # lagrangian or pidlagrangian
        learning_rate: Union[float, Callable] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 64,
        n_epochs: int = 10,
        reward_gamma: float = 0.99,
        reward_gae_lambda: float = 0.95,
        cost_gamma: float = 0.99,
        cost_gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_reward_vf: Optional[float] = None,
        clip_range_cost_vf: Optional[float] = None,
        ent_coef: float = 0.0,
        reward_vf_coef: float = 0.5,
        cost_vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        penalty_initial_value: float = 1,
        penalty_learning_rate: float = 0.01,
        penalty_min_value: Optional[float] = None,
        update_penalty_after: int = 1,
        budget: float = 0.,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        pid_kwargs: Optional[Dict[str, Any]] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        recon_obs: bool = False,
        env_configs: dict = False,
        input_action: bool = True
    ):

        super(PPODistributionalLagrangian, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            reward_gamma=reward_gamma,
            reward_gae_lambda=reward_gae_lambda,
            cost_gamma=cost_gamma,
            cost_gae_lambda=cost_gae_lambda,
            ent_coef=ent_coef,
            reward_vf_coef=reward_vf_coef,
            cost_vf_coef=cost_vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            recon_obs=recon_obs,
            env_configs=env_configs,
            input_action=input_action
        )
        self.hl_kappa_k = 1.0 #hubel loss kappa k
        self.cost_gamma = cost_gamma
        self.algo_type = algo_type
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_reward_vf = clip_range_reward_vf
        self.clip_range_cost_vf = clip_range_cost_vf
        self.target_kl = target_kl

        self.penalty_initial_value = penalty_initial_value
        self.penalty_learning_rate = penalty_learning_rate
        self.penalty_min_value = penalty_min_value
        self.update_penalty_after = update_penalty_after
        self.budget = budget
        self.pid_kwargs = pid_kwargs

        if _init_setup_model:
            self._setup_model()

    def idx2vector(self, indices, height, width):
        vector_all = []
        if isinstance(indices, torch.Tensor):
            for idx in indices:
                map = np.zeros(shape=[height, width])
                x, y = int(torch.round(idx[0])), int(torch.round(idx[1]))
                # if x - idx[0] != 0:
                #     print('debug')
                map[x, y] = 1  # + idx[0] - x + idx[1] - y
                vector_all.append(map.flatten())
            return torch.Tensor(np.array(vector_all))
        else:
            for idx in indices:
                map = np.zeros(shape=[height, width])
                x, y = int(round(idx[0], 0)), int(round(idx[1], 0))
                # if x - idx[0] != 0:
                #     print('debug')
                map[x, y] = 1  # + idx[0] - x + idx[1] - y
                vector_all.append(map.flatten())
            return np.asarray(vector_all)

    def _setup_model(self) -> None:
        super(PPODistributionalLagrangian, self)._setup_model()

        if self.algo_type == 'lagrangian':
            self.dual = DualVariable(self.budget, self.penalty_learning_rate, self.penalty_initial_value, self.penalty_min_value)
        elif self.algo_type == 'pidlagrangian':
            self.dual = PIDLagrangian(alpha=self.pid_kwargs['alpha'],
                                      penalty_init=self.pid_kwargs['penalty_init'],
                                      Kp=self.pid_kwargs['Kp'],
                                      Ki=self.pid_kwargs['Ki'],
                                      Kd=self.pid_kwargs['Kd'],
                                      pid_delay=self.pid_kwargs['pid_delay'],
                                      delta_p_ema_alpha=self.pid_kwargs['delta_p_ema_alpha'],
                                      delta_d_ema_alpha=self.pid_kwargs['delta_d_ema_alpha'])
        else:
            raise ValueError("Unrecognized value for argument 'algo_type' in PPODistributionalLagrangian")
        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_reward_vf is not None:
            if isinstance(self.clip_range_reward_vf, (float, int)):
                assert self.clip_range_reward_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_reward_vf = get_schedule_fn(self.clip_range_reward_vf)

        if self.clip_range_cost_vf is not None:
            if isinstance(self.clip_range_cost_vf, (float, int)):
                assert self.clip_range_cost_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_cost_vf = get_schedule_fn(self.clip_range_cost_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer.
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # self._update_learning_rate(self.policy.optimizer_QN)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value functions
        if self.clip_range_reward_vf is not None:
            clip_range_reward_vf = self.clip_range_reward_vf(self._current_progress_remaining)
        if self.clip_range_cost_vf is not None:
            clip_range_cost_vf = self.clip_range_cost_vf(self._current_progress_remaining)

        entropy_losses, all_kl_divs = [], []
        pg_losses, reward_value_losses, cost_value_losses = [], [], []
        DQ_losses = []
        clip_fractions = []

        # Train for gradient_steps epochs
        early_stop_epoch = self.n_epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions  #[64,6]
                new_actions = rollout_data.new_actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()
                    new_actions = rollout_data.new_actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                # if that line is commented (as in SAC)
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)
                # 用了由mlp组成的value net和cost net来预测reward和cost
                # cost values: VaR(75%)
                reward_values, cost_values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)

                reward_values = reward_values.flatten()
                cost_values = cost_values.flatten()


                # qutile regression update
                # self.policy.optimizer_QN.zero_grad()

                # compute targets cost value
                with torch.no_grad():
                    #_latent_pi, _latent_vf, _latent_cvf, _latent_sde = self.policy._get_latent(rollout_data.new_observations)
                    new_obs = rollout_data.new_observations
                    new_features = self.policy.extract_features(new_obs)

                if self.recon_obs:
                    new_features = self.idx2vector(new_features, height=self.env_configs['map_height'],
                                              width=self.env_configs['map_width']).to(self.device)
                if len(new_actions.shape) != len(new_features.shape):
                    new_actions = new_actions.view(new_features.shape[0], -1)

                if self.policy.method == 'QRDQN' or self.policy.method=='SplineDQN' or self.policy.method=='NCQR':
                    with torch.no_grad():
                        # if self.input_action:
                        #     distributional_cost_values_targets_next = self.policy.cost_value_net_target(
                        #             th.cat([new_features, new_actions], dim=1))
                        # else:
                        #     distributional_cost_values_targets_next = self.policy.cost_value_net_target(new_features)

                        # distributional_cost_values_targets_next = distributional_cost_values_targets_next.unsqueeze(
                        #         -1).transpose(1,2)

                        distributional_cost_values_targets_next = self.policy.cost_value_net_target(new_features, new_actions)
                    # TODO
                    # costs = rollout_data.costs.view(-1,1)
                    # dones = rollout_data.dones.view(-1,1)
                    costs = rollout_data.costs.unsqueeze(1)
                    dones = rollout_data.dones.unsqueeze(1)
                    # (observations, actions, costs, dones, new_obs, new_actions)
                    distributional_cost_values_targets = costs + \
                        (self.cost_gamma * distributional_cost_values_targets_next.to(self.device) * (1 - dones))

                    #compute local cost value
                    #_latent_pi, _latent_vf, _latent_cvf, _latent_sde = self.policy._get_latent(rollout_data.new_observations)
                    with torch.no_grad():
                        features = self.policy.extract_features(rollout_data.observations)
                    if self.recon_obs:
                        features = self.idx2vector(features, height=self.env_configs['map_height'],
                                              width=self.env_configs['map_width']).to(self.device)
                    if len(actions.shape) != len(features.shape):
                        actions = actions.view(features.shape[0], -1)
                    # if self.input_action:
                    #     distributional_cost_values_expected = self.policy.cost_value_net_local(th.cat([features, actions], dim=1))
                    # else:
                    #     distributional_cost_values_expected = self.policy.cost_value_net_local(features)
                    distributional_cost_values_expected = self.policy.cost_value_net_local(features, actions)

                    # print(distributional_cost_values_expected[0])
                    # print(th.mean(distributional_cost_values_expected[0]))
                    # print(th.mean(rollout_data.cost_returns))
                    # print('------------------------------------------------')

                    # distributional_cost_values_expected = distributional_cost_values_expected.unsqueeze(-1)

                    T_theta_tile = distributional_cost_values_targets.view(-1, self.policy.N, 1).expand(-1, self.policy.N, self.policy.N).to(self.device)  # target
                    theta_a_tile = distributional_cost_values_expected.view(-1, 1, self.policy.N).expand(-1, self.policy.N, self.policy.N).to(self.device) # local

                    quantile_tau = torch.arange(0.5 * (1 / self.policy.N), 1, 1 / self.policy.N).view(1, self.policy.N).to(self.device)

                    # compute loss
                    # td_error = distributional_cost_values_targets - distributional_cost_values_expected
                    # huber_l = torch.where(td_error.abs() <= self.hl_kappa_k, 0.5 * td_error.pow(2), self.hl_kappa_k * (td_error.abs() - 0.5 * self.hl_kappa_k))
                    # quantil_l = abs(self.policy.quantile_tau.to(self.device) - (td_error.detach() < 0).float()) * huber_l.to(self.device) / 1.0

                    error_loss = T_theta_tile - theta_a_tile
                    huber_loss = F.smooth_l1_loss(theta_a_tile, T_theta_tile.detach(), reduction='none')
                    value_loss = (quantile_tau - (error_loss < 0).float()).abs() * huber_loss
                    DQ_loss = value_loss.mean(dim=2).sum(dim=1).mean()

                    # DQ_loss = quantil_l.sum(dim=1).mean(dim=1)  # keepdim=True if per weights get multipl
                    # DQ_loss = DQ_loss.mean()

                elif self.policy.method == 'IQN':
                    with torch.no_grad():
                        # if self.input_action:
                        #     distributional_cost_values_targets_next, _ = self.policy.cost_value_net_target(
                        #         th.cat([new_features, new_actions], dim=1))
                        # else:
                        #     distributional_cost_values_targets_next, _ = self.policy.cost_value_net_target(new_features)

                        # distributional_cost_values_targets_next = distributional_cost_values_targets_next.unsqueeze(
                        #     -1).transpose(1, 2)
                        distributional_cost_values_targets_next, taus_next = self.policy.cost_value_net_target(new_features, new_actions)

                    # costs = rollout_data.costs.view(-1, 1)
                    # dones = rollout_data.dones.view(-1, 1)
                    costs = rollout_data.costs.unsqueeze(1)
                    dones = rollout_data.dones.unsqueeze(1)

                    distributional_cost_values_targets = costs + \
                                                         (self.cost_gamma * distributional_cost_values_targets_next.to(
                                                             self.device) * (1 - dones))

                    # compute local cost value
                    # _latent_pi, _latent_vf, _latent_cvf, _latent_sde = self.policy._get_latent(rollout_data.new_observations)
                    with torch.no_grad():
                        features = self.policy.extract_features(rollout_data.observations)
                    if self.recon_obs:
                        features = self.idx2vector(features, height=self.env_configs['map_height'],
                                              width=self.env_configs['map_width']).to(self.device)
                    if len(actions.shape) != len(features.shape):
                        actions = actions.view(features.shape[0], -1)
                    # if self.input_action:
                    #     distributional_cost_values_expected, taus = self.policy.cost_value_net_local(
                    #         th.cat([features, actions], dim=1))
                    # else:
                    #     distributional_cost_values_expected, taus = self.policy.cost_value_net_local(features)
                    #
                    # distributional_cost_values_expected = distributional_cost_values_expected.unsqueeze(-1)
                    distributional_cost_values_expected, taus = self.policy.cost_value_net_local(features, actions)

                    # compute loss
                    # td_error = distributional_cost_values_targets - distributional_cost_values_expected
                    # huber_l = torch.where(td_error.abs() <= self.hl_kappa_k, 0.5 * td_error.pow(2),
                    #                       self.hl_kappa_k * (td_error.abs() - 0.5 * self.hl_kappa_k))
                    # quantil_l = abs(
                    #     taus.to(self.device) - (td_error.detach() < 0).float()) * huber_l.to(
                    #     self.device) / 1.0
                    #
                    # DQ_loss = quantil_l.sum(dim=1).mean(dim=1)  # keepdim=True if per weights get multipl
                    # DQ_loss = DQ_loss.mean()
                    T_theta_tile = distributional_cost_values_targets.view(-1, self.policy.N, 1).expand(-1, self.policy.N,
                                                                                   self.policy.N).to(self.device)  # target
                    theta_a_tile =  distributional_cost_values_expected.view(-1, 1, self.policy.N).expand(-1, self.policy.N,
                                                                                  self.policy.N).to(self.device)  # current
                    taus = taus.view(1, self.policy.N).to(self.device)

                    error_loss = T_theta_tile - theta_a_tile
                    huber_loss = F.smooth_l1_loss(theta_a_tile, T_theta_tile.detach(), reduction='none')
                    value_loss = (taus - (error_loss < 0).float()).abs() * huber_loss
                    DQ_loss = value_loss.mean(dim=2).sum(dim=1).mean()

                #cost_value_losses.append(DQ_loss.item())

                # DQ_loss.backward()
                # Clip grad norm
                # th.nn.utils.clip_grad_norm_(self.policy.cost_value_net_local.parameters(), self.max_grad_norm)
                # self.policy.optimizer_QN.step()

                # soft update target net with update_tau
                for target_param, local_param in zip(self.policy.cost_value_net_target.parameters(), self.policy.cost_value_net_local.parameters()):
                    target_param.data.copy_(
                        self.policy.tau_update * local_param.data + (1.0 - self.policy.tau_update) * target_param.data)

                # Normalize reward advantages
                reward_advantages = rollout_data.reward_advantages - rollout_data.reward_advantages.mean()
                reward_advantages /= (rollout_data.reward_advantages.std() + 1e-8) #[64]


                # Center but NOT rescale cost advantages
                #cost_advantages = rollout_data.cost_advantages - rollout_data.cost_advantages.mean()
                #cost_advantages /= (rollout_data.cost_advantages.std() + 1e-8)
                cost_values_toLclip = rollout_data.old_cost_values.to(self.device)
                cost_values_toLclip /= (rollout_data.old_cost_values.std() +1e-8)

                # Ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # Clipped surrogate loss
                # L-CLIP
                policy_loss_1 = reward_advantages * ratio
                policy_loss_2 = reward_advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Add cost to loss
                current_penalty = self.dual.nu().item()
                policy_loss += current_penalty * th.mean(cost_values_toLclip * ratio)
                policy_loss /= (1 + current_penalty)

                # Logging
                pg_losses.append(policy_loss.item()) #float list


                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction) #float list

                # default: None
                if self.clip_range_reward_vf is None:
                    # No clipping
                    reward_values_pred = reward_values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    reward_values_pred = rollout_data.old_reward_values + th.clamp(
                        reward_values - rollout_data.old_reward_values, -clip_range_reward_vf, clip_range_reward_vf
                    )
                # default: None
                if self.clip_range_cost_vf is None:
                    # No clipping
                    cost_values_pred = cost_values
                else:
                    # Clip the difference between old and new cost
                    # NOTE: this depends on the cost scaling
                    cost_values_pred = rollout_data.old_cost_values + th.clamp(
                        cost_values - rollout_data.old_cost_values, -clip_range_cost_vf, clip_range_cost_vf
                    )

                # Value loss using the TD(gae_lambda) target
                reward_value_loss = F.mse_loss(rollout_data.reward_returns, reward_values_pred)
                cost_value_loss = F.mse_loss(rollout_data.cost_returns, cost_values_pred.to(self.device))
                reward_value_losses.append(reward_value_loss.item())
                cost_value_losses.append(cost_value_loss.item())

                DQ_losses.append(DQ_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())


                #TODO
                loss = (policy_loss # -L-clip
                        + (self.ent_coef) * entropy_loss # -entropy_loss
                        + (self.reward_vf_coef) * reward_value_loss # reward_value_loss
                        + (self.cost_vf_coef) * DQ_loss)

                # loss = (policy_loss # -L-clip
                #         + self.ent_coef * entropy_loss # -entropy_loss
                #         + reward_value_loss)
                        #+ self.reward_vf_coef * reward_value_loss) # reward_value_loss
                        #+ self.cost_vf_coef * cost_value_loss)

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                approx_kl_divs.append(th.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy())

            all_kl_divs.append(np.mean(approx_kl_divs))

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                early_stop_epoch = epoch
                if self.verbose > 0:
                    print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                break

        self._n_updates += self.n_epochs

        # Update dual variable using original (unnormalized) cost
        # TODO: Experiment with discounted cost.
        average_cost = np.mean(self.rollout_buffer.orig_costs)
        total_cost = np.sum(self.rollout_buffer.orig_costs)
        if self.update_penalty_after is None or ((self._n_updates/self.n_epochs) % self.update_penalty_after == 0):
            self.dual.update_parameter(average_cost)

        mean_reward_advantages = np.mean(self.rollout_buffer.reward_advantages.flatten())
        mean_cost_advantages = np.mean(self.rollout_buffer.cost_advantages.flatten())

        explained_reward_var = explained_variance(self.rollout_buffer.reward_returns.flatten(), self.rollout_buffer.reward_values.flatten())
        explained_cost_var = explained_variance(self.rollout_buffer.cost_returns.flatten(), self.rollout_buffer.cost_values.flatten())

        # Logs
        logger.record("train/entropy_loss", np.mean(entropy_losses))
        logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        logger.record("train/reward_value_loss", np.mean(reward_value_losses))
        logger.record("train/cost_value_loss", np.mean(cost_value_losses))
        logger.record("train/DQ_loss", np.mean(DQ_losses))
        logger.record("train/approx_kl", np.mean(approx_kl_divs))
        logger.record("train/clip_fraction", np.mean(clip_fractions))
        logger.record("train/loss", loss.item())
        logger.record("train/mean_reward_advantages", mean_reward_advantages)
        logger.record("train/mean_cost_advantages", mean_cost_advantages)
        logger.record("train/reward_explained_variance", explained_reward_var)
        logger.record("train/cost_explained_variance", explained_cost_var)
        logger.record("train/nu", self.dual.nu().item())
        logger.record("train/nu_loss", self.dual.loss.item())
        logger.record("train/average_cost", average_cost)
        logger.record("train/total_cost", total_cost)
        logger.record("train/early_stop_epoch", early_stop_epoch)
        if hasattr(self.policy, "log_std"):
            logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/clip_range", clip_range)
        if self.clip_range_reward_vf is not None:
            logger.record("train/clip_range_reward_vf", clip_range_reward_vf)
        if self.clip_range_cost_vf is not None:
            logger.record("train/clip_range_cost_vf", clip_range_cost_vf)

    def learn(
        self,
        total_timesteps: int,
        cost_function: Union[str,Callable],
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "PPODistributionalLagrangian",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "PPODistributionalLagrangian":

        return super(PPODistributionalLagrangian, self).learn(
            total_timesteps=total_timesteps,
            cost_function=cost_function,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

# using distributional cost to calculate cost-advantage, then used in L-LICP
class PPODistributionalLagrangianCostAdv(OnPolicyWithCostAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) augmented with a Lagrangian (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        algo_type: str = 'lagrangian',         # lagrangian or pidlagrangian
        learning_rate: Union[float, Callable] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 64,
        n_epochs: int = 10,
        reward_gamma: float = 0.99,
        reward_gae_lambda: float = 0.95,
        cost_gamma: float = 0.99,
        cost_gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_reward_vf: Optional[float] = None,
        clip_range_cost_vf: Optional[float] = None,
        ent_coef: float = 0.0,
        reward_vf_coef: float = 0.5,
        cost_vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        penalty_initial_value: float = 1,
        penalty_learning_rate: float = 0.01,
        penalty_min_value: Optional[float] = None,
        update_penalty_after: int = 1,
        budget: float = 0.,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        pid_kwargs: Optional[Dict[str, Any]] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        recon_obs: bool = False,
        env_configs: dict = False,
        input_action:bool=True,
    ):

        super(PPODistributionalLagrangianCostAdv, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            reward_gamma=reward_gamma,
            reward_gae_lambda=reward_gae_lambda,
            cost_gamma=cost_gamma,
            cost_gae_lambda=cost_gae_lambda,
            ent_coef=ent_coef,
            reward_vf_coef=reward_vf_coef,
            cost_vf_coef=cost_vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            recon_obs=recon_obs,
            env_configs=env_configs,
            input_action=input_action,
        )
        self.hl_kappa_k = 1.0 #hubel loss kappa k
        self.cost_gamma = cost_gamma
        self.algo_type = algo_type
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_reward_vf = clip_range_reward_vf
        self.clip_range_cost_vf = clip_range_cost_vf
        self.target_kl = target_kl

        self.penalty_initial_value = penalty_initial_value
        self.penalty_learning_rate = penalty_learning_rate
        self.penalty_min_value = penalty_min_value
        self.update_penalty_after = update_penalty_after
        self.budget = budget
        self.pid_kwargs = pid_kwargs

        if _init_setup_model:
            self._setup_model()

    def idx2vector(self, indices, height, width):
        vector_all = []
        if isinstance(indices, torch.Tensor):
            for idx in indices:
                map = np.zeros(shape=[height, width])
                x, y = int(torch.round(idx[0])), int(torch.round(idx[1]))
                # if x - idx[0] != 0:
                #     print('debug')
                map[x, y] = 1  # + idx[0] - x + idx[1] - y
                vector_all.append(map.flatten())
            return torch.Tensor(np.array(vector_all))
        else:
            for idx in indices:
                map = np.zeros(shape=[height, width])
                x, y = int(round(idx[0], 0)), int(round(idx[1], 0))
                # if x - idx[0] != 0:
                #     print('debug')
                map[x, y] = 1  # + idx[0] - x + idx[1] - y
                vector_all.append(map.flatten())
            return np.asarray(vector_all)

    def quantile_regression_loss(self, expected, target, N):

        T_theta_tile = target.view(-1, N, 1).expand(-1, N, N).to(self.device)
        theta_a_tile = expected.view(-1, 1, N).expand(-1, N, N).to(self.device)
        quantile_tau = torch.arange(0.5 * (1 / N), 1, 1 / N).view(1, N).to(self.device)
        error_loss = T_theta_tile - theta_a_tile
        huber_loss = F.smooth_l1_loss(theta_a_tile, T_theta_tile.detach(), reduction='none')
        value_loss = (quantile_tau - (error_loss < 0).float()).abs() * huber_loss
        DQ_loss = value_loss.mean(dim=2).sum(dim=1).mean()
        return DQ_loss

    def _setup_model(self) -> None:
        super(PPODistributionalLagrangianCostAdv, self)._setup_model()

        if self.algo_type == 'lagrangian':
            self.dual = DualVariable(self.budget, self.penalty_learning_rate, self.penalty_initial_value, self.penalty_min_value)
        elif self.algo_type == 'pidlagrangian':
            self.dual = PIDLagrangian(alpha=self.pid_kwargs['alpha'],
                                      penalty_init=self.pid_kwargs['penalty_init'],
                                      Kp=self.pid_kwargs['Kp'],
                                      Ki=self.pid_kwargs['Ki'],
                                      Kd=self.pid_kwargs['Kd'],
                                      pid_delay=self.pid_kwargs['pid_delay'],
                                      delta_p_ema_alpha=self.pid_kwargs['delta_p_ema_alpha'],
                                      delta_d_ema_alpha=self.pid_kwargs['delta_d_ema_alpha'])
        else:
            raise ValueError("Unrecognized value for argument 'algo_type' in PPODistributionalLagrangianCostAdv")
        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_reward_vf is not None:
            if isinstance(self.clip_range_reward_vf, (float, int)):
                assert self.clip_range_reward_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_reward_vf = get_schedule_fn(self.clip_range_reward_vf)

        if self.clip_range_cost_vf is not None:
            if isinstance(self.clip_range_cost_vf, (float, int)):
                assert self.clip_range_cost_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_cost_vf = get_schedule_fn(self.clip_range_cost_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer.
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # self._update_learning_rate(self.policy.optimizer_QN)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value functions
        if self.clip_range_reward_vf is not None:
            clip_range_reward_vf = self.clip_range_reward_vf(self._current_progress_remaining)
        if self.clip_range_cost_vf is not None:
            clip_range_cost_vf = self.clip_range_cost_vf(self._current_progress_remaining)

        entropy_losses, all_kl_divs = [], []
        pg_losses, reward_value_losses, cost_value_losses = [], [], []
        DQ_losses = []
        clip_fractions = []

        # Train for gradient_steps epochs
        early_stop_epoch = self.n_epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions  #[64,6]
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                # if that line is commented (as in SAC)
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)
                # 用了由mlp组成的value net和cost net来预测reward和cost
                # cost values: VaR(75%)
                reward_values, cost_values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)

                reward_values = reward_values.flatten()
                cost_values = cost_values.flatten()


                # qutile regression update
                # self.policy.optimizer_QN.zero_grad()

                # compute targets cost value
                with torch.no_grad():
                    #_latent_pi, _latent_vf, _latent_cvf, _latent_sde = self.policy._get_latent(rollout_data.new_observations)
                    new_obs = rollout_data.new_observations
                    new_features = self.policy.extract_features(new_obs)

                if self.recon_obs:
                    new_features = self.idx2vector(new_features, height=self.env_configs['map_height'],
                                              width=self.env_configs['map_width']).to(self.device)

                if self.policy.method == 'QRDQN' or self.policy.method=='SplineDQN' or self.policy.method=='NCQR':
                    with torch.no_grad():
                        # if self.input_action:
                        #     distributional_cost_values_targets_next = self.policy.cost_value_net_target(
                        #             th.cat([new_features, new_actions], dim=1))
                        # else:
                        #     distributional_cost_values_targets_next = self.policy.cost_value_net_target(new_features)

                        # distributional_cost_values_targets_next = distributional_cost_values_targets_next.unsqueeze(
                        #         -1).transpose(1,2)

                        distributional_cost_values_targets_next = self.policy.cost_value_net_local(new_features)
                    # TODO
                    # costs = rollout_data.costs.view(-1,1)
                    # dones = rollout_data.dones.view(-1,1)
                    costs = rollout_data.costs.unsqueeze(1)
                    dones = rollout_data.dones.unsqueeze(1)
                    # (observations, actions, costs, dones, new_obs, new_actions)
                    distributional_cost_values_targets = costs + \
                        (self.cost_gamma * distributional_cost_values_targets_next.to(self.device) * (1 - dones))

                    #compute local cost value
                    #_latent_pi, _latent_vf, _latent_cvf, _latent_sde = self.policy._get_latent(rollout_data.new_observations)
                    with torch.no_grad():
                        features = self.policy.extract_features(rollout_data.observations)
                    if self.recon_obs:
                        features = self.idx2vector(features, height=self.env_configs['map_height'],
                                              width=self.env_configs['map_width']).to(self.device)
                    if len(actions.shape) != len(features.shape):
                        actions = actions.view(features.shape[0], -1)
                    # if self.input_action:
                    #     distributional_cost_values_expected = self.policy.cost_value_net_local(th.cat([features, actions], dim=1))
                    # else:
                    #     distributional_cost_values_expected = self.policy.cost_value_net_local(features)
                    distributional_cost_values_expected = self.policy.cost_value_net_local(features)

                    # print(distributional_cost_values_expected[0])
                    # print(th.mean(distributional_cost_values_expected[0]))
                    # print(th.mean(rollout_data.cost_returns))
                    # print('------------------------------------------------')

                    # distributional_cost_values_expected = distributional_cost_values_expected.unsqueeze(-1)

                    # T_theta_tile = distributional_cost_values_targets.view(-1, self.policy.N, 1).expand(-1, self.policy.N, self.policy.N).to(self.device) # target
                    # theta_a_tile = distributional_cost_values_expected.view(-1, 1, self.policy.N).expand(-1, self.policy.N, self.policy.N).to(self.device)# local
                    #
                    # quantile_tau = torch.arange(0.5 * (1 / self.policy.N), 1, 1 / self.policy.N).view(1, self.policy.N).to(self.device)
                    #
                    # # compute loss
                    # # td_error = distributional_cost_values_targets - distributional_cost_values_expected
                    # # huber_l = torch.where(td_error.abs() <= self.hl_kappa_k, 0.5 * td_error.pow(2), self.hl_kappa_k * (td_error.abs() - 0.5 * self.hl_kappa_k))
                    # # quantil_l = abs(self.policy.quantile_tau.to(self.device) - (td_error.detach() < 0).float()) * huber_l.to(self.device) / 1.0
                    #
                    # error_loss = T_theta_tile - theta_a_tile
                    # huber_loss = F.smooth_l1_loss(theta_a_tile, T_theta_tile.detach(), reduction='none')
                    # value_loss = (quantile_tau - (error_loss < 0).float()).abs() * huber_loss
                    # DQ_loss = value_loss.mean(dim=2).sum(dim=1).mean()
                    # # quantile_tau = torch.arange(0.5 * (1 / self.policy.N), 1, 1 / self.policy.N).view(1,self.policy.N).to(self.device)
                    DQ_loss = self.quantile_regression_loss(distributional_cost_values_expected, distributional_cost_values_targets, self.policy.N)

                    # DQ_loss = quantil_l.sum(dim=1).mean(dim=1)  # keepdim=True if per weights get multipl
                    # DQ_loss = DQ_loss.mean()

                elif self.policy.method == 'IQN':
                    with torch.no_grad():
                        # if self.input_action:
                        #     distributional_cost_values_targets_next, _ = self.policy.cost_value_net_target(
                        #         th.cat([new_features, new_actions], dim=1))
                        # else:
                        #     distributional_cost_values_targets_next, _ = self.policy.cost_value_net_target(new_features)

                        # distributional_cost_values_targets_next = distributional_cost_values_targets_next.unsqueeze(
                        #     -1).transpose(1, 2)
                        distributional_cost_values_targets_next, taus_next = self.policy.cost_value_net_local(new_features)

                    # costs = rollout_data.costs.view(-1, 1)
                    # dones = rollout_data.dones.view(-1, 1)
                    costs = rollout_data.costs.unsqueeze(1)
                    dones = rollout_data.dones.unsqueeze(1)

                    distributional_cost_values_targets = costs + \
                                                         (self.cost_gamma * distributional_cost_values_targets_next.to(
                                                             self.device) * (1 - dones))

                    # compute local cost value
                    # _latent_pi, _latent_vf, _latent_cvf, _latent_sde = self.policy._get_latent(rollout_data.new_observations)
                    with torch.no_grad():
                        features = self.policy.extract_features(rollout_data.observations)
                    if self.recon_obs:
                        features = self.idx2vector(features, height=self.env_configs['map_height'],
                                              width=self.env_configs['map_width']).to(self.device)
                    if len(actions.shape) != len(features.shape):
                        actions = actions.view(features.shape[0], -1)
                    # if self.input_action:
                    #     distributional_cost_values_expected, taus = self.policy.cost_value_net_local(
                    #         th.cat([features, actions], dim=1))
                    # else:
                    #     distributional_cost_values_expected, taus = self.policy.cost_value_net_local(features)
                    #
                    # distributional_cost_values_expected = distributional_cost_values_expected.unsqueeze(-1)
                    distributional_cost_values_expected, taus = self.policy.cost_value_net_local(features)

                    # compute loss
                    # td_error = distributional_cost_values_targets - distributional_cost_values_expected
                    # huber_l = torch.where(td_error.abs() <= self.hl_kappa_k, 0.5 * td_error.pow(2),
                    #                       self.hl_kappa_k * (td_error.abs() - 0.5 * self.hl_kappa_k))
                    # quantil_l = abs(
                    #     taus.to(self.device) - (td_error.detach() < 0).float()) * huber_l.to(
                    #     self.device) / 1.0
                    #
                    # DQ_loss = quantil_l.sum(dim=1).mean(dim=1)  # keepdim=True if per weights get multipl
                    # DQ_loss = DQ_loss.mean()
                    # T_theta_tile = distributional_cost_values_targets.view(-1, self.policy.N, 1).expand(-1, self.policy.N,
                    #                                                                self.policy.N).to(self.device)  # target
                    # theta_a_tile =  distributional_cost_values_expected.view(-1, 1, self.policy.N).expand(-1, self.policy.N,
                    #                                                               self.policy.N).to(self.device)  # current
                    # taus = taus.view(1, self.policy.N).to(self.device)
                    #
                    # error_loss = T_theta_tile - theta_a_tile
                    # huber_loss = F.smooth_l1_loss(theta_a_tile, T_theta_tile.detach(), reduction='none')
                    # value_loss = (taus - (error_loss < 0).float()).abs() * huber_loss
                    # DQ_loss = value_loss.mean(dim=2).sum(dim=1).mean()
                    DQ_loss = self.quantile_regression_loss(distributional_cost_values_expected,
                                                            distributional_cost_values_targets, self.policy.N)

                # DQ_loss.backward()
                # Clip grad norm
                # th.nn.utils.clip_grad_norm_(self.policy.cost_value_net_local.parameters(), self.max_grad_norm)
                # self.policy.optimizer_QN.step()

                # soft update target net with update_tau
                # for target_param, local_param in zip(self.policy.cost_value_net_target.parameters(), self.policy.cost_value_net_local.parameters()):
                #     target_param.data.copy_(
                #         self.policy.tau_update * local_param.data + (1.0 - self.policy.tau_update) * target_param.data)

                # Normalize reward advantages
                reward_advantages = rollout_data.reward_advantages - rollout_data.reward_advantages.mean()
                reward_advantages /= (rollout_data.reward_advantages.std() + 1e-8) #[64]


                # Center but NOT rescale cost advantages
                cost_advantages = rollout_data.cost_advantages - rollout_data.cost_advantages.mean()
                #cost_advantages /= (rollout_data.cost_advantages.std() + 1e-8)


                # Ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # Clipped surrogate loss
                # L-CLIP
                policy_loss_1 = reward_advantages * ratio
                policy_loss_2 = reward_advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Add cost to loss
                current_penalty = self.dual.nu().item()
                policy_loss += current_penalty * th.mean(cost_advantages * ratio)
                policy_loss /= (1 + current_penalty)

                # Logging
                pg_losses.append(policy_loss.item()) #float list


                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction) #float list

                # default: None
                if self.clip_range_reward_vf is None:
                    # No clipping
                    reward_values_pred = reward_values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    reward_values_pred = rollout_data.old_reward_values + th.clamp(
                        reward_values - rollout_data.old_reward_values, -clip_range_reward_vf, clip_range_reward_vf
                    )
                # default: None
                if self.clip_range_cost_vf is None:
                    # No clipping
                    cost_values_pred = cost_values
                else:
                    # Clip the difference between old and new cost
                    # NOTE: this depends on the cost scaling
                    cost_values_pred = rollout_data.old_cost_values + th.clamp(
                        cost_values - rollout_data.old_cost_values, -clip_range_cost_vf, clip_range_cost_vf
                    )

                # Value loss using the TD(gae_lambda) target
                reward_value_loss = F.mse_loss(rollout_data.reward_returns, reward_values_pred)
                cost_value_loss = F.mse_loss(rollout_data.cost_returns, cost_values_pred.to(self.device))
                # print(th.mean(rollout_data.cost_returns))
                # print(th.mean(cost_values_pred))
                # print(th.mean(cost_value_loss))
                # print('---------------------------------------')


                reward_value_losses.append(reward_value_loss.item())
                cost_value_losses.append(cost_value_loss.item())
                DQ_losses.append(DQ_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                #TODO
                loss = (policy_loss # -L-clip
                        + (self.ent_coef) * entropy_loss # -entropy_loss
                        + (self.reward_vf_coef) * reward_value_loss # reward_value_loss
                        + (self.cost_vf_coef) * DQ_loss)

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                approx_kl_divs.append(th.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy())

                #self.policy.soft_update(self.policy.cost_value_net_target, self.policy.cost_value_net_local, self.policy.tau_update)

            all_kl_divs.append(np.mean(approx_kl_divs))

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                early_stop_epoch = epoch
                if self.verbose > 0:
                    print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                break

        self._n_updates += self.n_epochs

        # Update dual variable using original (unnormalized) cost
        # TODO: Experiment with discounted cost.
        average_cost = np.mean(self.rollout_buffer.orig_costs)
        total_cost = np.sum(self.rollout_buffer.orig_costs)
        if self.update_penalty_after is None or ((self._n_updates/self.n_epochs) % self.update_penalty_after == 0):
            self.dual.update_parameter(average_cost)

        mean_reward_advantages = np.mean(self.rollout_buffer.reward_advantages.flatten())
        mean_cost_advantages = np.mean(self.rollout_buffer.cost_advantages.flatten())

        explained_reward_var = explained_variance(self.rollout_buffer.reward_returns.flatten(), self.rollout_buffer.reward_values.flatten())
        explained_cost_var = explained_variance(self.rollout_buffer.cost_returns.flatten(), self.rollout_buffer.cost_values.flatten())

        # Logs
        logger.record("train/entropy_loss", np.mean(entropy_losses))
        logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        logger.record("train/reward_value_loss", np.mean(reward_value_losses))
        logger.record("train/cost_value_loss", np.mean(cost_value_losses))
        logger.record("train/DQ_loss", np.mean(DQ_losses))
        logger.record("train/approx_kl", np.mean(approx_kl_divs))
        logger.record("train/clip_fraction", np.mean(clip_fractions))
        logger.record("train/loss", loss.item())
        logger.record("train/mean_reward_advantages", mean_reward_advantages)
        logger.record("train/mean_cost_advantages", mean_cost_advantages)
        logger.record("train/reward_explained_variance", explained_reward_var)
        logger.record("train/cost_explained_variance", explained_cost_var)
        logger.record("train/nu", self.dual.nu().item())
        logger.record("train/nu_loss", self.dual.loss.item())
        logger.record("train/average_cost", average_cost)
        logger.record("train/total_cost", total_cost)
        logger.record("train/early_stop_epoch", early_stop_epoch)
        if hasattr(self.policy, "log_std"):
            logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/clip_range", clip_range)
        if self.clip_range_reward_vf is not None:
            logger.record("train/clip_range_reward_vf", clip_range_reward_vf)
        if self.clip_range_cost_vf is not None:
            logger.record("train/clip_range_cost_vf", clip_range_cost_vf)

    def learn(
        self,
        total_timesteps: int,
        cost_function: Union[str,Callable],
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "PPODistributionalLagrangianCostAdv",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "PPODistributionalLagrangianCostAdv":

        return super(PPODistributionalLagrangianCostAdv, self).learn(
            total_timesteps=total_timesteps,
            cost_function=cost_function,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )
