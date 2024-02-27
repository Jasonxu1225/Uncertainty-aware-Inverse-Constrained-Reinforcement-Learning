import copy
import os
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import random
import numpy as np
import torch
from tqdm import tqdm
import torch.optim as optim
from torch.nn import functional as F
import torch.nn as nn
import torch as th

from common.cns_visualization import traj_visualization_2d
from stable_baselines3.common.dual_variable import DualVariable
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common import logger
from stable_baselines3.common.vec_env import VecNormalizeWithCost
from stable_baselines3.common.preprocessing import get_action_dim, preprocess_obs


class DistributionalPolicyIterationLagrange(ABC):

    def __init__(self,
                 env: Union[GymEnv, str],
                 max_iter: int,
                 n_actions: int,
                 height: int,  # table length
                 width: int,  # table width
                 terminal_states: int,
                 stopping_threshold: float,
                 seed: int,
                 gamma: float = 0.99,
                 v0: float = 0.0,
                 budget: float = 0.,
                 apply_lag: bool = True,
                 penalty_initial_value: float = 1,
                 penalty_learning_rate: float = 0.01,
                 penalty_min_value: Optional[float] = None,
                 penalty_max_value: Optional[float] = None,
                 log_file=None,
                 N: int = 64,
                 cost_quantile: int = 48,
                 tau_update: float = 0.01,
                 LR_QN: float = 0.001,
                 qnet_layers: Optional[List[int]] = [256, 256],
                 type: str = 'CVaR',
                 prob_yita=None,
                 method=None,
                 recon_obs=True,
                 device=None,
                 weight=None,
                 ):
        super(DistributionalPolicyIterationLagrange, self).__init__()
        self.stopping_threshold = stopping_threshold
        self.gamma = gamma
        self.env = env
        self.log_file = log_file
        self.max_iter = max_iter
        self.n_actions = n_actions
        self.terminal_states = terminal_states
        self.v0 = v0
        self.seed = seed
        self.height = height
        self.width = width
        self.penalty_initial_value = penalty_initial_value
        self.penalty_min_value = penalty_min_value
        self.penalty_max_value = penalty_max_value
        self.penalty_learning_rate = penalty_learning_rate
        self.apply_lag = apply_lag
        self.budget = budget
        self.num_timesteps = 0
        self.admissible_actions = None
        self.cost_quantile = cost_quantile
        self.type = type
        self.prob_yita = prob_yita
        self.method = method
        self.qnet_layers = qnet_layers
        self.LR_QN = LR_QN
        self.tau_update = tau_update
        self.N = N
        self.recon_obs = recon_obs
        self.hl_kappa_k = 1.0
        self.quantile_tau = torch.FloatTensor([i / self.N for i in range(1, self.N + 1)])
        self.device = device
        self.weight = weight
        self._setup_model()

        # Z(s, a)
        if self.recon_obs:
            feature_dim = self.height * self.width
            acs_dim = self.n_actions
        else:
            feature_dim = 2
            acs_dim = 1
        if self.method == 'QRDQN':
            self.qnetwork_local = QRDQN(self.qnet_layers, feature_dim, acs_dim, self.N).to(self.device)
            # self.qnetwork_target = QRDQN(self.qnet_layers, feature_dim, self.n_actions, self.N)
        elif self.method == 'IQN':
            self.qnetwork_local = IQN(self.qnet_layers, feature_dim, acs_dim, self.N, self.device).to(self.device)
            # self.qnetwork_target = IQN(self.qnet_layers, feature_dim, self.n_actions, self.N, self.device)
        elif self.method == 'SplineDQN':
            self.qnetwork_local = SplineDQN(self.qnet_layers, feature_dim, acs_dim, self.N, self.device).to(self.device)
            # self.qnetwork_target = SplineDQN(self.qnet_layers, feature_dim, self.n_actions, self.N, self.device)
        elif self.method == 'NCQR':
            self.qnetwork_local = NCQR(self.qnet_layers, feature_dim, acs_dim, self.N, self.device).to(self.device)
            # self.qnetwork_target = NCQR(self.qnet_layers, feature_dim, self.n_actions, self.N, self.device)

        # Z(s)
        # if self.recon_obs:
        #     self.qnetwork_local = QRDQN(self.height * self.width, self.qnet_layers, self.N)
        #     self.qnetwork_target = QRDQN(self.height * self.width, self.qnet_layers, self.N)
        # else:
        #     self.qnetwork_local = QRDQN(self.env.observation_space.shape, self.qnet_layers, self.N)
        #     self.qnetwork_target = QRDQN(self.env.observation_space.shape, self.qnet_layers, self.N)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR_QN)

    def _setup_model(self) -> None:
        self.dual = DualVariable(self.budget,
                                 self.penalty_learning_rate,
                                 self.penalty_initial_value,
                                 min_clamp=self.penalty_min_value,
                                 max_clamp=self.penalty_max_value)
        self.v_m = self.get_init_v()
        self.pi = self.get_equiprobable_policy()

    def get_init_v(self):
        v_m = self.v0 * np.ones((self.height, self.width))
        # # Value function of terminal state must be 0
        # v0[self.e_x, self.e_y] = 0
        return v_m

    def get_equiprobable_policy(self):
        pi = 1 / self.n_actions * np.ones((self.height, self.width, self.n_actions))
        return pi

    def learn(self,
              total_timesteps: int,
              cost_function: Union[str, Callable],
              latent_info_str: Union[str, Callable] = '',
              callback=None, ):
        policy_stable, dual_stable = False, False
        iter = 0
        for iter in tqdm(range(total_timesteps)):
            if policy_stable and dual_stable:
                print("\nStable at Iteration {0}.".format(iter), file=self.log_file)
                break
            self.num_timesteps += 1
            # Run the policy evaluation
            self.policy_evaluation(cost_function)
            # Run the policy improvement algorithm
            policy_stable = self.policy_improvement(cost_function)
            cumu_reward, length, dual_stable = self.dual_update(cost_function)
        logger.record("train/iterations", iter)
        logger.record("train/cumulative rewards", cumu_reward)
        logger.record("train/length", length)

    def step(self, action):
        return self.env.step(np.asarray([action]))

    def dual_update(self, cost_function):
        """policy rollout for recording training performance"""
        obs = self.env.reset()
        cumu_reward, length = 0, 0
        actions_game, obs_game, costs_game = [], [], []
        while True:
            actions, _ = self.predict(obs=obs, state=None)
            actions_game.append(actions[0])
            obs_primes, rewards, dones, infos = self.step(actions)
            if type(cost_function) is str:
                costs = np.array([info.get(cost_function, 0) for info in infos])
                if isinstance(self.env, VecNormalizeWithCost):
                    orig_costs = self.env.get_original_cost()
                else:
                    orig_costs = costs
            else:
                costs = cost_function(obs, actions)
                orig_costs = costs
            self.admissible_actions = infos[0]['admissible_actions']
            costs_game.append(orig_costs)
            obs = obs_primes
            obs_game.append(obs[0])
            done = dones[0]
            if done:
                break
            cumu_reward += rewards[0]
            length += 1
        costs_game_mean = np.asarray(costs_game).mean()
        self.dual.update_parameter(torch.tensor(costs_game_mean))
        penalty = self.dual.nu().item()
        print("Performance: dual {0}, cost: {1}, states: {2}, "
              "actions: {3}, rewards: {4}.".format(penalty,
                                                   costs_game_mean.tolist(),
                                                   np.asarray(obs_game).tolist(),
                                                   np.asarray(actions_game).tolist(),
                                                   cumu_reward),
              file=self.log_file,
              flush=True)
        dual_stable = True if costs_game_mean == 0 else False
        return cumu_reward, length, dual_stable

    def policy_evaluation(self, cost_function):
        iter = 0

        delta = self.stopping_threshold + 1
        while delta >= self.stopping_threshold and iter <= self.max_iter - 1:
            old_v = self.v_m.copy()
            delta = 0

            # Traverse all states
            for x in range(self.height):
                for y in range(self.width):
                    # Run one iteration of the Bellman update rule for the value function
                    self.bellman_update(old_v, x, y, cost_function)
                    # Compute difference
                    delta = max(delta, abs(old_v[x, y] - self.v_m[x, y]))
            iter += 1
        print("\nThe Policy Evaluation algorithm converged after {} iterations".format(iter),
              file=self.log_file)

    def policy_improvement(self, cost_function):
        """Applies the Policy Improvement step."""
        policy_stable = True

        # Iterate states
        for x in range(self.height):
            for y in range(self.width):
                if [x, y] in self.terminal_states:
                    continue
                old_pi = self.pi[x, y, :].copy()

                # Iterate all actions
                action_values = []
                for action in range(self.n_actions):
                    states = self.env.reset_with_values(info_dicts=[{'states': [x, y]}])
                    assert states[0][0] == x and states[0][1] == y
                    # Compute next state
                    s_primes, rewards, dones, infos = self.step(action)
                    # Get cost from environment.
                    if type(cost_function) is str:
                        costs = np.array([info.get(cost_function, 0) for info in infos])
                        if isinstance(self.env, VecNormalizeWithCost):
                            orig_costs = self.env.get_original_cost()
                        else:
                            orig_costs = costs
                    else:
                        costs = cost_function(states, [action])
                        orig_costs = costs
                    # if x==2:
                    #     print('x:'+str(x)+'y:'+str(y)+'ac:'+str(action)+'cost:'+str(orig_costs))

                    if self.recon_obs:
                        recon_states = torch.tensor(self.idx2vector(states, height=self.height, width=self.width)).to(
                            torch.float32).to(self.device)
                        recon_actions = torch.nn.functional.one_hot(torch.tensor(action), num_classes=self.n_actions).to(self.device)
                    else:
                        recon_states = torch.tensor(states).to(torch.float32)
                        recon_actions = torch.tensor(action)
                    if len(recon_actions.shape) != len(recon_states.shape):
                        recon_actions = recon_actions.view(recon_states.shape[0], -1)
                    with torch.no_grad():
                        if self.method == 'QRDQN' or self.method == 'SplineDQN' or self.method == 'NCQR':
                            distributional_cost_values = self.qnetwork_local.to(self.device)(recon_states,
                                                                                             recon_actions)
                            if self.type == 'VaR':
                                # Caculate the cost values using VaR method
                                cost_values = distributional_cost_values[:, self.cost_quantile - 1].view(
                                    distributional_cost_values.shape[0], 1)
                            elif self.type == 'CVaR':
                                # Caculate the cost values using CVaR method
                                VaR = distributional_cost_values[:, self.cost_quantile - 1].view(
                                    distributional_cost_values.shape[0], 1)
                                alpha = self.cost_quantile / self.N
                                exp = th.mean(abs(distributional_cost_values - VaR), dim=1).view(
                                    distributional_cost_values.shape[0], 1)
                                cost_values = VaR + exp / (1.0 - alpha)
                            elif self.type == 'Prob':

                                num = torch.zeros(distributional_cost_values.shape[0], 1)
                                cost_values = torch.zeros(distributional_cost_values.shape[0], 1)

                                for i in range(0, distributional_cost_values.shape[0]):
                                    for quant in range(0, self.N):
                                        quant_value = distributional_cost_values[i, quant]
                                        if quant_value >= self.prob_yita:
                                            num[i] = num[i] + 1
                                    cost_values[i] = num[i] * 1.0 / self.N
                            elif self.type == 'Expectation':
                                cost_values = torch.mean(distributional_cost_values, dim=1).view(
                                    distributional_cost_values.shape[0], 1)
                            dis_costs = cost_values.squeeze(1)
                        elif self.method == 'IQN':
                            dis_costs, _ = self.qnetwork_local.to(self.device)(recon_states, recon_actions)
                            dis_costs = torch.mean(dis_costs, dim=1)

                    current_penalty = self.dual.nu().item()
                    # lag_costs = self.apply_lag * current_penalty * orig_costs[0]
                    lag_costs = self.apply_lag * current_penalty * (self.weight*dis_costs[0]+(1-self.weight)*orig_costs[0])
                    # Get value
                    curr_val = rewards[0] - lag_costs + self.gamma * self.v_m[s_primes[0][0], s_primes[0][1]]
                    # curr_val = self.v_m[s_primes[0][0], s_primes[0][1]]
                    action_values.append(curr_val.cpu())
                best_actions = np.argwhere(action_values == np.amax(action_values)).flatten().tolist()
                # Define new policy
                self.define_new_policy(x, y, best_actions)

                # Check whether the policy has changed
                if not (old_pi == self.pi[x, y, :]).all():
                    policy_stable = False

        return policy_stable

    def define_new_policy(self, x, y, best_actions):
        """Defines a new policy given the new best actions.
        Args:
            pi (array): numpy array representing the policy
            x (int): x value position of the current state
            y (int): y value position of the current state
            best_actions (list): list with best actions
            actions (list): list of every possible action
        """

        prob = 1 / len(best_actions)

        for a in range(self.n_actions):
            self.pi[x, y, a] = prob if a in best_actions else 0

    def bellman_update(self, old_v, x, y, cost_function):
        if [x, y] in self.terminal_states:
            return
        total = 0
        for action in range(self.n_actions):
            states = self.env.reset_with_values(info_dicts=[{'states': [x, y]}])
            assert states[0][0] == x and states[0][1] == y
            # Get next state
            s_primes, rewards, dones, infos = self.step(action)
            # with torch.no_grad():
            new_action, _ = self.predict(obs=s_primes, state=None)
            # Get cost from environment.
            if type(cost_function) is str:
                costs = np.array([info.get(cost_function, 0) for info in infos])
                if isinstance(self.env, VecNormalizeWithCost):
                    orig_costs = self.env.get_original_cost()
                else:
                    orig_costs = costs
            else:
                costs = cost_function(states, [action])
                orig_costs = costs
            # print(x, y, rewards[0], orig_costs[0])

            # [state, action, reward, cost, s_prime, done]

            # states = preprocess_obs(states, self.env.observation_space)
            # s_primes = preprocess_obs(s_primes, self.env.observation_space)

            if self.recon_obs:
                recon_states = torch.tensor(self.idx2vector(states, height=self.height, width=self.width)).to(
                    torch.float32).to(self.device)
                recon_s_primes = torch.tensor(self.idx2vector(s_primes, height=self.height, width=self.width)).to(
                    torch.float32).to(self.device)
                recon_actions = torch.nn.functional.one_hot(torch.tensor(action), num_classes=self.n_actions).to(
                    self.device)
                recon_new_actions = torch.nn.functional.one_hot(torch.tensor(new_action),
                                                                num_classes=self.n_actions).to(self.device)
            else:
                recon_states = torch.tensor(states).to(torch.float32)
                recon_s_primes = torch.tensor(s_primes).to(torch.float32)
                recon_actions = torch.tensor(action)
                recon_new_actions = torch.tensor(new_action)
            if len(recon_actions.shape) != len(recon_states.shape):
                recon_actions = recon_actions.view(recon_states.shape[0], -1)
                recon_new_actions = recon_new_actions.view(recon_states.shape[0], -1)
            if self.method == 'QRDQN' or self.method == 'SplineDQN' or self.method == 'NCQR':
                with torch.no_grad():
                    Q_targets_next = self.qnetwork_local.to(self.device)(recon_s_primes, recon_new_actions)

                q_costs = torch.tensor(orig_costs).view(-1, 1).to(self.device)
                q_dones = torch.tensor(int(dones)).view(-1, 1).to(self.device)

                Q_targets = q_costs + (self.gamma * Q_targets_next * (1 - q_dones))

                Q_expected = self.qnetwork_local.to(self.device)(recon_states, recon_actions)

                T_theta_tile = Q_targets.view(-1, self.N, 1).expand(-1, self.N, self.N).to(self.device)  # target
                theta_a_tile = Q_expected.view(-1, 1, self.N).expand(-1, self.N, self.N).to(self.device)  # local

                quantile_tau = torch.arange(0.5 * (1 / self.N), 1, 1 / self.N).view(1, self.N).to(self.device)

                error_loss = T_theta_tile - theta_a_tile
                huber_loss = F.smooth_l1_loss(theta_a_tile, T_theta_tile.detach(), reduction='none')
                value_loss = (quantile_tau - (error_loss < 0).float()).abs() * huber_loss
                td_loss = value_loss.mean(dim=2).sum(dim=1).mean()

            elif self.method == 'IQN':
                with torch.no_grad():
                    Q_targets_next, _ = self.qnetwork_local.to(self.device)(recon_s_primes, recon_new_actions)

                q_costs = torch.tensor(orig_costs).view(-1, 1)
                q_dones = torch.tensor(int(dones)).view(-1, 1)

                Q_targets = q_costs + (self.gamma * Q_targets_next * (1 - q_dones))

                Q_expected, taus = self.qnetwork_local.to(self.device)(recon_states, recon_actions)

                T_theta_tile = Q_targets.view(-1, self.N, 1).expand(-1, self.N, self.N).to(self.device)  # target
                theta_a_tile = Q_expected.view(-1, 1, self.N).expand(-1, self.N, self.N).to(self.device)  # local
                taus = taus.view(1, self.N).to(self.device)

                error_loss = T_theta_tile - theta_a_tile
                huber_loss = F.smooth_l1_loss(theta_a_tile, T_theta_tile.detach(), reduction='none')
                value_loss = (taus - (error_loss < 0).float()).abs() * huber_loss
                td_loss = value_loss.mean(dim=2).sum(dim=1).mean()

            self.optimizer.zero_grad()
            td_loss.backward()
            self.optimizer.step()

            # for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            #     target_param.data.copy_(
            #         self.tau_update * local_param.data + (1.0 - self.tau_update) * target_param.data)

            # gamma_values = self.gamma * old_v[s_primes[0][0], s_primes[0][1]]
            # current_penalty = self.dual.nu().item()
            # lag_costs = self.apply_lag * current_penalty * orig_costs[0]
            # total += self.pi[x, y, action] * (rewards[0] - lag_costs + gamma_values)

            with torch.no_grad():
                if self.method == 'QRDQN' or self.method == 'SplineDQN' or self.method == 'NCQR':
                    distributional_cost_values = self.qnetwork_local.to(self.device)(recon_states, recon_actions)
                    if self.type == 'VaR':
                        # Caculate the cost values using VaR method
                        cost_values = distributional_cost_values[:, self.cost_quantile - 1].view(
                            distributional_cost_values.shape[0], 1)
                    elif self.type == 'CVaR':
                        # Caculate the cost values using CVaR method
                        VaR = distributional_cost_values[:, self.cost_quantile - 1].view(
                            distributional_cost_values.shape[0], 1)
                        alpha = self.cost_quantile / self.N
                        exp = th.mean(abs(distributional_cost_values - VaR), dim=1).view(
                            distributional_cost_values.shape[0], 1)
                        cost_values = VaR + exp / (1.0 - alpha)
                    elif self.type == 'Prob':

                        num = torch.zeros(distributional_cost_values.shape[0], 1)
                        cost_values = torch.zeros(distributional_cost_values.shape[0], 1)

                        for i in range(0, distributional_cost_values.shape[0]):
                            for quant in range(0, self.N):
                                quant_value = distributional_cost_values[i, quant]
                                if quant_value >= self.prob_yita:
                                    num[i] = num[i] + 1
                            cost_values[i] = num[i] * 1.0 / self.N
                    elif self.type == 'Expectation':
                        cost_values = torch.mean(distributional_cost_values, dim=1).view(
                            distributional_cost_values.shape[0], 1)
                    dis_costs = cost_values.squeeze(1)
                elif self.method == 'IQN':
                    dis_costs, _ = self.qnetwork_local.to(self.device)(recon_states, recon_actions)
                    dis_costs = torch.mean(dis_costs, dim=1)

            gamma_values = self.gamma * old_v[s_primes[0][0], s_primes[0][1]]
            current_penalty = self.dual.nu().item()
            lag_costs = self.apply_lag * current_penalty * (
                        self.weight * dis_costs[0] + (1 - self.weight) * orig_costs[0])
            # lag_costs = self.apply_lag * current_penalty * orig_costs[0]
            total += self.pi[x, y, action] * (rewards[0] - lag_costs + gamma_values)

        self.v_m[x, y] = total

    def predict(self, obs, state, deterministic=None):
        policy_prob = copy.copy(self.pi[int(obs[0][0]), int(obs[0][1])])
        if self.admissible_actions is not None:
            for c_a in range(self.n_actions):
                if c_a not in self.admissible_actions:
                    policy_prob[c_a] = -float('inf')
        best_actions = np.argwhere(policy_prob == np.amax(policy_prob)).flatten().tolist()
        action = random.choice(best_actions)
        return np.asarray([action]), state

    def save(self, save_path):
        state_dict = dict(
            pi=self.pi,
            v_m=self.v_m,
            gamma=self.gamma,
            max_iter=self.max_iter,
            n_actions=self.n_actions,
            terminal_states=self.terminal_states,
            seed=self.seed,
            height=self.height,
            width=self.width,
            budget=self.budget,
            num_timesteps=self.num_timesteps,
            stopping_threshold=self.stopping_threshold,
        )
        torch.save(state_dict, save_path)

    def idx2vector(self, indices, height, width):
        vector_all = []
        if isinstance(indices, torch.Tensor):
            for idx in indices:
                map = np.zeros(shape=[height, width])
                x, y = int(torch.round(idx[0].float())), int(torch.round(idx[1].float()))
                # if x - idx[0] != 0:
                #     print('debug')
                map[x, y] = 1  # + idx[0] - x + idx[1] - y
                vector_all.append(map.flatten())
            return torch.Tensor(np.array(vector_all)).to(self.device)
        else:
            for idx in indices:
                map = np.zeros(shape=[height, width])
                x, y = int(round(idx[0], 0)), int(round(idx[1], 0))
                # if x - idx[0] != 0:
                #     print('debug')
                map[x, y] = 1  # + idx[0] - x + idx[1] - y
                vector_all.append(map.flatten())
            return np.asarray(vector_all)

    def get_cost_distribution(self, obs: th.Tensor, acs: th.Tensor, deterministic: bool = False) -> Tuple[
        th.Tensor, th.Tensor, th.Tensor]:  # important-----------------
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        with torch.no_grad():
            if self.recon_obs:
                features = self.idx2vector(obs, height=self.height, width=self.width)
                acs = torch.nn.functional.one_hot(torch.tensor(acs), num_classes=self.n_actions)

            if self.method == 'QRDQN' or self.method == 'SplineDQN' or self.method == 'NCQR':
                # if self.input_action:
                #     distributional_cost_values = self.cost_value_net_local(th.cat([features, actions], dim=1))
                # else:
                #     distributional_cost_values = self.cost_value_net_local(features)
                distributional_cost_values = self.qnetwork_local.to(self.device)(features, acs)
            elif self.method == 'IQN':
                # if self.input_action:
                #     distributional_cost_values, _ = self.cost_value_net_local(th.cat([features, actions], dim=1))
                # else:
                #     distributional_cost_values, _ = self.cost_value_net_local(features)
                distributional_cost_values, _ = self.qnetwork_local.to(self.device)(features, acs)

            # Caculate the cost values using VaR method
            cost_values_var = distributional_cost_values[:, self.cost_quantile - 1].view(
                distributional_cost_values.shape[0], 1)

            # Caculate the cost values using CVaR method
            VaR = cost_values_var
            alpha = self.cost_quantile / self.N
            exp = th.mean(abs(distributional_cost_values - VaR), dim=1).view(distributional_cost_values.shape[0], 1)
            cost_values_cvar = VaR + exp / (1.0 - alpha)

            # Caculate the cost values using Expectation method
            cost_values_exp = torch.mean(distributional_cost_values, dim=1).view(distributional_cost_values.shape[0], 1)

        return distributional_cost_values, cost_values_var, cost_values_cvar, cost_values_exp


def load_pi(model_path, iter_msg, log_file):
    if iter_msg == 'best':
        model_path = os.path.join(model_path, "best_nominal_model")
    else:
        model_path = os.path.join(model_path, 'model_{0}_itrs'.format(iter_msg), 'nominal_agent')
    print('Loading model from {0}'.format(model_path), flush=True, file=log_file)

    state_dict = torch.load(model_path)

    pi = state_dict["pi"]
    v_m = state_dict["v_m"]
    gamma = state_dict["gamma"]
    max_iter = state_dict["max_iter"]
    n_actions = state_dict["n_actions"]
    terminal_states = state_dict["terminal_states"]
    seed = state_dict["seed"]
    height = state_dict["height"]
    width = state_dict["width"]
    budget = state_dict["budget"]
    stopping_threshold = state_dict["stopping_threshold"]

    create_iteration_agent = lambda: DistributionalPolicyIterationLagrange(
        env=None,
        max_iter=max_iter,
        n_actions=n_actions,
        height=height,  # table length
        width=width,  # table width
        terminal_states=terminal_states,
        stopping_threshold=stopping_threshold,
        seed=seed,
        gamma=gamma,
        budget=budget, )
    iteration_agent = create_iteration_agent()
    iteration_agent.pi = pi
    iteration_agent.v_m = v_m

    return iteration_agent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4


def fan_in_uniform_init(tensor, fan_in=None):
    """Utility function for initializing actor and critic"""
    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1. / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -w, w)


class QRDQN(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_dim, num_support, seed: int = 0):
        super(QRDQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        num_outputs = action_dim

        # Layer 1
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.ln1 = nn.LayerNorm(hidden_size[0])

        # Layer 2
        # In the second layer the actions will be inserted also
        self.linear2 = nn.Linear(hidden_size[0] + num_outputs, hidden_size[1])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        # Output layer (single value)
        self.V = nn.Linear(hidden_size[1], num_support)

        # Weight Init
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        nn.init.uniform_(self.V.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.V.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs, actions):
        x = inputs

        # Layer 1
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        # Layer 2
        x = torch.cat((x, actions), 1)  # Insert the actions
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)

        # Output
        V = self.V(x)
        return V


class IQN(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_dim, num_support, device, seed: int = 0):
        super(IQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        num_outputs = action_dim
        self.num_support = num_support

        # Layer 1
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.ln1 = nn.LayerNorm(hidden_size[0])

        # Layer 2
        # In the second layer the actions will be inserted also
        self.linear2 = nn.Linear(hidden_size[0] + num_outputs, hidden_size[1])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        # Output layer (single value)
        self.V = nn.Linear(hidden_size[1], 1)

        # phi
        self.phi = nn.Linear(1, hidden_size[1], bias=False)
        self.phi_bias = nn.Parameter(torch.zeros(hidden_size[1]), requires_grad=True)

        self.linear3 = nn.Linear(hidden_size[1], hidden_size[1])

        # Weight Init
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        nn.init.uniform_(self.V.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.V.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs, actions):
        x = inputs

        # Layer 1
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        # Layer 2
        x = torch.cat((x, actions), 1)  # Insert the actions
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)

        # tau
        tau = torch.rand(self.num_support, 1).to(device)
        quants = torch.arange(0, self.num_support, 1.0).to(device)
        cos_trans = torch.cos(quants * tau * np.pi).unsqueeze(2)  # (num_support, num_support, 1)
        rand_feat = F.relu(self.phi(cos_trans).mean(1) + self.phi_bias.unsqueeze(0)).unsqueeze(0)

        x = x.unsqueeze(1)
        x = x * rand_feat

        x = F.relu(self.linear3(x))

        # Output
        V = self.V(x).transpose(1, 2)  # (bs_size, 1, num_support)
        V = V.squeeze(1)
        return V, tau


class SplineDQN(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_dim, num_support, device, seed: int = 0):
        super(SplineDQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        num_outputs = action_dim
        self.device = device

        self.num_support = num_support
        self.K = num_support

        self.min_bin_width = 1e-3
        self.min_bin_height = 1e-3
        self.min_derivative = 1e-3

        # Layer 1
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.ln1 = nn.LayerNorm(hidden_size[0])

        # Layer 2
        # In the second layer the actions will be inserted also
        self.linear2 = nn.Linear(hidden_size[0] + num_outputs, hidden_size[1])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        # Output layer (knots)
        self.V = nn.Linear(hidden_size[1], (3 * self.K - 1))

        # Scale
        self.alpha = nn.Linear(hidden_size[1], 1)
        self.beta = nn.Linear(hidden_size[1], 1)

        # Weight Init
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        fan_in_uniform_init(self.V.weight)
        fan_in_uniform_init(self.V.bias)

        nn.init.uniform_(self.alpha.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.alpha.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

        nn.init.uniform_(self.beta.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.beta.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs, actions):
        batch_size = inputs.size(0)
        x = inputs

        # Layer 1
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        # Layer 2
        x = torch.cat((x, actions), 1)  # Insert the actions
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)

        # Output
        spline_param = self.V(x)
        scale_a = self.alpha(x)
        scale_a = torch.exp(scale_a)
        scale_b = self.beta(x)

        # split the last dimention to W, H, D
        W, H, D = torch.split(spline_param, self.K, dim=1)
        W, H = torch.softmax(W, dim=1), torch.softmax(H, dim=1)
        W = self.min_bin_width + (1 - self.min_bin_width * self.K) * W
        H = self.min_bin_height + (1 - self.min_bin_height * self.K) * H
        D = self.min_derivative + F.softplus(D)
        D = F.pad(D, pad=(1, 1))
        constant = np.log(np.exp(1 - 1e-3) - 1)
        D[..., 0] = constant
        D[..., -1] = constant

        # start and end x of each bin
        cumwidths = torch.cumsum(W, dim=-1).to(self.device)
        cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
        cumwidths[..., -1] = 1.0
        widths = cumwidths[..., 1:] - cumwidths[..., :-1]  # (batch_sz, K)

        # start and end y of each bin
        cumheights = torch.cumsum(H, dim=-1)
        cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
        cumheights = (scale_a * cumheights + scale_b).to(self.device)
        heights = cumheights[..., 1:] - cumheights[..., :-1]

        # get bin index for each tau
        tau = torch.arange(0.5 * (1 / self.num_support), 1, 1 / self.num_support).to(self.device)
        tau = tau.expand((batch_size, self.num_support))

        cumwidths_expand = cumwidths.unsqueeze(dim=1).to(self.device)
        cumwidths_expand = cumwidths_expand.expand(-1, self.num_support, -1)  # (batch_sz, num_support, K+1)

        bin_idx = self.searchsorted_(cumwidths_expand, tau)

        # collect number
        input_cumwidths = cumwidths.gather(-1, bin_idx)
        input_bin_widths = widths.gather(-1, bin_idx)

        input_cumheights = cumheights.gather(-1, bin_idx)
        input_heights = heights.gather(-1, bin_idx)

        delta = heights / widths

        input_delta = delta.gather(-1, bin_idx)

        input_derivatives = D.to(self.device).gather(-1, bin_idx)
        input_derivatives_plus_one = D[..., 1:].to(self.device).gather(-1, bin_idx)

        # calculate quadratic spline for each tau
        theta = (tau - input_cumwidths) / input_bin_widths

        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + (
                input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
        outputs = input_cumheights + numerator / denominator

        return outputs

    def searchsorted_(self, bin_locations, inputs):
        return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


class NCQR(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_dim, num_support, device, seed: int = 0):
        super(NCQR, self).__init__()
        self.seed = torch.manual_seed(seed)
        num_outputs = action_dim
        self.device = device

        # Layer 1
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.ln1 = nn.LayerNorm(hidden_size[0])

        # Layer 2
        # In the second layer the actions will be inserted also
        self.linear2 = nn.Linear(hidden_size[0] + num_outputs, hidden_size[1])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        # Output layer (single value)
        self.V = nn.Linear(hidden_size[1], num_support)

        # Scale
        self.alpha = nn.Linear(hidden_size[1], 1)
        self.beta = nn.Linear(hidden_size[1], 1)

        # Weight Init
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        fan_in_uniform_init(self.V.weight)
        fan_in_uniform_init(self.V.bias)

        nn.init.uniform_(self.alpha.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.alpha.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

        nn.init.uniform_(self.beta.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.beta.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs, actions):
        x = inputs

        # Layer 1
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        # Layer 2
        x = torch.cat((x, actions), 1)  # Insert the actions
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)

        # Output
        quant = self.V(x)
        quant = torch.softmax(quant, dim=-1)
        quant = torch.cumsum(quant, dim=-1)

        # scale
        scale_a = F.relu(self.alpha(x))
        scale_b = self.beta(x)

        V = scale_a * quant + scale_b
        return V