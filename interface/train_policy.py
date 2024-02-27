import json
import os
import sys
import time
import gym
import numpy as np
import datetime
import yaml
from matplotlib import pyplot as plt

cwd = os.getcwd()
sys.path.append(cwd.replace('/interface', ''))
from common.cns_sampler import ConstrainedRLSampler
from common.cns_visualization import traj_visualization_2d, constraint_visualization_2d
from utils.true_constraint_functions import get_true_cost_function
from stable_baselines3.iteration import PolicyIterationLagrange, DistributionalPolicyIterationLagrange
from utils.env_utils import check_if_duplicate_seed
from common.cns_env import make_train_env, make_eval_env, sync_envs_normalization_ppo
from utils.plot_utils import plot_curve
from exploration.exploration import ExplorationRewardCallback
from stable_baselines3 import PPO, PPOLagrangian, PPODistributionalLagrangian, PPODistributionalLagrangianCostAdv
from stable_baselines3.common import logger
from common.cns_evaluation import evaluate_icrl_policy
from stable_baselines3.common.vec_env import VecNormalize

from utils.data_utils import ProgressBarManager, del_and_make, read_args, load_config, process_memory
from utils.model_utils import load_ppo_config, load_policy_iteration_config
import warnings
import torch
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mtick

warnings.filterwarnings("ignore")


def train(args):
    config, debug_mode, log_file_path, partial_data, num_threads, seed = load_config(args)

    if num_threads > 1:
        multi_env = True
        config.update({'multi_env': True})
    else:
        multi_env = False
        config.update({'multi_env': False})

    if log_file_path is not None:
        log_file = open(log_file_path, 'w')
    else:
        log_file = None
    debug_msg = ''
    if debug_mode:
        config['verbose'] = 2  # the verbosity level: 0 no output, 1 info, 2 debug
        if 'PPO' in config.keys():
            config['PPO']['forward_timesteps'] = 2000  # 2000
            config['PPO']['n_steps'] = 200
        else:
            config['iteration']['max_iter'] = 2
        config['running']['n_eval_episodes'] = 10
        config['running']['save_every'] = 1
        debug_msg = 'debug-'
        partial_data = True
    if partial_data:
        debug_msg += 'part-'

    if num_threads is not None:
        config['env']['num_threads'] = num_threads

    print(json.dumps(config, indent=4), file=log_file, flush=True)
    current_time_date = datetime.datetime.now().strftime('%b-%d-%Y-%H:%M')
    # today = datetime.date.today()
    # currentTime = today.strftime("%b-%d-%Y-%h-%m")
    save_model_mother_dir = '{0}/{1}/{5}{2}{3}-{4}-seed_{6}/'.format(
        config['env']['save_dir'],
        config['task'],
        args.config_file.split('/')[-1].split('.')[0],
        '-multi_env' if multi_env else '',
        current_time_date,
        debug_msg,
        seed
    )

    if not os.path.exists('{0}/{1}/'.format(config['env']['save_dir'], config['task'])):
        os.mkdir('{0}/{1}/'.format(config['env']['save_dir'], config['task']))
    if not os.path.exists(save_model_mother_dir):
        os.mkdir(save_model_mother_dir)
    print("Saving to the file: {0}".format(save_model_mother_dir), file=log_file, flush=True)

    with open(os.path.join(save_model_mother_dir, "model_hyperparameters.yaml"), "w") as hyperparam_file:
        yaml.dump(config, hyperparam_file)

    mem_prev = process_memory()
    time_prev = time.time()
    # Create the vectorized environments
    train_env, env_configs = make_train_env(env_id=config['env']['train_env_id'],
                                            config_path=config['env']['config_path'],
                                            save_dir=save_model_mother_dir,
                                            base_seed=seed,
                                            group=config['group'],
                                            num_threads=num_threads,
                                            use_cost=config['env']['use_cost'],
                                            normalize_obs=not config['env']['dont_normalize_obs'],
                                            normalize_reward=not config['env']['dont_normalize_reward'],
                                            normalize_cost=not config['env']['dont_normalize_cost'],
                                            cost_info_str=config['env']['cost_info_str'],
                                            reward_gamma=config['env']['reward_gamma'],
                                            cost_gamma=config['env']['cost_gamma'],
                                            log_file=log_file,
                                            part_data=partial_data,
                                            multi_env=multi_env,
                                            noise_mean=config['env']['noise_mean'] if 'Noise' in config['env'][
                                                'train_env_id'] else None,
                                            noise_std=config['env']['noise_std'] if 'Noise' in config['env'][
                                                'train_env_id'] else None,
                                            noise_seed=seed if 'Noise' in config['env'][
                                                'train_env_id'] else None,
                                            circle_info=config['env']['circle_info'] if 'Circle' in config[
                                                'env']['train_env_id'] else None,
                                            )

    save_test_mother_dir = os.path.join(save_model_mother_dir, "test/")
    if not os.path.exists(save_test_mother_dir):
        os.mkdir(save_test_mother_dir)

    eval_env, env_configs = make_eval_env(env_id=config['env']['eval_env_id'],
                                          config_path=config['env']['config_path'],
                                          save_dir=save_test_mother_dir,
                                          group=config['group'],
                                          use_cost=config['env']['use_cost'],
                                          normalize_obs=not config['env']['dont_normalize_obs'],
                                          cost_info_str=config['env']['cost_info_str'],
                                          log_file=log_file,
                                          part_data=partial_data,
                                          noise_mean=config['env']['noise_mean'] if 'Noise' in config['env'][
                                              'train_env_id'] else None,
                                          noise_std=config['env']['noise_std'] if 'Noise' in config['env'][
                                              'train_env_id'] else None,
                                          noise_seed=seed if 'Noise' in config['env'][
                                              'train_env_id'] else None,
                                          circle_info=config['env']['circle_info'] if 'Circle' in config[
                                              'env']['train_env_id'] else None,
                                          )

    if 'WGW' in config['env']['train_env_id']:
        sampler = ConstrainedRLSampler(rollouts=10,
                                       store_by_game=True,  # I move the step out
                                       cost_info_str=None,
                                       sample_multi_env=False,
                                       env_id=config['env']['eval_env_id'],
                                       env=eval_env)

    mem_loading_environment = process_memory()
    time_loading_environment = time.time()
    print("Loading environment consumed memory: {0:.2f}/{1:.2f} and time {2:.2f}:".format(
        float(mem_loading_environment - mem_prev) / 1000000,
        float(mem_loading_environment) / 1000000,
        time_loading_environment - time_prev),
        file=log_file, flush=True)
    mem_prev = mem_loading_environment
    time_prev = time_loading_environment

    # Set specs
    is_discrete = isinstance(train_env.action_space, gym.spaces.Discrete)
    # print('is_discrete', is_discrete, file=log_file, flush=True)
    obs_dim = train_env.observation_space.shape[0]
    acs_dim = train_env.action_space.n if is_discrete else train_env.action_space.shape[0]

    # Logger
    if log_file is None:
        ppo_logger = logger.HumanOutputFormat(sys.stdout)
    else:
        ppo_logger = logger.HumanOutputFormat(log_file)

    if 'WGW' in config['env']['train_env_id']:
        # with open(config['env']['config_path'], "r") as config_file:
        #     env_configs = yaml.safe_load(config_file)
        ture_cost_function = get_true_cost_function(env_id=config['env']['train_env_id'],
                                                    env_configs=env_configs)
        constraint_visualization_2d(cost_function=ture_cost_function,
                                    feature_range=config['env']["visualize_info_ranges"],
                                    select_dims=config['env']["record_info_input_dims"],
                                    num_points_per_feature=env_configs['map_height'],
                                    obs_dim=2,
                                    acs_dim=1,
                                    save_path=save_model_mother_dir
                                    )

    if config['group'] == 'PPO':
        ppo_parameters = load_ppo_config(config=config,
                                         train_env=train_env,
                                         seed=seed,
                                         log_file=log_file)
        forward_timesteps = config['PPO']['forward_timesteps']
        create_policy_agent = lambda: PPO(**ppo_parameters)
    elif config['group'] == 'PPO-Lag':
        ppo_parameters = load_ppo_config(config=config,
                                         train_env=train_env,
                                         seed=seed,
                                         log_file=log_file)
        forward_timesteps = config['PPO']['forward_timesteps']
        if 'WGW' in config['env']['train_env_id']:
            ppo_parameters.update({
                "env_configs": env_configs,
            })
        if config['PPO']['policy_name'] == "DistributionalTwoCriticsMlpPolicy":
            if 'cost_adv' in config['PPO'].keys() and config['PPO']['cost_adv'] == True:
                create_policy_agent = lambda: PPODistributionalLagrangianCostAdv(**ppo_parameters)
            else:
                create_policy_agent = lambda: PPODistributionalLagrangian(**ppo_parameters)
        else:
            create_policy_agent = lambda: PPOLagrangian(**ppo_parameters)
    elif config['group'] == 'PI-Lag':
        iteration_parameters = load_policy_iteration_config(config=config,
                                                            env_configs=env_configs,
                                                            train_env=train_env,
                                                            seed=seed,
                                                            log_file=log_file)
        forward_timesteps = config['iteration']['forward_timesteps']
        if 'Distributional' in config.keys():
            create_policy_agent = lambda: DistributionalPolicyIterationLagrange(**iteration_parameters)
        else:
            create_policy_agent = lambda: PolicyIterationLagrange(**iteration_parameters)
    else:
        raise ValueError("Unknown ppo group: {0}".format(config['group']))
    policy_agent = create_policy_agent()

    # Callbacks
    all_callbacks = []
    if 'PPO' in config['group'] and config['PPO']['use_curiosity_driven_exploration']:
        explorationCallback = ExplorationRewardCallback(obs_dim, acs_dim, device=config.device)
        all_callbacks.append(explorationCallback)

    timesteps = 0.
    mem_before_training = process_memory()
    time_before_training = time.time()
    print("Setting model consumed memory: {0:.2f}/{1:.2f} and time: {2:.2f}".format(
        float(mem_before_training - mem_prev) / 1000000,
        float(mem_before_training) / 1000000,
        time_before_training - time_prev),
        file=log_file, flush=True)
    mem_prev = mem_before_training
    time_prev = time_before_training

    # Train
    start_time = time.time()
    print("\nBeginning training", file=log_file, flush=True)
    best_true_reward = -np.inf
    for itr in range(config['running']['n_iters']):

        # Update agent
        with ProgressBarManager(forward_timesteps) as callback:
            if config['group'] == 'PPO':
                policy_agent.learn(total_timesteps=forward_timesteps,
                                   callback=callback)
            else:
                policy_agent.learn(
                    total_timesteps=forward_timesteps,
                    cost_function=config['env']['cost_info_str'],  # Cost should come from cost wrapper
                    callback=[callback] + all_callbacks
                )
            forward_metrics = logger.Logger.CURRENT.name_to_value
            timesteps += policy_agent.num_timesteps

        mem_during_training = process_memory()
        time_during_training = time.time()
        print("Itr: {3}, Training consumed memory: {0:.2f}/{1:.2f} and time {2:.2f}".format(
            float(mem_during_training - mem_prev) / 1000000,
            float(mem_during_training) / 1000000,
            time_during_training - time_prev,
            itr), file=log_file, flush=True)
        mem_prev = mem_during_training
        time_prev = time_during_training

        # Evaluate:
        # reward on true environment
        save_path = save_model_mother_dir + '/model_{0}_itrs'.format(itr)
        if itr % config['running']['save_every'] == 0:
            del_and_make(save_path)
        else:
            save_path = None
        mean_reward, std_reward, mean_nc_reward, std_nc_reward, record_infos, costs = \
            evaluate_icrl_policy(model=policy_agent,
                                 env=eval_env,
                                 render=True if 'Circle' in config['env']['train_env_id'] else False,
                                 record_info_names=config['env']["record_info_names"],
                                 n_eval_episodes=config['running']['n_eval_episodes'],
                                 deterministic=False,
                                 cost_info_str=config['env']['cost_info_str'],
                                 save_path=save_path, )
        if 'WGW' in config['env']['train_env_id'] and itr % config['running']['save_every'] == 0:
            orig_observations, observations, actions, rewards, sum_rewards, lengths = sampler.sample_from_agent(
                policy_agent=policy_agent,
                new_env=eval_env,
            )
            traj_visualization_2d(config=config,
                                  observations=orig_observations,
                                  save_path=save_path, )
            if 'PPO' not in config['group']:
                plt.figure(figsize=(5, 5))
                plt.matshow(policy_agent.v_m, origin='lower')
                plt.gca().xaxis.set_ticks_position('bottom')
                plt.colorbar()
                plt.savefig(os.path.join(save_path, "v_m_aid.png"))

        # Save
        if itr % config['running']['save_every'] == 0:
            # path = save_model_mother_dir + '/model_{0}_itrs'.format(itr)
            # del_and_make(path)
            policy_agent.save(os.path.join(save_path, "nominal_agent"))
            if isinstance(train_env, VecNormalize):
                train_env.save(os.path.join(save_path, "train_env_stats.pkl"))
            if costs is not None:

                for record_info_name in config['env']["record_info_names"]:
                    if record_info_name != 'obs':
                        plot_record_infos, plot_costs = zip(*sorted(zip(record_infos[record_info_name], costs)))
                        plot_curve(draw_keys=[record_info_name],
                                   x_dict={record_info_name: plot_record_infos},
                                   y_dict={record_info_name: plot_costs},
                                   xlabel=record_info_name,
                                   ylabel='cost',
                                   save_name=os.path.join(save_path, "{0}".format(record_info_name)),
                                   apply_scatter=True
                                   )

                    # plot in WGW
                    if record_info_name == 'obs' and 'Distributional' in config.keys() and 'PI' in config['group']:

                        obs_data_list = []
                        acs_data_list = []
                        for i in range(1,3):
                            for j in range(0,7):
                                for k in range(8):
                                    obs_data_list.append([i,j])
                        for i in range(4,6):
                            for j in range(0,7):
                                for k in range(8):
                                    obs_data_list.append([i,j])
                        for i in range(int(len(obs_data_list)/8)):
                            for j in range(8):
                                acs_data_list.append([j])
                        plot_obs = np.array(obs_data_list)
                        plot_acs = np.array(acs_data_list)

                        for i in range(len(plot_obs)):
                            sampled_plot_obs = torch.tensor(plot_obs[i]).view(1, -1).to(policy_agent.device)
                            sampled_plot_acs = torch.tensor(plot_acs[i]).to(policy_agent.device)
                            plot_cost_distribution, plot_cost_var, plot_cost_cvar, plot_cost_exp = \
                                policy_agent.get_cost_distribution(sampled_plot_obs, sampled_plot_acs)

                            # convert to numpy
                            plot_cost_distribution = plot_cost_distribution.view(-1).cpu().numpy()
                            quantile_tau = torch.arange(0.5 * (1 / policy_agent.N), 1, 1 / policy_agent.N).cpu().numpy()
                            plot_cost_var = np.full(quantile_tau.shape[0], plot_cost_var.cpu())
                            plot_cost_cvar = np.full(quantile_tau.shape[0], plot_cost_cvar.cpu())
                            plot_cost_exp = np.full(quantile_tau.shape[0], plot_cost_exp.cpu())
                            # plot_true_cost = np.full(quantile_tau.shape[0], plot_costs[i])

                            # draw pictures
                            fig = plt.figure(figsize=(8, 8))
                            ax = fig.add_subplot(1, 1, 1)
                            plt.title('Estimated Constraint at State:' + str(plot_obs[i])+'with Action'+str(plot_acs[i]), fontsize=16)
                            plt.xlabel('Quantile', fontsize=16)
                            plt.ylabel('Quantile Value', fontsize=16)
                            plt.xticks(fontsize=16)
                            plt.yticks(fontsize=16)
                            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%01.2lf'))
                            plt.plot(quantile_tau, plot_cost_distribution, color='red', label='Distribution')
                            plt.plot(quantile_tau, plot_cost_var, color='blue', label='Var')
                            plt.plot(quantile_tau, plot_cost_cvar, color='orange', label='Cvar')
                            plt.plot(quantile_tau, plot_cost_exp, color='green', label='Expectation')
                            plt.legend(fontsize=16)
                            plt.savefig(os.path.join(save_path, "{0}.png".format('pos:' + str(plot_obs[i])+'ac'+str(plot_acs[i]))))
                            plt.clf()

        # (2) best
        if mean_nc_reward > best_true_reward:
            print("Saving new best model", flush=True, file=log_file)
            policy_agent.save(os.path.join(save_model_mother_dir, "best_nominal_model"))
            if isinstance(train_env, VecNormalize):
                train_env.save(os.path.join(save_model_mother_dir, "train_env_stats.pkl"))

        # Update best metrics
        if mean_nc_reward > best_true_reward:
            best_true_reward = mean_nc_reward

        # Collect metrics
        metrics = {
            "time(m)": (time.time() - start_time) / 60,
            "run_iter": itr,
            "timesteps": timesteps,
            "true/mean_nc_reward": mean_nc_reward,
            "true/std_nc_reward": std_nc_reward,
            "true/mean_reward": mean_reward,
            "true/std_reward": std_reward,
            "best_true/best_reward": best_true_reward
        }

        metrics.update({k.replace("train/", "forward/"): v for k, v in forward_metrics.items()})

        # Log
        if config['verbose'] > 0:
            ppo_logger.write(metrics, {k: None for k in metrics.keys()}, step=itr)

        mem_during_testing = process_memory()
        time_during_testing = time.time()
        print("Itr: {3}, Validating consumed memory: {0:.2f}/{1:.2f} and time {2:.2f}".format(
            float(mem_during_testing - mem_prev) / 1000000,
            float(mem_during_testing) / 1000000,
            time_during_testing - time_prev,
            itr), file=log_file, flush=True)
        mem_prev = mem_during_testing
        time_prev = time_during_testing


if __name__ == "__main__":
    args = read_args()
    train(args)
