import copy
import os
import numpy as np
import sys
cwd = os.getcwd()
sys.path.append(cwd.replace('/interface', ''))

from plot_results.plot_results_dirs import get_plot_results_dir
from utils.data_utils import read_running_logs, compute_moving_average, mean_std_plot_results, \
    mean_std_plot_valid_rewards, mean_std_test_results
from utils.plot_utils import plot_curve, plot_shadow_curve


def plot_results(mean_results_moving_avg_dict,
                 std_results_moving_avg_dict,
                 episode_plots,
                 ylim, label, method_names,
                 save_label,
                 legend_dict=None,
                 linestyle_dict=None,
                 legend_size=20,
                 # legend_size=20,
                 axis_size=None,
                 img_size=None,
                 title=None):
    plot_mean_y_dict = {}
    plot_std_y_dict = {}
    plot_x_dict = {}
    for method_name in method_names:
        # plot_x_dict.update({method_name: [i for i in range(len(mean_results_moving_avg_dict[method_name]))]})
        plot_x_dict.update({method_name: episode_plots[method_name]})
        plot_mean_y_dict.update({method_name: mean_results_moving_avg_dict[method_name]})
        plot_std_y_dict.update({method_name: std_results_moving_avg_dict[method_name]})
    if save_label is not None:
        plot_name = './plot_results/{0}'.format(save_label)
    else:
        plot_name = None
    plot_shadow_curve(draw_keys=method_names,
                      x_dict_mean=plot_x_dict,
                      y_dict_mean=plot_mean_y_dict,
                      x_dict_std=plot_x_dict,
                      y_dict_std=plot_std_y_dict,
                      img_size=img_size if img_size is not None else (6, 5.8),
                      ylim=ylim,
                      title=title,
                      xlabel='Episode',
                      ylabel=label,
                      legend_dict=legend_dict,
                      legend_size=legend_size,
                      #legend_size=None,
                      linestyle_dict=linestyle_dict,
                      axis_size=axis_size if axis_size is not None else 24,
                      title_size=20,
                      #title_size=20,
                      plot_name=plot_name, )


def generate_plots():
    axis_size = None
    save = True
    save_msg = ''
    modes = ['train']
    plot_mode = 'all-methods'

    # env_id = 'WGW-v0'
    # max_episodes = 100  # 100
    # average_num = 10
    # max_reward = 1
    # min_reward = 0
    # plot_key = ['reward', 'reward_nc', 'constraint', 'reward_valid']
    # label_key = ['reward', 'reward_nc', 'Constraint Violation Rate', 'reward_valid']
    # label_key = [None, None, None, None]
    # img_size = None
    # save = True
    # save_msg = '_setting-1-noise-1e-2'
    # title = 'Girdworld-1 with Prob0.99'
    # # save_msg = 'test'
    # # title = 'test'
    # constraint_keys = ['constraint']
    # plot_y_lim_dict = {'reward': None,
    #                    'reward_nc': None,
    #                    'constraint': None,
    #                    'reward_valid': None,
    #                    }
    #
    # method_names_labels_dict = {

    #     'BC2L-setting1-noise1e-2': 'BC2L',
    #     'GAIL-setting1-noise1e-2': 'GACL',
    #     'ICRL-setting1-noise1e-2':'ICRL',
    #     'VICRL-setting1-noise1e-2': 'VICRL',
    #     'GICRL-setting1-noise1e-2': 'UAICRL',

    # }

    env_id = 'HCWithPos-v0'
    max_episodes = 6000
    average_num = 300
    max_reward = float('inf')
    min_reward = 0
    plot_key = ['reward', 'reward_nc', 'constraint', 'reward_valid']
    label_key = ['reward', 'reward_nc', 'Constraint Violation Rate', 'Feasible Rewards']
    #label_key = [None, None, None, None]
    save = True
    constraint_keys = ['constraint']
    plot_y_lim_dict = {'reward': None,
                       'reward_nc': None,
                       'constraint': None,
                       'reward_valid': None,
                       }
    img_size = (8, 8)
    noise = '1e-1'
    plot_mode = 'Noise-{0}'.format(noise)
    title = 'Blocked Half-Cheetah with Noise $\mathcal{N}(0,0.1)$'
    # title = 'Blocked Half-Cheetah'
    # plot_mode = 'Total'
    # title = 'Blocked Half-Cheetah with Stochastic Noise'

    method_names_labels_dict = {

        # "PPO_lag_td_HC_with-action_noise-1e-1": "PPO_Lag",
        # "PPO_dis_lag_Spline_exp_costadv_HC_with-action_noise-1e-1": 'DLPO-Neutral',
        # "PPO_dis_lag_Spline_CVaR_costadv_HC_with-action_noise-1e-1": 'DLPO-Averse,

        "BC2L_with-action_noise-1e-1-storebygame": 'BC2L',
        "GAIL_with-action_noise-1e-1-storebygame": 'GACL',
        "ICRL_with-action_noise-1e-1-storebygame": 'ICRL',
        "VICRL_with-action_noise-1e-1-storebygame": 'VICRL',
        "CICRL-HC-noise-1e-1": "UAICRL-NRS",
        "DICRL_Spline_CVaR_with-action_noise-1e-1-storebygame": 'UAICRL-NDA',
        "CDICRL_Spline_EXP_with-action_noise-1e-1-storebygame": 'UAICRL',
    }

    # env_id = 'AntWall-V0'
    # max_episodes = 10000 #12000
    # average_num = 600
    # # title = 'Blocked Ant'
    # max_reward = float('inf')
    # min_reward = 0
    # plot_key = ['reward', 'constraint', 'reward_valid', 'reward_nc']
    # # plot_key = ['reward', 'constraint', 'reward_valid']
    # #label_key = ['reward', 'Constraint Violation Rate', 'Feasible Rewards', 'reward_nc']
    # label_key = [None, None, None, None]
    # img_size = (8, 8)
    # save = True
    # noise = '1e-1'
    # plot_mode = 'Noise-{0}'.format(noise)
    # title = 'Blocked Ant with Noise $\mathcal{N}(0,0.1)$'
    # # title = 'Blocked Ant'
    # # plot_mode = 'Total'
    # # title = 'Blocked Ant with Stochastic Noise'
    # plot_y_lim_dict = {'reward': None,
    #                    'reward_nc': None,
    #                    'constraint': None,
    #                    'reward_valid': None}
    # constraint_keys = ['constraint']
    # method_names_labels_dict = {
    #
    #     "PPO-Lag-AntWall-noise-1e-1": 'PPO-Lag',
    #     "PPO-dis-Lag-Spline-EXP-AntWall-noise-1e-1": 'DLPO-Neutral',
    #     "PPO-dis-Lag-Spline-CVaR-AntWall-noise-1e-1": 'DLPO-Averse',
    #
    #     'BC2L-AntWall-noise-1e-1': 'BC2L',
    #     'GAIL-AntWall-noise-1e-1': 'GACL',
    #     'ICRL-AntWall-noise-1e-1': 'ICRL',
    #     'VICRL-AntWall-noise-1e-1': 'VICRL',
    #     'CICRL-AntWall-noise-1e-1': 'UAICRL-NRS',
    #     'DICRL-Spline-EXP-AntWall-noise-1e-1': 'UAICRL-NDA',
    #     'CDICRL-Spline-EXP-AntWall-noise-1e-1': 'UAICRL',
    #
    # }

    # env_id = 'InvertedPendulumWall-v0'
    # max_episodes = 60000
    # average_num = 1500
    # noise = '1e-1'
    # plot_mode = 'Noise-{0}'.format(noise)
    # title = 'Biased Pendulumn with Noise $\mathcal{N}(0,0.1)$'
    # # title = 'Biased Pendulum'
    # # plot_mode = 'Total'
    # # title = 'Biased Pendulum with Stochastic Noise'
    # max_reward = float('inf')
    # min_reward = 0
    # plot_key = ['reward', 'reward_nc', 'constraint', 'reward_valid']
    # label_key = ['reward', 'reward_nc', 'Constraint Violation Rate', 'reward_valid']
    # label_key = [None, None, None, None]
    # img_size = (8, 8)
    # save = True
    # plot_y_lim_dict = {'reward': None,
    #                    'reward_nc': None,
    #                    'constraint': None,
    #                    'reward_valid': None}
    # constraint_keys = ['constraint']
    # method_names_labels_dict = {
    #
    #     # 'PPO_lag_Pendulum_noise1e-1':'PPO-Lag',
    #     # 'PPO_dis_lag_Spline_EXP_Pendulum_noise1e-1': 'DLPO-Neutral',
    #     # 'PPO_dis_lag_Spline_CVaR_Pendulum_noise1e-1':'DLPO-Averse',
    #
    #     'BC2L_Pendulum-noise1e-1': 'BC2L',
    #     'GAIL_Pendulum-noise1e-1': 'GACL',
    #     'ICRL_Pendulum-noise1e-1': 'ICRL',
    #     'VICRL_Pendulum-noise1e-1': 'VICRL',
    #     'CICRL_Pendulum-noise1e-1': 'UAICRL-NRS',
    #     'DICRL_Spline_CVaR_Pendulum-noise1e-1': 'UAICRL-NDA',
    #     'CDICRL_Spline_CVaR_Pendulum-noise1e-1': 'UAICRL',
    # }

    # env_id = 'WalkerWithPos-v0'
    # max_episodes = 80000
    # average_num = 500
    # noise = '1e-1'
    # plot_mode = 'Noise-{0}'.format(noise)
    # title = 'Blocked Walker with Noise $\mathcal{N}(0,0.1)$'
    # # title = 'Blocked Walker'
    # # plot_mode = 'Total'
    # # title = 'Blocked Walker with Stochastic Noise'
    # max_reward = float('inf')
    # min_reward = 0
    # plot_key = ['reward', 'reward_nc', 'constraint', 'reward_valid']
    # #label_key = ['reward', 'reward_nc', 'Constraint Violation Rate', 'reward_valid']
    # label_key = [None, None, None, None]
    # # plot_y_lim_dict = {'reward': (0, 700),
    # #                    'reward_nc': (0, 700),
    # #                    'constraint': (0, 1)}
    # save = True
    # img_size = (8, 8)
    # plot_y_lim_dict = {'reward': None,
    #                    'reward_nc': None,
    #                    'constraint': None,
    #                    'reward_valid': None}
    # constraint_keys = ['constraint']
    # method_names_labels_dict = {
    #
    #     # 'PPO_lag_Walker_noise1e-1': 'PPO-Lag',
    #     # 'PPO_dis_lag_Spline_EXP_Walker_noise1e-1': 'DLPO-Neutral',
    #     # 'PPO_dis_lag_Spline_CVaR_Walker_noise1e-1': 'DLPO-Averse',
    #
    #     'BC2L_Walker_noise1e-1': 'BC2L',
    #     'GAIL_Walker_noise1e-1': 'GACL',
    #     'ICRL_Walker_noise1e-1':'ICRL',
    #     'VICRL_Walker_noise1e-1': 'VICRL',
    #     'CICRL_Walker_noise1e-1': 'UAICRL-NRS',
    #     'DICRL_Spline_EXP_Walker_noise1e-1':'UAICRL-NDA',
    #     'CDICRL_Spline_EXP_Walker_noise1e-1': 'UAICRL',
    # }

    # env_id = 'SwimmerWithPos-v0'
    # max_episodes = 10000
    # average_num = 1000 # 2500
    # noise = '1e-1'
    # plot_mode = 'Noise-{0}'.format(noise)
    # title = 'Blocked Swimmer with Noise $\mathcal{N}(0,0.1)$'
    # # title = 'Blocked Swimmer'
    # # plot_mode = 'Total'
    # # title = 'Blocked Swimmer with Stochastic Noise'
    # max_reward = float('inf')
    # min_reward = 0
    # plot_key = ['reward', 'reward_nc', 'constraint', 'reward_valid']
    # #label_key = ['reward', 'reward_nc', 'Constraint Violation Rate', 'Feasible Rewards']
    # label_key = [None, None, None, None]
    # img_size = (8, 8)
    # save = True
    # plot_y_lim_dict = {'reward': None,
    #                    'reward_nc': None,
    #                    'constraint': None,
    #                    'reward_valid': None}
    # constraint_keys = ['constraint']
    # method_names_labels_dict = {
    #
    #     'ppo_lag_SwmWithPos-v0-noise-1e-1':'PPO_Lag',
    #     'ppo_dis_lag_Spline_EXP_SwmWithPos-v0-noise-1e-1': 'DLPO-Neutral',
    #     'ppo_dis_lag_Spline_CVaR_SwmWithPos-v0-noise-1e-1':'DLPO-Averse',
    #
    #     'BC2L_SwmWithPos-v0-noise-1e-1': 'BC2L',
    #     'GAIL_SwmWithPos-v0-noise-1e-1': 'GACL',
    #     'ICRL_SwmWithPos-v0-noise-1e-1': 'ICRL',
    #     'VICRL_SwmWithPos-v0-noise-1e-1': 'VICRL',
    #     'CICRL_SwmWithPos-v0-noise-1e-1': 'UAICRL-NRS',
    #     'DICRL_Spline_EXP_SwmWithPos-v0-noise-1e-1': 'UAICRL-NDA',
    #     'CDICRL_Spline_EXP_SwmWithPos-v0-noise-1e-1': 'UAICRL',
    # }

   #  env_id = 'highD_velocity_constraint'
   #  max_episodes = 3500
   #  average_num = 100
   #  max_reward = 50
   #  min_reward = 0
   #  axis_size = 20
   #  img_size = [8.5, 6.5]
   #  save = True
   #  noise = '1e-1'
   #  plot_mode = 'Noise-{0}'.format(noise)
   #  title = 'HighD Velocity Constraint with Noise $\mathcal{N}(0,0.1)$'
   # # title = 'HighD Velocity Constraint'
   #  #plot_mode = 'Total'
   #  #title = 'HighD Velocity Constraint '
   #  constraint_keys = ['is_over_speed']
   #  plot_key = ['reward', 'reward_nc', 'reward_valid', 'is_collision', 'is_off_road',
   #              'is_goal_reached', 'is_time_out', 'avg_velocity', 'is_over_speed', 'success_rate']
   #  label_key = ['Rewards', 'Feasible Rewards', 'Feasible Rewards', 'Collision Rate', 'Off Road Rate',
   #               'Goal Reached Rate', 'Time Out Rate', 'Avg. Velocity', 'Speed Constraint Violation Rate',
   #               'Success Rate']
   #  plot_y_lim_dict = {'reward': None,
   #                     'reward_nc': None,
   #                     'reward_valid': None,
   #                     'is_collision': None,
   #                     'is_off_road': None,
   #                     'is_goal_reached': None,
   #                     'is_time_out': None,
   #                     'avg_velocity': None,
   #                     'is_over_speed': None,
   #                     'success_rate': None}
   #  bound_results = {
   #      'reward': 50,
   #      'reward_nc': 50,
   #      'reward_valid': 50,
   #      'is_collision': 0,
   #      'is_off_road': 0,
   #      'is_goal_reached': 0,
   #      'is_time_out': 0,
   #      'is_over_speed': 0,
   #      'success_rate': 1,
   #  }
   #  method_names_labels_dict = {
   #      "BC2L-vel-noise-1e-1": 'BC2L',
   #      "GAIL-vel-noise-1e-1": 'GAIL',
   #      "ICLR-vel-noise-1e-1": 'ICRL',
   #      "VICRL-vel-noise-1e-1": 'VICRL',
   #      "CDICLR-CVaR-QRDQN-vel-noise-1e-1": 'UAICRL',
   #  }

    if plot_mode == 'part':
        for method_name in method_names_labels_dict.copy().keys():
            if 'PPO' not in method_names_labels_dict[method_name]:
                del method_names_labels_dict[method_name]
    else:
        method_names_labels_dict = method_names_labels_dict

    linestyle_all = {
        'PPO-Lag': '-',
        "PI-Lag": '-' if plot_mode == 'part' else '--',
        "GACL": '-',
        "GAIL": '-',
        "BC2L": '-',
        "MECL": '-.',
        "ICRL": '-',
        "VICRL": "-",
        "UAICRL": '-',
        "UAICRL-NDA": '-',
        "UAICRL-NRS": '-',
        "DLPO-Neutral": '-',
        "DLPO-Averse": '-',
    }

    linestyle_dict = {}
    for method_name in method_names_labels_dict.keys():
        for linestyle_key in linestyle_all.keys():
            if method_names_labels_dict[method_name] == linestyle_key:
                linestyle_dict.update({method_name: linestyle_all[linestyle_key]})

    for mode in modes:
        # plot_key = ['reward', 'is_collision', 'is_off_road', 'is_goal_reached', 'is_time_out']
        log_path_dict = get_plot_results_dir(env_id)

        all_mean_dict = {}
        all_std_dict = {}
        all_episodes_dict = {}
        for method_name in method_names_labels_dict.keys():
            all_results = []
            # all_valid_rewards = []
            # all_valid_episodes = []
            if method_name == 'Bound':
                results = {}
                for key in bound_results:
                    results.update({key: [bound_results[key] for item in range(max_episodes + 1000)]})
                all_results.append(results)
            else:
                for log_path in log_path_dict[method_name]:
                    monitor_path_all = []
                    if mode == 'train':
                        run_files = os.listdir(log_path)
                        for file in run_files:
                            if 'monitor' in file:
                                monitor_path_all.append(log_path + file)
                    else:
                        monitor_path_all.append(log_path + 'test/test.monitor.csv')
                    if (method_names_labels_dict[method_name] == "PPO" or
                        method_names_labels_dict[method_name] == "PPO_lag" or
                        method_names_labels_dict[method_name] == "PI-Lag") and plot_mode != "part":
                        if 'reward_nc' in plot_key:
                            plot_key[plot_key.index('reward_nc')] = 'reward'
                    # rewards, is_collision, is_off_road, is_goal_reached, is_time_out = read_running_logs(log_path=log_path)
                    results, valid_rewards, valid_episodes = read_running_logs(monitor_path_all=monitor_path_all,
                                                                               read_keys=plot_key,
                                                                               max_reward=max_reward,
                                                                               min_reward=min_reward,
                                                                               max_episodes=max_episodes + float(
                                                                                   max_episodes / 5),
                                                                               constraint_keys=constraint_keys)
                    if (method_names_labels_dict[method_name] == "PPO" or
                        method_names_labels_dict[method_name] == "PPO_lag" or
                        method_names_labels_dict[method_name] == "PI-Lag") and plot_mode != "part":
                        results_copy_ = copy.copy(results)
                        for key in results.keys():
                            fill_value = np.mean(results_copy_[key][-100:])
                            results[key] = [fill_value for item in range(max_episodes + 1000)]
                    # all_valid_rewards.append(valid_rewards)
                    # all_valid_episodes.append(valid_episodes)
                    all_results.append(results)
            if mode == 'test':
                mean_std_test_results(all_results, method_name)

            mean_dict, std_dict, episodes = mean_std_plot_results(all_results)
            # mean_valid_rewards, std_valid_rewards, valid_episodes = \
            #     mean_std_plot_valid_rewards(all_valid_rewards, all_valid_episodes)
            # mean_dict.update({'reward_valid': mean_valid_rewards})
            # std_dict.update({'reward_valid': std_valid_rewards})
            # episodes.update({'reward_valid': valid_episodes})
            all_mean_dict.update({method_name: {}})
            all_std_dict.update({method_name: {}})
            all_episodes_dict.update({method_name: {}})

            if not os.path.exists(os.path.join('./plot_results/', env_id)):
                os.mkdir(os.path.join('./plot_results/', env_id))
            if not os.path.exists(os.path.join('./plot_results/', env_id, method_name)):
                os.mkdir(os.path.join('./plot_results/', env_id, method_name))

            for idx in range(len(plot_key)):
                print(method_name, plot_key[idx])
                if method_name == 'Bound' and (plot_key[idx] == 'avg_distance' or plot_key[idx] == 'avg_velocity'):
                    continue
                mean_results_moving_average = compute_moving_average(result_all=mean_dict[plot_key[idx]],
                                                                     average_num=average_num)
                std_results_moving_average = compute_moving_average(result_all=std_dict[plot_key[idx]],
                                                                    average_num=average_num)
                episode_plot = episodes[plot_key[idx]][:len(mean_results_moving_average)]
                if max_episodes:
                    mean_results_moving_average = mean_results_moving_average[:max_episodes]
                    std_results_moving_average = std_results_moving_average[:max_episodes]
                    episode_plot = episode_plot[:max_episodes]
                all_mean_dict[method_name].update({plot_key[idx]: mean_results_moving_average})
                if (method_names_labels_dict[method_name] == "PPO" or
                    method_names_labels_dict[method_name] == "PPO_lag") and plot_mode != "part":
                    all_std_dict[method_name].update({plot_key[idx]: np.zeros(std_results_moving_average.shape)})
                else:
                    all_std_dict[method_name].update({plot_key[idx]: std_results_moving_average / 2})
                all_episodes_dict[method_name].update({plot_key[idx]: episode_plot})
                plot_results(mean_results_moving_avg_dict={method_name: mean_results_moving_average},
                             std_results_moving_avg_dict={method_name: std_results_moving_average},
                             episode_plots={method_name: episode_plot},
                             label=plot_key[idx],
                             method_names=[method_name],
                             ylim=plot_y_lim_dict[plot_key[idx]],
                             save_label=os.path.join(env_id, method_name, plot_key[idx] + '_' + mode),
                             title=title,
                             axis_size=axis_size,
                             img_size=img_size,
                             linestyle_dict=linestyle_dict,
                             )
        for idx in range(len(plot_key)):
            mean_results_moving_avg_dict = {}
            std_results_moving_avg_dict = {}
            espisode_dict = {}
            plot_method_names = list(method_names_labels_dict.keys())
            for method_name in method_names_labels_dict.keys():
                if method_name == 'Bound' and (plot_key[idx] == 'avg_distance' or plot_key[idx] == 'avg_velocity'):
                    plot_method_names.remove('Bound')
                    continue
                mean_results_moving_avg_dict.update({method_name: all_mean_dict[method_name][plot_key[idx]]})
                std_results_moving_avg_dict.update({method_name: all_std_dict[method_name][plot_key[idx]]})
                espisode_dict.update({method_name: all_episodes_dict[method_name][plot_key[idx]]})
                # if (plot_key[idx] == 'reward_valid' or plot_key[idx] == 'constraint') and mode == 'test':
                #     print(method_name, plot_key[idx],
                #           all_mean_dict[method_name][plot_key[idx]][-1],
                #           all_std_dict[method_name][plot_key[idx]][-1])
                print(plot_key[idx],
                      method_name,
                      np.mean(mean_results_moving_avg_dict[method_name][-100:]),
                      np.mean(std_results_moving_avg_dict[method_name][-100:]))
            if save:
                save_label = os.path.join(env_id,
                                          plot_key[idx] + '_' + mode + save_msg + '_' + env_id + '_' + plot_mode)
            else:
                save_label = None

            plot_results(mean_results_moving_avg_dict=mean_results_moving_avg_dict,
                         std_results_moving_avg_dict=std_results_moving_avg_dict,
                         episode_plots=espisode_dict,
                         label=label_key[idx],
                         method_names=plot_method_names,
                         ylim=plot_y_lim_dict[plot_key[idx]],
                         save_label=save_label,
                         # legend_size=18,
                         legend_dict=method_names_labels_dict,
                         title=title,
                         axis_size=axis_size,
                         img_size=img_size,
                         linestyle_dict=linestyle_dict,
                         )


if __name__ == "__main__":
    generate_plots()
