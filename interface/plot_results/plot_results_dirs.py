def get_plot_results_dir(env_id):
    if env_id == 'WGW-v0':
        log_path_dict = {
            'ICRL-setting1-noise1e-2': [
                '../save_model/ICRL-WallGrid/ICRL-WGW-S1-1e-2/',
            ],
            'ICRL-setting1-noise1e-3': [
                '../save_model/ICRL-WallGrid/ICRL-WGW-S1-1e-3/',
            ],
            'VICRL-setting1-noise1e-2': [
                '../save_model/VICRL-WallGrid/VICRL-WGW-S1-1e-2/',
            ],
            'VICRL-setting1-noise1e-3': [
                '../save_model/VICRL-WallGrid/VICRL-WGW-S1-1e-3/',
            ],

            'BC2L-setting1-noise1e-2': [
                '../save_model/Binary-WallGrid/BC2L-WGW-S1-1e-2/',
            ],
            'BC2L-setting1-noise1e-3': [
                '../save_model/Binary-WallGrid/BC2L-WGW-S1-1e-3/',
            ],

            'GAIL-setting1-noise1e-2': [
                '../save_model/GAIL-WallGrid/GAIL-WGW-S1-1e-2/',
            ],
            'GAIL-setting1-noise1e-3': [
                '../save_model/GAIL-WallGrid/GAIL-WGW-S1-1e-3/',
            ],


            'DICRL-setting1-noise1e-2': [
                '../save_model/ICRL-WallGrid/DICRL-QRDQN-WGW-S1-1e-2/',
            ],
            'DICRL-setting1-noise1e-3': [
                '../save_model/ICRL-WallGrid/DICRL-QRDQN-WGW-S1-1e-3/',
            ],

            'GICRL-setting1-noise1e-2': [
                '../save_model/ICRL-WallGrid/GICRL-WGW-S1-1e-2/',
            ],
            'GICRL-setting1-noise1e-3': [
                '../save_model/ICRL-WallGrid/GICRL-WGW-S1-1e-3/',
            ],

            'BC2L-setting2-noise1e-2': [
                '../save_model/Binary-WallGrid/BC2L-WGW-S2-1e-2/',
            ],
            'BC2L-setting2-noise1e-3': [
                '../save_model/Binary-WallGrid/BC2L-WGW-S2-1e-3/',
            ],

            'GAIL-setting2-noise1e-2': [
                '../save_model/GAIL-WallGrid/GAIL-WGW-S2-1e-2/',
            ],
            'GAIL-setting2-noise1e-3': [
                '../save_model/GAIL-WallGrid/GAIL-WGW-S2-1e-3/',
            ],

            'ICRL-setting2-noise1e-2': [
                '../save_model/ICRL-WallGrid/ICRL-WGW-S2-1e-2/',
            ],
            'ICRL-setting2-noise1e-3': [
                '../save_model/ICRL-WallGrid/ICRL-WGW-S2-1e-3/',
            ],
            'VICRL-setting2-noise1e-2': [
                '../save_model/VICRL-WallGrid/VICRL-WGW-S2-1e-2/',
            ],
            'VICRL-setting2-noise1e-3': [
                '../save_model/VICRL-WallGrid/VICRL-WGW-S2-1e-3/',
            ],

            'DICRL-setting2-noise1e-2': [
                '../save_model/ICRL-WallGrid/DICRL-QRDQN-WGW-S2-1e-2/',
            ],
            'DICRL-setting2-noise1e-3': [
                '../save_model/ICRL-WallGrid/DICRL-QRDQN-WGW-S2-1e-3/',
            ],

            'GICRL-setting2-noise1e-2': [
                '../save_model/ICRL-WallGrid/GICRL-WGW-S2-1e-2/',
            ],
            'GICRL-setting2-noise1e-3': [
                '../save_model/ICRL-WallGrid/GICRL-WGW-S2-1e-3/',
            ],


            'ICRL-setting3-noise1e-2': [
                '../save_model/ICRL-WallGrid/ICRL-WGW-S3-1e-2/',
            ],
            'ICRL-setting3-noise1e-3': [
                '../save_model/ICRL-WallGrid/ICRL-WGW-S3-1e-3/',
            ],
            'VICRL-setting3-noise1e-2': [
                '../save_model/VICRL-WallGrid/VICRL-WGW-S3-1e-2/',
            ],
            'VICRL-setting3-noise1e-3': [
                '../save_model/VICRL-WallGrid/VICRL-WGW-S3-1e-3/',
            ],

            'BC2L-setting3-noise1e-2': [
                '../save_model/Binary-WallGrid/BC2L-WGW-S3-1e-2/',
            ],
            'BC2L-setting3-noise1e-3': [
                '../save_model/Binary-WallGrid/BC2L-WGW-S3-1e-3/',
            ],
            'GAIL-setting3-noise1e-2': [
                '../save_model/GAIL-WallGrid/GAIL-WGW-S3-1e-2/',
            ],
            'GAIL-setting3-noise1e-3': [
                '../save_model/GAIL-WallGrid/GAIL-WGW-S3-1e-3/',
            ],
            'DICRL-setting3-noise1e-2': [
                '../save_model/ICRL-WallGrid/DICRL-QRDQN-WGW-S3-1e-2/',
            ],
            'DICRL-setting3-noise1e-3': [
                '../save_model/ICRL-WallGrid/DICRL-QRDQN-WGW-S3-1e-3/',
            ],

            'GICRL-setting3-noise1e-2': [
                '../save_model/ICRL-WallGrid/GICRL-WGW-S3-1e-2/',
            ],
            'GICRL-setting3-noise1e-3': [
                '../save_model/ICRL-WallGrid/GICRL-WGW-S3-1e-3/',
            ],

        }
    elif env_id == 'HCWithPos-v0':
        log_path_dict = {
            "PPO_lag_td_HC_with-action_noise-1e-1": [
                '../save_model/4.8/train_ppo_lag_td_n50_HCWithPos-v0-1e-1-multi_env-Apr-08-2023-13:23-seed_123/',
                '../save_model/4.8/train_ppo_lag_td_n50_HCWithPos-v0-1e-1-multi_env-Apr-08-2023-13:23-seed_321/',
                '../save_model/4.8/train_ppo_lag_td_n50_HCWithPos-v0-1e-1-multi_env-Apr-08-2023-13:23-seed_456/',
                '../save_model/4.8/train_ppo_lag_td_n50_HCWithPos-v0-1e-1-multi_env-Apr-08-2023-13:23-seed_654/',
            ],

            "PPO_dis_lag_Spline_exp_costadv_HC_with-action_noise-1e-1": [
                '../save_model/4.7/train_ppo_dis_lag_Spline_EXP_costadv_n50_HCWithPos-v0-noise-1e-1-multi_env-Apr-05-2023-10:20-seed_123/',
                '../save_model/4.7/train_ppo_dis_lag_Spline_EXP_costadv_n50_HCWithPos-v0-noise-1e-1-multi_env-Apr-05-2023-10:20-seed_321/',
                '../save_model/4.7/train_ppo_dis_lag_Spline_EXP_costadv_n50_HCWithPos-v0-noise-1e-1-multi_env-Apr-06-2023-22:30-seed_456/',
                '../save_model/4.7/train_ppo_dis_lag_Spline_EXP_costadv_n50_HCWithPos-v0-noise-1e-1-multi_env-Apr-06-2023-22:32-seed_654/',
            ],

            "PPO_dis_lag_Spline_CVaR_costadv_HC_with-action_noise-1e-1": [
                '../save_model/4.7/train_ppo_dis_lag_Spline_CVaR_costadv_n50_HCWithPos-v0-noise-1e-1-multi_env-Apr-05-2023-10:19-seed_123/',
                '../save_model/4.7/train_ppo_dis_lag_Spline_CVaR_costadv_n50_HCWithPos-v0-noise-1e-1-multi_env-Apr-05-2023-10:19-seed_321/',
                '../save_model/4.7/train_ppo_dis_lag_Spline_CVaR_costadv_n50_HCWithPos-v0-noise-1e-1-multi_env-Apr-06-2023-22:33-seed_654/',
                '../save_model/4.7/train_ppo_dis_lag_Spline_CVaR_costadv_n50_HCWithPos-v0-noise-1e-1-multi_env-Apr-06-2023-22:34-seed_456/',
            ],

            "BC2L_HC_with-action_noise-1e-1": [
                '../save_model/Binary-HC/train_BC2L_HCWithPos-v0_noise-1e-1-multi_env-May-09-2023-00:15-seed_123/',
                '../save_model/Binary-HC/train_BC2L_HCWithPos-v0_noise-1e-1-multi_env-May-09-2023-00:15-seed_321/',
                '../save_model/Binary-HC/train_BC2L_HCWithPos-v0_noise-1e-1-multi_env-May-09-2023-00:15-seed_456/',
                '../save_model/Binary-HC/train_BC2L_HCWithPos-v0_noise-1e-1-multi_env-May-09-2023-00:15-seed_654/',
            ],

            "GAIL_with-action_noise-1e-1": [
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_noise-1e-1-multi_env-May-09-2023-19:14-seed_123/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_noise-1e-1-multi_env-May-09-2023-19:14-seed_321/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_noise-1e-1-multi_env-May-09-2023-19:14-seed_456/',
                '../save_model/GAIL-HC/train_GAIL_HCWithPos-v0_noise-1e-1-multi_env-May-09-2023-19:14-seed_654/',
            ],

            "VICRL_HC_with-action_noise-1e-1": [
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_noise-1e-1-multi_env-May-09-2023-19:18-seed_123/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_noise-1e-1-multi_env-May-09-2023-19:18-seed_321/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_noise-1e-1-multi_env-May-09-2023-19:18-seed_456/',
                '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_noise-1e-1-multi_env-May-09-2023-19:18-seed_654/',

            ],

            "ICRL_HC_with-action_noise-1e-1": [
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_noise-1e-1-notsbg-multi_env-Apr-15-2023-20:15-seed_123/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_noise-1e-1-notsbg-multi_env-Apr-15-2023-20:15-seed_321/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_noise-1e-1-notsbg-multi_env-Apr-15-2023-20:15-seed_456/',
                '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_noise-1e-1-notsbg-multi_env-Apr-15-2023-20:15-seed_654/',
            ],

            "DICRL_Spline_CVaR_HC_with-action_noise-1e-1": [
                '../save_model/ICRL-HC/train_DICRL_Spline_CVaR_HCWithPos-v0_noise-1e-1-notsbg-multi_env-Apr-15-2023-20:15-seed_123/',
                '../save_model/ICRL-HC/train_DICRL_Spline_CVaR_HCWithPos-v0_noise-1e-1-notsbg-multi_env-Apr-15-2023-20:15-seed_321/',
                '../save_model/ICRL-HC/train_DICRL_Spline_CVaR_HCWithPos-v0_noise-1e-1-notsbg-multi_env-Apr-15-2023-20:15-seed_456/',
                '../save_model/ICRL-HC/train_DICRL_Spline_CVaR_HCWithPos-v0_noise-1e-1-notsbg-multi_env-Apr-15-2023-20:15-seed_654/',
            ],

            "DICRL_Spline_EXP_HC_with-action_noise-1e-1": [
                '../save_model/ICRL-HC/train_DICRL_Spline_EXP_HCWithPos-v0_noise-1e-1-notsbg-multi_env-Apr-15-2023-20:15-seed_123/',
                '../save_model/ICRL-HC/train_DICRL_Spline_EXP_HCWithPos-v0_noise-1e-1-notsbg-multi_env-Apr-15-2023-20:15-seed_321/',
                '../save_model/ICRL-HC/train_DICRL_Spline_EXP_HCWithPos-v0_noise-1e-1-notsbg-multi_env-Apr-15-2023-20:15-seed_456/',
                '../save_model/ICRL-HC/train_DICRL_Spline_EXP_HCWithPos-v0_noise-1e-1-notsbg-multi_env-Apr-15-2023-20:15-seed_654/',
            ],

            "CDICRL_Spline_EXP_HC_with-action_noise-1e-1": [
                '../save_model/ICRL-HC/CDICRL/train_ContinuesDICRL_Spline_EXP_HCWithPos-v0_noise-1e-1-notsbg-multi_env-Apr-19-2023-10:03-seed_123/',
                '../save_model/ICRL-HC/CDICRL/train_ContinuesDICRL_Spline_EXP_HCWithPos-v0_noise-1e-1-notsbg-multi_env-Apr-19-2023-10:03-seed_321/',
                '../save_model/ICRL-HC/CDICRL/train_ContinuesDICRL_Spline_EXP_HCWithPos-v0_noise-1e-1-notsbg-multi_env-Apr-19-2023-10:03-seed_456/',
                '../save_model/ICRL-HC/CDICRL/train_ContinuesDICRL_Spline_EXP_HCWithPos-v0_noise-1e-1-notsbg-multi_env-Apr-19-2023-10:03-seed_654/',
            ],

            "CDICRL_Spline_CVaR_HC_with-action_noise-1e-1": [
                '../save_model/ICRL-HC/CDICRL/train_ContinuesDICRL_Spline_CVaR_HCWithPos-v0_noise-1e-1-notsbg-multi_env-Apr-19-2023-16:14-seed_123/',
                '../save_model/ICRL-HC/CDICRL/train_ContinuesDICRL_Spline_CVaR_HCWithPos-v0_noise-1e-1-notsbg-multi_env-Apr-19-2023-16:14-seed_321/',
                '../save_model/ICRL-HC/CDICRL/train_ContinuesDICRL_Spline_CVaR_HCWithPos-v0_noise-1e-1-notsbg-multi_env-Apr-19-2023-16:14-seed_456/',
                '../save_model/ICRL-HC/CDICRL/train_ContinuesDICRL_Spline_CVaR_HCWithPos-v0_noise-1e-1-notsbg-multi_env-Apr-19-2023-16:14-seed_654/',
            ],
        }
    elif env_id == 'AntWall-V0':
        log_path_dict = {
            'PPO-Lag-AntWall-noise-1e-1': [
                '../save_model/PPO-Lag-AntWall/train_ppo_lag_n30_AntWall-v0-noise1e-1-multi_env-Apr-08-2023-20:19-seed_123/',
                '../save_model/PPO-Lag-AntWall/train_ppo_lag_n30_AntWall-v0-noise1e-1-multi_env-Apr-08-2023-20:19-seed_321/',
                '../save_model/PPO-Lag-AntWall/train_ppo_lag_n30_AntWall-v0-noise1e-1-multi_env-Apr-08-2023-20:19-seed_456/',
                '../save_model/PPO-Lag-AntWall/train_ppo_lag_n30_AntWall-v0-noise1e-1-multi_env-Apr-08-2023-20:19-seed_654/',
            ],

            'PPO-dis-Lag-Spline-CVaR-AntWall-noise-1e-1': [
                '../save_model/PPO-Lag-AntWall/hyper-new/train_ppo_dis_lag_n30_Spline_CVaR_hyper9_AntWall-v0-noise1e-1-multi_env-Apr-22-2023-20:30-seed_123/',
                '../save_model/PPO-Lag-AntWall/hyper-new/train_ppo_dis_lag_n30_Spline_CVaR_hyper9_AntWall-v0-noise1e-1-multi_env-Apr-22-2023-20:30-seed_321/',
                '../save_model/PPO-Lag-AntWall/hyper-new/train_ppo_dis_lag_n30_Spline_CVaR_hyper9_AntWall-v0-noise1e-1-multi_env-Apr-22-2023-20:30-seed_456/',
                '../save_model/PPO-Lag-AntWall/hyper-new/train_ppo_dis_lag_n30_Spline_CVaR_hyper9_AntWall-v0-noise1e-1-multi_env-Apr-22-2023-20:30-seed_654/',
            ],

            'PPO-dis-Lag-Spline-EXP-AntWall-noise-1e-1': [
                '../save_model/PPO-Lag-AntWall/hyper-new/train_ppo_dis_lag_n30_Spline_EXP_hyper9_AntWall-v0-noise1e-1-multi_env-Apr-22-2023-13:55-seed_123/',
                '../save_model/PPO-Lag-AntWall/hyper-new/train_ppo_dis_lag_n30_Spline_EXP_hyper9_AntWall-v0-noise1e-1-multi_env-Apr-22-2023-13:55-seed_321/',
                '../save_model/PPO-Lag-AntWall/hyper-new/train_ppo_dis_lag_n30_Spline_EXP_hyper9_AntWall-v0-noise1e-1-multi_env-Apr-22-2023-13:55-seed_456/',
                '../save_model/PPO-Lag-AntWall/hyper-new/train_ppo_dis_lag_n30_Spline_EXP_hyper9_AntWall-v0-noise1e-1-multi_env-Apr-22-2023-13:55-seed_654/',
            ],

            'BC2L-AntWall-noise-1e-1': [
                '../save_model/Binary-AntWall/train_BC2L_AntWall-v0_noise-1e-1-multi_env-May-09-2023-00:15-seed_123/',
                '../save_model/Binary-AntWall/train_BC2L_AntWall-v0_noise-1e-1-multi_env-May-09-2023-00:15-seed_321/',
                '../save_model/Binary-AntWall/train_BC2L_AntWall-v0_noise-1e-1-multi_env-May-09-2023-00:15-seed_456/',
                '../save_model/Binary-AntWall/train_BC2L_AntWall-v0_noise-1e-1-multi_env-May-09-2023-00:15-seed_654/',
            ],

            'GAIL-AntWall-noise-1e-1': [
                '../save_model/GAIL-AntWall/train_GAIL_AntWall-v0_noise-1e-1-multi_env-May-08-2023-14:20-seed_123/',
                '../save_model/GAIL-AntWall/train_GAIL_AntWall-v0_noise-1e-1-multi_env-May-08-2023-14:20-seed_321/',
                '../save_model/GAIL-AntWall/train_GAIL_AntWall-v0_noise-1e-1-multi_env-May-08-2023-14:20-seed_456/',
                '../save_model/GAIL-AntWall/train_GAIL_AntWall-v0_noise-1e-1-multi_env-May-08-2023-14:20-seed_654/',
            ],

            'VICRL-AntWall-noise-1e-1': [
                '../save_model/VICRL-AntWall/train_VICRL_AntWall-v0_noise-1e-1-multi_env-May-09-2023-19:18-seed_123/',
                '../save_model/VICRL-AntWall/train_VICRL_AntWall-v0_noise-1e-1-multi_env-May-09-2023-19:18-seed_321/',
                '../save_model/VICRL-AntWall/train_VICRL_AntWall-v0_noise-1e-1-multi_env-May-09-2023-19:18-seed_456/',
                '../save_model/VICRL-AntWall/train_VICRL_AntWall-v0_noise-1e-1-multi_env-May-09-2023-19:18-seed_654/',
            ],

            'ICRL-AntWall-noise-1e-1': [
                '../save_model/ICRL-AntWall/train_ICRL_AntWall-v0-noise1e-1-multi_env-Apr-19-2023-21:17-seed_123/',
                '../save_model/ICRL-AntWall/train_ICRL_AntWall-v0-noise1e-1-multi_env-Apr-19-2023-21:17-seed_321/',
                '../save_model/ICRL-AntWall/train_ICRL_AntWall-v0-noise1e-1-multi_env-Apr-19-2023-21:17-seed_456/',
                '../save_model/ICRL-AntWall/train_ICRL_AntWall-v0-noise1e-1-multi_env-Apr-19-2023-21:17-seed_654/',
            ],

            'DICRL-Spline-EXP-AntWall-noise-1e-1': [
                '../save_model/ICRL-AntWall/hyper/train_DICRL_Spline_EXP_hyper9_AntWall-v0-noise1e-1-multi_env-Apr-22-2023-20:36-seed_123/',
                '../save_model/ICRL-AntWall/hyper/train_DICRL_Spline_EXP_hyper9_AntWall-v0-noise1e-1-multi_env-Apr-22-2023-20:36-seed_321/',
                '../save_model/ICRL-AntWall/hyper/train_DICRL_Spline_EXP_hyper9_AntWall-v0-noise1e-1-multi_env-Apr-22-2023-21:07-seed_456/',
                '../save_model/ICRL-AntWall/hyper/train_DICRL_Spline_EXP_hyper9_AntWall-v0-noise1e-1-multi_env-Apr-22-2023-20:36-seed_654/',
            ],

            'DICRL-Spline-CVaR-AntWall-noise-1e-1': [
                '../save_model/ICRL-AntWall/hyper/train_DICRL_Spline_CVaR_hyper9_AntWall-v0-noise1e-1-multi_env-Apr-22-2023-20:36-seed_123/',
                '../save_model/ICRL-AntWall/hyper/train_DICRL_Spline_CVaR_hyper9_AntWall-v0-noise1e-1-multi_env-Apr-22-2023-20:36-seed_321/',
                '../save_model/ICRL-AntWall/hyper/train_DICRL_Spline_CVaR_hyper9_AntWall-v0-noise1e-1-multi_env-Apr-22-2023-20:36-seed_456/',
                '../save_model/ICRL-AntWall/hyper/train_DICRL_Spline_CVaR_hyper9_AntWall-v0-noise1e-1-multi_env-Apr-22-2023-20:36-seed_654/',
            ],

            'CDICRL-Spline-EXP-AntWall-noise-1e-1': [
                '../save_model/ICRL-AntWall/CDICRL/train_CDICRL_Spline_EXP_hyper9_AntWall-v0-noise1e-1-multi_env-Apr-23-2023-03:06-seed_123/',
                '../save_model/ICRL-AntWall/CDICRL/train_CDICRL_Spline_EXP_hyper9_AntWall-v0-noise1e-1-multi_env-Apr-23-2023-03:06-seed_321/',
                '../save_model/ICRL-AntWall/CDICRL/train_CDICRL_Spline_EXP_hyper9_AntWall-v0-noise1e-1-multi_env-Apr-22-2023-22:29-seed_456/',
                '../save_model/ICRL-AntWall/CDICRL/train_CDICRL_Spline_EXP_hyper9_AntWall-v0-noise1e-1-multi_env-Apr-23-2023-03:06-seed_654/',
            ],

            'CDICRL-Spline-CVaR-AntWall-noise-1e-1': [
                '../save_model/ICRL-AntWall/CDICRL/train_CDICRL_Spline_CVaR_hyper9_AntWall-v0-noise1e-1-multi_env-Apr-23-2023-03:06-seed_123/',
                '../save_model/ICRL-AntWall/CDICRL/train_CDICRL_Spline_CVaR_hyper9_AntWall-v0-noise1e-1-multi_env-Apr-23-2023-03:06-seed_321/',
                '../save_model/ICRL-AntWall/CDICRL/train_CDICRL_Spline_CVaR_hyper9_AntWall-v0-noise1e-1-multi_env-Apr-23-2023-03:06-seed_456/',
                '../save_model/ICRL-AntWall/CDICRL/train_CDICRL_Spline_CVaR_hyper9_AntWall-v0-noise1e-1-multi_env-Apr-23-2023-03:06-seed_654/',
            ],
        }
    elif env_id == 'InvertedPendulumWall-v0':
        log_path_dict = {
            'PPO_lag_Pendulum_noise1e-1': [
                '../save_model/PPO-Lag-InvertedPendulumWall/train_ppo_lag_InvertedPendulumWall-v0-noise-1e-1-multi_env-Apr-13-2023-13:52-seed_123/',
                '../save_model/PPO-Lag-InvertedPendulumWall/train_ppo_lag_InvertedPendulumWall-v0-noise-1e-1-multi_env-Apr-13-2023-13:52-seed_321/',
                '../save_model/PPO-Lag-InvertedPendulumWall/train_ppo_lag_InvertedPendulumWall-v0-noise-1e-1-multi_env-Apr-13-2023-13:52-seed_456/',
                '../save_model/PPO-Lag-InvertedPendulumWall/train_ppo_lag_InvertedPendulumWall-v0-noise-1e-1-multi_env-Apr-13-2023-13:52-seed_654/',
            ],

            'PPO_dis_lag_Spline_CVaR_Pendulum_noise1e-1': [
                '../save_model/PPO-Lag-InvertedPendulumWall/train_ppo_dis_lag_Spline_CVaR_InvertedPendulumWall-v0-noise-1e-1-multi_env-Apr-13-2023-13:52-seed_123/',
                '../save_model/PPO-Lag-InvertedPendulumWall/train_ppo_dis_lag_Spline_CVaR_InvertedPendulumWall-v0-noise-1e-1-multi_env-Apr-13-2023-13:52-seed_321/',
                '../save_model/PPO-Lag-InvertedPendulumWall/train_ppo_dis_lag_Spline_CVaR_InvertedPendulumWall-v0-noise-1e-1-multi_env-Apr-13-2023-13:52-seed_456/',
                '../save_model/PPO-Lag-InvertedPendulumWall/train_ppo_dis_lag_Spline_CVaR_InvertedPendulumWall-v0-noise-1e-1-multi_env-Apr-13-2023-13:52-seed_654/',
            ],

            'PPO_dis_lag_Spline_EXP_Pendulum_noise1e-1': [
                '../save_model/PPO-Lag-InvertedPendulumWall/train_ppo_dis_lag_Spline_EXP_InvertedPendulumWall-v0-noise-1e-1-multi_env-Apr-13-2023-13:52-seed_123/',
                '../save_model/PPO-Lag-InvertedPendulumWall/train_ppo_dis_lag_Spline_EXP_InvertedPendulumWall-v0-noise-1e-1-multi_env-Apr-13-2023-13:52-seed_321/',
                '../save_model/PPO-Lag-InvertedPendulumWall/train_ppo_dis_lag_Spline_EXP_InvertedPendulumWall-v0-noise-1e-1-multi_env-Apr-13-2023-13:52-seed_456/',
                '../save_model/PPO-Lag-InvertedPendulumWall/train_ppo_dis_lag_Spline_EXP_InvertedPendulumWall-v0-noise-1e-1-multi_env-Apr-13-2023-13:52-seed_654/',
            ],

            'BC2L_Pendulum-noise1e-1': [
                '../save_model/Binary-Inverted/train_BC2L_InvertedPendulumWall-v0_noise-1e-1-multi_env-May-09-2023-13:30-seed_123/',
                '../save_model/Binary-Inverted/train_BC2L_InvertedPendulumWall-v0_noise-1e-1-multi_env-May-09-2023-13:30-seed_321/',
                '../save_model/Binary-Inverted/train_BC2L_InvertedPendulumWall-v0_noise-1e-1-multi_env-May-09-2023-13:30-seed_456/',
                '../save_model/Binary-Inverted/train_BC2L_InvertedPendulumWall-v0_noise-1e-1-multi_env-May-09-2023-13:30-seed_654/',
            ],

            'GAIL_Pendulum-noise1e-1': [
                '../save_model/GAIL-Inverted/train_GAIL_InvertedPendulumWall-v0_noise-1e-1-multi_env-May-08-2023-19:12-seed_123/',
                '../save_model/GAIL-Inverted/train_GAIL_InvertedPendulumWall-v0_noise-1e-1-multi_env-May-08-2023-19:12-seed_321/',
                '../save_model/GAIL-Inverted/train_GAIL_InvertedPendulumWall-v0_noise-1e-1-multi_env-May-08-2023-19:12-seed_456/',
                '../save_model/GAIL-Inverted/train_GAIL_InvertedPendulumWall-v0_noise-1e-1-multi_env-May-08-2023-19:12-seed_654/',
            ],

            'VICRL_Pendulum-noise1e-1': [
                '../save_model/VICRL-Inverted/train_VICRL_InvertedPendulumWall-v0_noise-1e-1-multi_env-May-10-2023-01:05-seed_123/',
                '../save_model/VICRL-Inverted/train_VICRL_InvertedPendulumWall-v0_noise-1e-1-multi_env-May-10-2023-01:05-seed_321/',
                '../save_model/VICRL-Inverted/train_VICRL_InvertedPendulumWall-v0_noise-1e-1-multi_env-May-10-2023-01:05-seed_456/',
                '../save_model/VICRL-Inverted/train_VICRL_InvertedPendulumWall-v0_noise-1e-1-multi_env-May-10-2023-01:05-seed_654/',
            ],

            'ICRL_Pendulum-noise1e-1': [
                '../save_model/ICRL-Inverted/train_ICRL_InvertedPendulumWall-v0-noise1e-1-multi_env-Apr-24-2023-23:04-seed_123/',
                '../save_model/ICRL-Inverted/train_ICRL_InvertedPendulumWall-v0-noise1e-1-multi_env-Apr-24-2023-23:04-seed_321/',
                '../save_model/ICRL-Inverted/train_ICRL_InvertedPendulumWall-v0-noise1e-1-multi_env-Apr-24-2023-23:04-seed_456/',
                '../save_model/ICRL-Inverted/train_ICRL_InvertedPendulumWall-v0-noise1e-1-multi_env-Apr-24-2023-23:04-seed_654/',
            ],

            'DICRL_Spline_EXP_Pendulum-noise1e-1': [
                '../save_model/ICRL-Inverted/my/my-train_DICRL_Spline_EXP_InvertedPendulumWall-v0-noise1e-1-multi_env-Apr-26-2023-00:25-seed_123/',
                '../save_model/ICRL-Inverted/my/my-train_DICRL_Spline_EXP_InvertedPendulumWall-v0-noise1e-1-multi_env-Apr-26-2023-00:25-seed_321/',
                '../save_model/ICRL-Inverted/my/my-train_DICRL_Spline_EXP_InvertedPendulumWall-v0-noise1e-1-multi_env-Apr-26-2023-00:25-seed_456/',
                '../save_model/ICRL-Inverted/my/my-train_DICRL_Spline_EXP_InvertedPendulumWall-v0-noise1e-1-multi_env-Apr-26-2023-00:25-seed_654/',
            ],

            'CDICRL_Spline_EXP_Pendulum-noise1e-1': [
                '../save_model/ICRL-Inverted/cdicrl-my/my-train_CDICRL_Spline_EXP_InvertedPendulumWall-v0-noise1e-1-multi_env-Apr-26-2023-22:01-seed_123/',
                '../save_model/ICRL-Inverted/cdicrl-my/my-train_CDICRL_Spline_EXP_InvertedPendulumWall-v0-noise1e-1-multi_env-Apr-26-2023-22:01-seed_321/',
                '../save_model/ICRL-Inverted/cdicrl-my/my-train_CDICRL_Spline_EXP_InvertedPendulumWall-v0-noise1e-1-multi_env-Apr-26-2023-22:01-seed_456/',
                '../save_model/ICRL-Inverted/cdicrl-my/my-train_CDICRL_Spline_EXP_InvertedPendulumWall-v0-noise1e-1-multi_env-Apr-26-2023-22:01-seed_654/',
            ],

            'DICRL_Spline_CVaR_Pendulum-noise1e-1': [
                '../save_model/ICRL-Inverted/my/my-train_DICRL_Spline_CVaR_InvertedPendulumWall-v0-noise1e-1-multi_env-Apr-26-2023-00:25-seed_123/',
                '../save_model/ICRL-Inverted/my/my-train_DICRL_Spline_CVaR_InvertedPendulumWall-v0-noise1e-1-multi_env-Apr-26-2023-00:25-seed_321/',
                '../save_model/ICRL-Inverted/my/my-train_DICRL_Spline_CVaR_InvertedPendulumWall-v0-noise1e-1-multi_env-Apr-26-2023-00:25-seed_456/',
                '../save_model/ICRL-Inverted/my/my-train_DICRL_Spline_CVaR_InvertedPendulumWall-v0-noise1e-1-multi_env-Apr-26-2023-00:25-seed_654/',
            ],

            'CDICRL_Spline_CVaR_Pendulum-noise1e-1': [
                '../save_model/ICRL-Inverted/cdicrl-my/my-train_CDICRL_Spline_CVaR_InvertedPendulumWall-v0-noise1e-1-multi_env-Apr-26-2023-22:01-seed_123/',
                '../save_model/ICRL-Inverted/cdicrl-my/my-train_CDICRL_Spline_CVaR_InvertedPendulumWall-v0-noise1e-1-multi_env-Apr-26-2023-22:01-seed_321/',
                '../save_model/ICRL-Inverted/cdicrl-my/my-train_CDICRL_Spline_CVaR_InvertedPendulumWall-v0-noise1e-1-multi_env-Apr-26-2023-22:01-seed_456/',
                '../save_model/ICRL-Inverted/cdicrl-my/my-train_CDICRL_Spline_CVaR_InvertedPendulumWall-v0-noise1e-1-multi_env-Apr-26-2023-22:01-seed_654/',
            ],

        }
    elif env_id == 'WalkerWithPos-v0':
        log_path_dict = {
            'PPO_lag_Walker_noise1e-1': [
                '../save_model/PPO-Lag-Walker/train_ppo_lag_WalkerWithPos-v0-noise-1e-1-multi_env-Apr-12-2023-12:31-seed_123/',
                '../save_model/PPO-Lag-Walker/train_ppo_lag_WalkerWithPos-v0-noise-1e-1-multi_env-Apr-12-2023-12:31-seed_321/',
                '../save_model/PPO-Lag-Walker/train_ppo_lag_WalkerWithPos-v0-noise-1e-1-multi_env-Apr-12-2023-12:31-seed_456/',
                '../save_model/PPO-Lag-Walker/train_ppo_lag_WalkerWithPos-v0-noise-1e-1-multi_env-Apr-12-2023-12:31-seed_654/',
            ],

            'PPO_dis_lag_Spline_CVaR_Walker_noise1e-1': [
                '../save_model/PPO-Lag-Walker/train_ppo_dis_lag_Spline-CVaR-WalkerWithPos-v0-noise-1e-1-multi_env-Apr-12-2023-18:58-seed_123/',
                '../save_model/PPO-Lag-Walker/train_ppo_dis_lag_Spline-CVaR-WalkerWithPos-v0-noise-1e-1-multi_env-Apr-12-2023-18:58-seed_321/',
                '../save_model/PPO-Lag-Walker/train_ppo_dis_lag_Spline-CVaR-WalkerWithPos-v0-noise-1e-1-multi_env-Apr-12-2023-18:58-seed_456/',
                '../save_model/PPO-Lag-Walker/train_ppo_dis_lag_Spline-CVaR-WalkerWithPos-v0-noise-1e-1-multi_env-Apr-12-2023-18:58-seed_654/',
            ],

            'PPO_dis_lag_Spline_EXP_Walker_noise1e-1': [
                '../save_model/PPO-Lag-Walker/train_ppo_dis_lag_Spline-EXP-WalkerWithPos-v0-noise-1e-1-multi_env-Apr-12-2023-18:58-seed_123/',
                '../save_model/PPO-Lag-Walker/train_ppo_dis_lag_Spline-EXP-WalkerWithPos-v0-noise-1e-1-multi_env-Apr-12-2023-18:58-seed_321/',
                '../save_model/PPO-Lag-Walker/train_ppo_dis_lag_Spline-EXP-WalkerWithPos-v0-noise-1e-1-multi_env-Apr-12-2023-18:58-seed_456/',
                '../save_model/PPO-Lag-Walker/train_ppo_dis_lag_Spline-EXP-WalkerWithPos-v0-noise-1e-1-multi_env-Apr-12-2023-18:59-seed_654/',
            ],

            'BC2L_Walker_noise1e-1': [
                '../save_model/Binary-Walker/train_BC2L_WalkerWithPos-v0_noise-1e-1-multi_env-May-09-2023-00:25-seed_123/',
                '../save_model/Binary-Walker/train_BC2L_WalkerWithPos-v0_noise-1e-1-multi_env-May-09-2023-00:25-seed_321/',
                '../save_model/Binary-Walker/train_BC2L_WalkerWithPos-v0_noise-1e-1-multi_env-May-09-2023-00:25-seed_456/',
                '../save_model/Binary-Walker/train_BC2L_WalkerWithPos-v0_noise-1e-1-multi_env-May-09-2023-00:25-seed_654/',
            ],

            'GAIL_Walker_noise1e-1': [
                '../save_model/GAIL-Walker/train_GAIL_WalkerWithPos-v0_noise-1e-1-multi_env-May-08-2023-19:12-seed_123/',
                '../save_model/GAIL-Walker/train_GAIL_WalkerWithPos-v0_noise-1e-1-multi_env-May-08-2023-19:12-seed_321/',
                '../save_model/GAIL-Walker/train_GAIL_WalkerWithPos-v0_noise-1e-1-multi_env-May-08-2023-19:12-seed_456/',
                '../save_model/GAIL-Walker/train_GAIL_WalkerWithPos-v0_noise-1e-1-multi_env-May-08-2023-19:12-seed_654/',
            ],

            'VICRL_Walker_noise1e-1': [
                '../save_model/VICRL-Walker/train_VICRL_WalkerWithPos-v0_noise-1e-1-multi_env-May-10-2023-01:05-seed_123/',
                '../save_model/VICRL-Walker/train_VICRL_WalkerWithPos-v0_noise-1e-1-multi_env-May-10-2023-01:05-seed_321/',
                '../save_model/VICRL-Walker/train_VICRL_WalkerWithPos-v0_noise-1e-1-multi_env-May-10-2023-01:05-seed_456/',
                '../save_model/VICRL-Walker/train_VICRL_WalkerWithPos-v0_noise-1e-1-multi_env-May-10-2023-01:05-seed_654/',
            ],

            'ICRL_Walker_noise1e-1': [
                '../save_model/ICRL-Walker/test/icrl-WalkerWithPos-v0-noise-1e-1-test-multi_env-Apr-22-2023-14:43-seed_123/',
                '../save_model/ICRL-Walker/test/icrl-WalkerWithPos-v0-noise-1e-1-test-multi_env-Apr-22-2023-14:43-seed_321/',
                '../save_model/ICRL-Walker/test/icrl-WalkerWithPos-v0-noise-1e-1-test-multi_env-Apr-22-2023-14:43-seed_456/',
                '../save_model/ICRL-Walker/test/icrl-WalkerWithPos-v0-noise-1e-1-test-multi_env-Apr-22-2023-14:43-seed_654/',
            ],

            'DICRL_Spline_EXP_Walker_noise1e-1': [
                '../save_model/ICRL-Walker/test/dicrl-spline-exp-WalkerWithPos-v0-noise-1e-1-test-multi_env-Apr-23-2023-15:42-seed_123/',
                '../save_model/ICRL-Walker/test/dicrl-spline-exp-WalkerWithPos-v0-noise-1e-1-test-multi_env-Apr-23-2023-15:42-seed_321/',
                '../save_model/ICRL-Walker/test/dicrl-spline-exp-WalkerWithPos-v0-noise-1e-1-test-multi_env-Apr-23-2023-15:42-seed_456/',
                '../save_model/ICRL-Walker/test/dicrl-spline-exp-WalkerWithPos-v0-noise-1e-1-test-multi_env-Apr-23-2023-15:42-seed_654/',
            ],

            'DICRL_Spline_CVaR_Walker_noise1e-1': [
                '../save_model/ICRL-Walker/test/dicrl-spline-cvar-WalkerWithPos-v0-noise-1e-1-test-multi_env-Apr-24-2023-20:37-seed_123/',
                '../save_model/ICRL-Walker/test/dicrl-spline-cvar-WalkerWithPos-v0-noise-1e-1-test-multi_env-Apr-24-2023-20:37-seed_321/',
                '../save_model/ICRL-Walker/test/dicrl-spline-cvar-WalkerWithPos-v0-noise-1e-1-test-multi_env-Apr-24-2023-20:37-seed_456/',
                '../save_model/ICRL-Walker/test/dicrl-spline-cvar-WalkerWithPos-v0-noise-1e-1-test-multi_env-Apr-24-2023-20:37-seed_654/',
            ],

            'CDICRL_Spline_EXP_Walker_noise1e-1': [
                '../save_model/ICRL-Walker/test3/cdicrl-spline-exp-WalkerWithPos-v0-noise-1e-1-test-multi_env-May-02-2023-11:47-seed_123/',
                '../save_model/ICRL-Walker/test3/cdicrl-spline-exp-WalkerWithPos-v0-noise-1e-1-test-multi_env-May-02-2023-11:47-seed_321/',
                '../save_model/ICRL-Walker/test3/cdicrl-spline-exp-WalkerWithPos-v0-noise-1e-1-test-multi_env-May-02-2023-11:47-seed_456/',
                '../save_model/ICRL-Walker/test3/cdicrl-spline-exp-WalkerWithPos-v0-noise-1e-1-test-multi_env-May-02-2023-11:47-seed_654/',
            ],

            'CDICRL_Spline_CVaR_Walker_noise1e-1': [
                '../save_model/ICRL-Walker/test3/cdicrl-spline-cvar-WalkerWithPos-v0-noise-1e-1-test-multi_env-May-02-2023-11:47-seed_123/',
                '../save_model/ICRL-Walker/test3/cdicrl-spline-cvar-WalkerWithPos-v0-noise-1e-1-test-multi_env-May-02-2023-11:47-seed_321/',
                '../save_model/ICRL-Walker/test3/cdicrl-spline-cvar-WalkerWithPos-v0-noise-1e-1-test-multi_env-May-02-2023-11:47-seed_456/',
                '../save_model/ICRL-Walker/test3/cdicrl-spline-cvar-WalkerWithPos-v0-noise-1e-1-test-multi_env-May-02-2023-11:47-seed_654/',
            ],
        }
    elif env_id == 'SwimmerWithPos-v0':
        log_path_dict = {
            'ppo_lag_SwmWithPos-v0-noise-1e-1': [
                '../save_model/PPO-Lag-Swm/new/train_ppo_lag_SwmWithPos-v0-noise1e-1-new-n130-multi_env-Apr-21-2023-14:35-seed_123/',
                '../save_model/PPO-Lag-Swm/new/train_ppo_lag_SwmWithPos-v0-noise1e-1-new-n130-multi_env-Apr-21-2023-14:35-seed_321/',
                '../save_model/PPO-Lag-Swm/new/train_ppo_lag_SwmWithPos-v0-noise1e-1-new-n130-multi_env-Apr-21-2023-14:35-seed_456/',
                '../save_model/PPO-Lag-Swm/new/train_ppo_lag_SwmWithPos-v0-noise1e-1-new-n130-multi_env-Apr-21-2023-14:35-seed_654/',
            ],

            'ppo_dis_lag_Spline_CVaR_SwmWithPos-v0-noise-1e-1': [
                '../save_model/PPO-Lag-Swm/new/train_ppo_dis_lag_Spline_CVaR_SwmWithPos-v0-noise1e-1-new-n130-multi_env-Apr-21-2023-14:35-seed_123/',
                '../save_model/PPO-Lag-Swm/new/train_ppo_dis_lag_Spline_CVaR_SwmWithPos-v0-noise1e-1-new-n130-multi_env-Apr-21-2023-14:35-seed_321/',
                '../save_model/PPO-Lag-Swm/new/train_ppo_dis_lag_Spline_CVaR_SwmWithPos-v0-noise1e-1-new-n130-multi_env-Apr-21-2023-14:35-seed_456/',
                '../save_model/PPO-Lag-Swm/new/train_ppo_dis_lag_Spline_CVaR_SwmWithPos-v0-noise1e-1-new-n130-multi_env-Apr-21-2023-14:35-seed_654/',
            ],

            'ppo_dis_lag_Spline_EXP_SwmWithPos-v0-noise-1e-1': [
                '../save_model/PPO-Lag-Swm/new/train_ppo_dis_lag_Spline_EXP_SwmWithPos-v0-noise1e-1-new-n130-multi_env-Apr-21-2023-14:35-seed_123/',
                '../save_model/PPO-Lag-Swm/new/train_ppo_dis_lag_Spline_EXP_SwmWithPos-v0-noise1e-1-new-n130-multi_env-Apr-21-2023-14:35-seed_321/',
                '../save_model/PPO-Lag-Swm/new/train_ppo_dis_lag_Spline_EXP_SwmWithPos-v0-noise1e-1-new-n130-multi_env-Apr-21-2023-14:35-seed_456/',
                '../save_model/PPO-Lag-Swm/new/train_ppo_dis_lag_Spline_EXP_SwmWithPos-v0-noise1e-1-new-n130-multi_env-Apr-21-2023-14:35-seed_654/',
            ],

            'ICRL_new_SwmWithPos-v0-noise-1e-1': [
                '../save_model/ICRL-Swm-old/new/train_ICRL_SwmWithPos-v0-noise1e-1-multi_env-Apr-21-2023-17:25-seed_123/',
                '../save_model/ICRL-Swm-old/new/train_ICRL_SwmWithPos-v0-noise1e-1-multi_env-Apr-21-2023-17:25-seed_321/',
                '../save_model/ICRL-Swm-old/new/train_ICRL_SwmWithPos-v0-noise1e-1-multi_env-Apr-21-2023-17:25-seed_456/',
                '../save_model/ICRL-Swm-old/new/train_ICRL_SwmWithPos-v0-noise1e-1-multi_env-Apr-21-2023-17:25-seed_654/',
            ],

            'ICRL_old_SwmWithPos-v0-noise-1e-1': [
                '../save_model/ICRL-Swm/old/train_ICRL_SwmWithPos-v0-noise1e-1-old-multi_env-Apr-23-2023-23:23-seed_123/',
                '../save_model/ICRL-Swm/old/train_ICRL_SwmWithPos-v0-noise1e-1-old-multi_env-Apr-23-2023-23:23-seed_321/',
                '../save_model/ICRL-Swm/old/train_ICRL_SwmWithPos-v0-noise1e-1-old-multi_env-Apr-23-2023-23:23-seed_456/',
                '../save_model/ICRL-Swm/old/train_ICRL_SwmWithPos-v0-noise1e-1-old-multi_env-Apr-23-2023-23:23-seed_654/',
            ],

            'DICRL_Spline_CVaR_new_SwmWithPos-v0-noise-1e-1': [
                '../save_model/ICRL-Swm/new/train_DICRL_Spline_CVaR_SwmWithPos-v0-noise1e-1-multi_env-Apr-21-2023-17:25-seed_123/',
                '../save_model/ICRL-Swm/new/train_DICRL_Spline_CVaR_SwmWithPos-v0-noise1e-1-multi_env-Apr-21-2023-17:25-seed_321/',
                '../save_model/ICRL-Swm/new/train_DICRL_Spline_CVaR_SwmWithPos-v0-noise1e-1-multi_env-Apr-21-2023-17:25-seed_456/',
                '../save_model/ICRL-Swm/new/train_DICRL_Spline_CVaR_SwmWithPos-v0-noise1e-1-multi_env-Apr-21-2023-17:25-seed_645/',
            ],

            'DICRL_Spline_EXP_new_SwmWithPos-v0-noise-1e-1': [
                '../save_model/ICRL-Swm/new/train_DICRL_Spline_EXP_SwmWithPos-v0-noise1e-1-multi_env-Apr-21-2023-17:25-seed_123/',
                '../save_model/ICRL-Swm/new/train_DICRL_Spline_EXP_SwmWithPos-v0-noise1e-1-multi_env-Apr-21-2023-17:25-seed_321/',
                '../save_model/ICRL-Swm/new/train_DICRL_Spline_EXP_SwmWithPos-v0-noise1e-1-multi_env-Apr-21-2023-17:25-seed_456/',
                '../save_model/ICRL-Swm/new/train_DICRL_Spline_EXP_SwmWithPos-v0-noise1e-1-multi_env-Apr-21-2023-17:25-seed_654/',
            ],

            'DICRL_Spline_EXP_old_SwmWithPos-v0-noise-1e-1': [
                '../save_model/ICRL-Swm/old/train_DICRL_Spline_EXP_SwmWithPos-v0-noise1e-1-old-multi_env-Apr-23-2023-23:31-seed_123/',
                '../save_model/ICRL-Swm/old/train_DICRL_Spline_EXP_SwmWithPos-v0-noise1e-1-old-multi_env-Apr-23-2023-23:31-seed_321/',
                '../save_model/ICRL-Swm/old/train_DICRL_Spline_EXP_SwmWithPos-v0-noise1e-1-old-multi_env-Apr-23-2023-23:31-seed_456/',
                '../save_model/ICRL-Swm/old/train_DICRL_Spline_EXP_SwmWithPos-v0-noise1e-1-old-multi_env-Apr-23-2023-23:31-seed_654/',
            ],

            'BC2L_SwmWithPos-v0-noise-1e-1': [
                '../save_model/Binary-Swm/train_BC2L_SwmWithPos-v0_noise-1e-1-multi_env-May-09-2023-18:51-seed_123/',
                '../save_model/Binary-Swm/train_BC2L_SwmWithPos-v0_noise-1e-1-multi_env-May-09-2023-18:51-seed_321/',
                '../save_model/Binary-Swm/train_BC2L_SwmWithPos-v0_noise-1e-1-multi_env-May-09-2023-18:51-seed_456/',
                '../save_model/Binary-Swm/train_BC2L_SwmWithPos-v0_noise-1e-1-multi_env-May-09-2023-18:51-seed_654/',
            ],

            'GAIL_SwmWithPos-v0-noise-1e-1': [
                '../save_model/GAIL-Swm/train_GAIL_SwmWithPos-v0_noise-1e-1-multi_env-May-12-2023-15:38-seed_123/',
                '../save_model/GAIL-Swm/train_GAIL_SwmWithPos-v0_noise-1e-1-multi_env-May-12-2023-15:38-seed_321/',
                '../save_model/GAIL-Swm/train_GAIL_SwmWithPos-v0_noise-1e-1-multi_env-May-12-2023-15:38-seed_456/',
                '../save_model/GAIL-Swm/train_GAIL_SwmWithPos-v0_noise-1e-1-multi_env-May-12-2023-15:38-seed_654/',
            ],

            'ICRL_SwmWithPos-v0-noise-1e-1': [
                '../save_model/ICRL-Swm/train_ICRL_SwmWithPos-v0-noise1e-1-multi_env-May-03-2023-14:09-seed_123/',
                '../save_model/ICRL-Swm/train_ICRL_SwmWithPos-v0-noise1e-1-multi_env-May-03-2023-14:09-seed_321/',
                '../save_model/ICRL-Swm/train_ICRL_SwmWithPos-v0-noise1e-1-multi_env-May-03-2023-14:09-seed_456/',
                '../save_model/ICRL-Swm/train_ICRL_SwmWithPos-v0-noise1e-1-multi_env-May-03-2023-14:09-seed_654/',
            ],

            'VICRL_SwmWithPos-v0-noise-1e-1': [
                '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_noise-1e-1-multi_env-May-10-2023-14:00-seed_123/',
                '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_noise-1e-1-multi_env-May-10-2023-14:00-seed_321/',
                '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_noise-1e-1-multi_env-May-10-2023-01:05-seed_456/',
                '../save_model/VICRL-Swm/train_VICRL_SwmWithPos-v0_noise-1e-1-multi_env-May-10-2023-14:00-seed_654/',
            ],

            'DICRL_Spline_EXP_SwmWithPos-v0-noise-1e-1': [
                '../save_model/ICRL-Swm/train_DICRL_Spline_EXP_SwmWithPos-v0-noise1e-1-multi_env-May-03-2023-14:09-seed_123/',
                '../save_model/ICRL-Swm/train_DICRL_Spline_EXP_SwmWithPos-v0-noise1e-1-multi_env-May-03-2023-14:09-seed_321/',
                '../save_model/ICRL-Swm/train_DICRL_Spline_EXP_SwmWithPos-v0-noise1e-1-multi_env-May-03-2023-14:09-seed_456/',
                '../save_model/ICRL-Swm/train_DICRL_Spline_EXP_SwmWithPos-v0-noise1e-1-multi_env-May-03-2023-14:09-seed_654/',
            ],
            'CDICRL_Spline_EXP_SwmWithPos-v0-noise-1e-1': [
                '../save_model/ICRL-Swm/CDICRL/train_CDICRL_Spline_EXP_SwmWithPos-v0-noise1e-1-multi_env-May-07-2023-21:20-seed_123/',
                '../save_model/ICRL-Swm/CDICRL/train_CDICRL_Spline_EXP_SwmWithPos-v0-noise1e-1-multi_env-May-07-2023-21:20-seed_321/',
                '../save_model/ICRL-Swm/CDICRL/train_CDICRL_Spline_EXP_SwmWithPos-v0-noise1e-1-multi_env-May-07-2023-21:20-seed_456/',
                '../save_model/ICRL-Swm/CDICRL/train_CDICRL_Spline_EXP_SwmWithPos-v0-noise1e-1-multi_env-May-07-2023-21:20-seed_654/',
            ],

            'DICRL_QRDQN_EXP_SwmWithPos-v0-noise-1e-1': [
                '../save_model/ICRL-Swm/QRDQN/train_DICRL_QRDQN_EXP_SwmWithPos-v0-noise1e-1-multi_env-May-07-2023-13:35-seed_123/',
                '../save_model/ICRL-Swm/QRDQN/train_DICRL_QRDQN_EXP_SwmWithPos-v0-noise1e-1-multi_env-May-07-2023-13:35-seed_321/',
                '../save_model/ICRL-Swm/QRDQN/train_DICRL_QRDQN_EXP_SwmWithPos-v0-noise1e-1-multi_env-May-07-2023-13:35-seed_456/',
                '../save_model/ICRL-Swm/QRDQN/train_DICRL_QRDQN_EXP_SwmWithPos-v0-noise1e-1-multi_env-May-07-2023-13:35-seed_654/',
            ],

            'DICRL_Spline_CVaR_SwmWithPos-v0-noise-1e-1': [
                '../save_model/ICRL-Swm/train_DICRL_Spline_CVaR_SwmWithPos-v0-noise1e-1-multi_env-May-03-2023-14:09-seed_123/',
                '../save_model/ICRL-Swm/train_DICRL_Spline_CVaR_SwmWithPos-v0-noise1e-1-multi_env-May-03-2023-14:09-seed_321/',
                '../save_model/ICRL-Swm/train_DICRL_Spline_CVaR_SwmWithPos-v0-noise1e-1-multi_env-May-03-2023-14:09-seed_456/',
                '../save_model/ICRL-Swm/train_DICRL_Spline_CVaR_SwmWithPos-v0-noise1e-1-multi_env-May-03-2023-14:09-seed_654/',
            ],
            'CDICRL_Spline_CVaR_SwmWithPos-v0-noise-1e-1': [
                '../save_model/ICRL-Swm/CDICRL/train_CDICRL_Spline_CVaR_SwmWithPos-v0-noise1e-1-multi_env-May-07-2023-21:20-seed_123/',
                '../save_model/ICRL-Swm/CDICRL/train_CDICRL_Spline_CVaR_SwmWithPos-v0-noise1e-1-multi_env-May-07-2023-21:20-seed_321/',
                '../save_model/ICRL-Swm/CDICRL/train_CDICRL_Spline_CVaR_SwmWithPos-v0-noise1e-1-multi_env-May-07-2023-21:20-seed_456/',
                '../save_model/ICRL-Swm/CDICRL/train_CDICRL_Spline_CVaR_SwmWithPos-v0-noise1e-1-multi_env-May-07-2023-21:20-seed_654/',
            ],
            'DICRL_QRDQN_CVaR_SwmWithPos-v0-noise-1e-1': [
                '../save_model/ICRL-Swm/QRDQN/train_DICRL_QRDQN_CVaR_SwmWithPos-v0-noise1e-1-multi_env-May-07-2023-13:35-seed_123/',
                '../save_model/ICRL-Swm/QRDQN/train_DICRL_QRDQN_CVaR_SwmWithPos-v0-noise1e-1-multi_env-May-07-2023-13:35-seed_321/',
                '../save_model/ICRL-Swm/QRDQN/train_DICRL_QRDQN_CVaR_SwmWithPos-v0-noise1e-1-multi_env-May-07-2023-13:35-seed_456/',
                '../save_model/ICRL-Swm/QRDQN/train_DICRL_QRDQN_CVaR_SwmWithPos-v0-noise1e-1-multi_env-May-07-2023-13:35-seed_654/',
            ],

        }
    elif env_id == 'highD_velocity_constraint':
        log_path_dict = {
            'BC2L-vel-noise-1e-1': [
                '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint-1e-1-multi_env-Aug-05-2023-14:22-seed_123/',
                '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint-1e-1-multi_env-Aug-05-2023-14:22-seed_321/',
                '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint-1e-1-multi_env-Aug-05-2023-14:22-seed_456/',
                '../save_model/Binary-highD-velocity/train_Binary_highD_velocity_constraint-1e-1-multi_env-Aug-05-2023-14:22-seed_654/',
            ],
            'GAIL-vel-noise-1e-1': [
                '../save_model/GAIL-highD-velocity/train_GAIL_highd_velocity_constraint-1e-1-multi_env-Aug-05-2023-14:22-seed_123/',
                '../save_model/GAIL-highD-velocity/train_GAIL_highd_velocity_constraint-1e-1-multi_env-Aug-05-2023-14:22-seed_321/',
                '../save_model/GAIL-highD-velocity/train_GAIL_highd_velocity_constraint-1e-1-multi_env-Aug-05-2023-14:22-seed_456/',
                '../save_model/GAIL-highD-velocity/train_GAIL_highd_velocity_constraint-1e-1-multi_env-Aug-05-2023-14:22-seed_654/',
            ],
            'ICRL-vel-noise-1e-1': [
                '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint-1e-1-multi_env-Aug-04-2023-15:50-seed_123/',
                '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint-1e-1-multi_env-Aug-04-2023-15:50-seed_321/',
                '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint-1e-1-multi_env-Aug-04-2023-15:50-seed_456/',
                '../save_model/ICRL-highD-velocity/train_ICRL_highD_velocity_constraint-1e-1-multi_env-Aug-04-2023-15:50-seed_654/',
            ],
            'VICRL-vel-noise-1e-1': [
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint-1e-1-multi_env-Aug-05-2023-14:22-seed_123/',
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint-1e-1-multi_env-Aug-05-2023-14:22-seed_321/',
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint-1e-1-multi_env-Aug-05-2023-14:22-seed_456/',
                '../save_model/VICRL-highD-velocity/train_VICRL_highD_velocity_constraint-1e-1-multi_env-Aug-05-2023-14:22-seed_654/',
            ],
            'CDICLR-CVaR-QRDQN-vel-noise-1e-1': [
                '../save_model/ICRL-highD-velocity/train_CDICRL_CVaR_highD_velocity_constraint-1e-1-multi_env-Aug-04-2023-15:50-seed_123/',
                '../save_model/ICRL-highD-velocity/train_CDICRL_CVaR_highD_velocity_constraint-1e-1-multi_env-Aug-04-2023-15:50-seed_321/',
                '../save_model/ICRL-highD-velocity/train_CDICRL_CVaR_highD_velocity_constraint-1e-1-multi_env-Aug-04-2023-15:50-seed_456/',
                '../save_model/ICRL-highD-velocity/train_CDICRL_CVaR_highD_velocity_constraint-1e-1-multi_env-Aug-04-2023-15:50-seed_654/',
            ],
        }
    else:
        raise ValueError("Unknown env id {0}".format(env_id))
    return log_path_dict
