# Uncertainty-aware Inverse Constrained Reinforcement Learning
![](https://github.com/Jasonxu1225/Uncertainty-aware-Inverse-Constrained-Reinforcement-Learning/blob/main/workflow.jpg)
This is the code for the paper [Uncertainty-aware Constraint Inference in Inverse Constrained Reinforcement Learning](https://openreview.net/pdf?id=ILYjDvUM6U) published at ICLR 2024. Note that:
1. Our project relies on [MuJoCo](https://mujoco.org/) and [CommonRoad](https://commonroad.in.tum.de/).
2. The implementation is based on the code from [ICRL-benchmark](https://github.com/Guiliang/ICRL-benchmarks-public/tree/main).

## Create Python Environment 
1. Please install the conda before proceeding.
2. Create conda environment and install the packages:
   
```
mkdir save_model
mkdir evaluate_model
conda env create -n py39 python=3.9 -f python_environment.yml
conda activate py39
```
You can also fistly install the Python 3.9 with Pytorch and then install the packages listed in `python_environment.yml`.

## Setup Experimental Environments 
### 1. Setup MuJoCo Environment (you can also refer to [MuJoCo Setup](https://github.com/Guiliang/ICRL-benchmarks-public/blob/main/virtual_env_tutorial.md))
1. Download the MuJoCo version 2.1 binaries for Linux or OSX.
2. Extract the downloaded mujoco210 directory into ~/.mujoco/mujoco210.
3. Install and use mujoco-py.
```
pip install -U 'mujoco-py<2.2,>=2.1'
pip install -e ./mujuco_environment

export MUJOCO_PY_MUJOCO_PATH=YOUR_MUJOCO_DIR/.mujoco/mujoco210
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:YOUR_MUJOCO_DIR/.mujoco/mujoco210/bin:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```
### 2. Setup CommonRoad Environment (you can also refer to [CommonRoad Setup](https://github.com/Guiliang/ICRL-benchmarks-public/blob/main/realisitic_env_tutorial.md))
```
sudo apt-get update
sudo apt-get install build-essential make cmake

# option 1: Install with sudo rights (cn-py37 is the name of conda environment).
cd ./commonroad_environment
bash ./scripts/install.sh -e cn-py37

# Option 2: Install without sudo rights
bash ./commonroad_environment/scripts/install.sh -e cn-py37 --no-root
```

## Generate Expert Demonstration
Note that we have generated the expert data for the ease of usage, and you can download it through [expert_data](123).

Alternatively, you can also generate your own dataset with different settings such as different constraints or noise levels through the following steps (here we use the Blocked Half-Cheetah environment with noise level 1e-3 as an example):
### 1. Train expert agents with ground-truth constraints.
Firstly we should train an expert agent (PPO-Lag) with ground-truth constraints:
```
# run PPO-Lag knowing the ground-truth constraints
python train_policy.py ../config/Mujoco/Blocked_HalfCheetah/train_PPO-Lag_HC-noise-1e-3.yaml -n 5 -s 123
```

### 2. Sample trajectories of the expert
After training the expert agent, we can get the expert demonstration through sampling from it:
```
# run data generation
python generate_data_for_constraint_inference.py -n 5 -mn expert_file_path -tn PPO-Lag-HC -ct no-constraint -rn 0
```
Note that you need you replace the `expert_file_path` by the saved path of your trained expert. You can find it through `save_model/PPO-Lag-HC/expert_file_path`.

## Train ICRL Algorithms
We use the `Blocked Half-Cheetah` environment with noise level 1e-1 and seed 123 as an example. You can also modify the noise level by using different configs and change the seed.

```
# train GACL
python train_gail.py ..config/Mujoco/Blocked_HalfCheetah/train_GAIL_HC-noise-1e-1.yaml -n 5 -s 123

# train BC2L
python train_icrl.py ...config/Mujoco/Blocked_HalfCheetah/train_BC2L_HC-noise-1e-1.yaml -n 5 -s 123

# train ICRL
python train_icrl.py ..config/Mujoco/Blocked_HalfCheetah/train_ICRL_HC-noise-1e-1.yaml -n 5 -s 123

# train VICRL
python train_icrl.py ..config/Mujoco/Blocked_HalfCheetah/train_VICRL_HC-noise-1e-1.yaml -n 5 -s 123

# train UAICRL
python train_icrl.py ..config/Mujoco/Blocked_HalfCheetah/train_UAICRL_HC-noise-1e-1.yaml -n 5 -s 123
```

## Evaluate Results
After training the algorithms, we can evaluate their performance through the following steps:
1. Modify `plot_results_dirs.py` in `interface/plot_results` to add the log path of different algorithms with respect to different environments.
2. Run `generate_running_plots.py` and check the results and figures in `plot_results` folder.

## Welcome to Cite and Star
If you have any questions, please contact me via shengxu1@link.cuhk.edu.cn.

If you feel the project helpful, please use the citation:
```
@inproceedings{xu2024uaicrl,
  title={Uncertainty-aware Constraint Inference in Inverse Constrained Reinforcement Learning},
  author={Xu, Sheng and Liu, Guiliang},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
  url={https://openreview.net/pdf?id=ILYjDvUM6U}
}
```
