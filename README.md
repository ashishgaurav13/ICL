## Inverse Constraint Learning
This repository contains the code for ICL paper. After you run any command, the
results will be logged to Wandb (online) as well as tensorboard (locally).

## Before running anything
* Install OpenMPI and `tensorflow,torch==1.11.0,gym==0.25`.
* Ensure Mujoco libraries (2.1.0) are installed.
* Install `mujoco,mujoco_py`.
* Install `custom_envs` package by running `pip install -e .` in the `custom_envs` directory.
* Install `safe_rl` package by running `pip install -e .` in the `safe_rl` directory.
* Update lines 410-413 in `tools/environments/exiD_environment.py` to reflect the directory of
ExiD dataset.
* Install `tools` package by running `pip install .` in the root directory. 

## Wandb setup
* Create an account on Wandb (https://wandb.ai)
* Install Wandb package: `pip3 install wandb`
* Login to wandb: `wandb login`

## High level workflow

* If you face any OpenGL error, install `Xvfb` and prefix the command with `xvfb-run -a`.
* For the rest of the commands, replace:
    * `SEED=1/2/3/4/5/anything`
    * `BETA=anything` (if `BETA=-1` then the default, defined in the config file, is used)
    * `ENV` is defined depending on the environments:
        * Gridworld (A): `ENV=gridworldA`
        * Gridworld (B): `ENV=gridworldB`
        * CartPole (MR): `ENV=cartpoleMR`
        * CartPole (Mid): `ENV=cartpoleM`
        * HighD: `ENV=highdgap`
        * Ant-Constrained: `ENV=ant`
        * HalfCheetah-Constrained: `ENV=hc`
        * ExiD: `ENV=exid`
* Expert data (either generate OR use saved data):
    * Use saved data: `cp expert-data/data-ENV.pt data.pt`
    * Generate for HighD environment: `python3 -B expert-highD.py`
    * Generate for ExiD environment: `python3 -B expert-exiD.py` (this uses
    data in `exidtraj`, already provided, which was generated using `prepare_exid_data.py`)
    * Generate for other environments: `python3 -B expert.py -c configs/ENV.json`
* Run methods
    * ICL: `python3 -B 03-icl-mix-improved.py -c configs/ENV.json -seed SEED -beta BETA`
    * GAIL-Constraint: `python3 -B 11-gail.py -c configs/gail-ENV.yaml -seed SEED`
    * ICRL: `python3 -B 12-icrl.py -c configs/icrl-ENV.yaml -seed SEED`

## Credits

Please check the individual repositories for licenses.

* ICRL code and `custom_envs`: 
  * https://github.com/shehryar-malik/icrl
  * https://github.com/Guiliang/constraint-learning
* OpenAI safety agents (`safe_rl`):
  * https://github.com/openai/safety-starter-agents
* HighD dataset
  * https://www.highd-dataset.com
  * We include one sample set of assets (#17) from the dataset in the code, since it is necessary to run the HighD environment.
* ExiD dataset
  * https://www.exid-dataset.com/
  * Free for non-commercial use, but to get the dataset, you must request it.
  * You must place this dataset in (any) directory and update `tools/environments/exiD_environment.py` as
  mentioned previously.
* Wise-Move environment
  * https://git.uwaterloo.ca/wise-lab/wise-move
  * https://github.com/ashishgaurav13/wm2
* Gridworld environment
  * https://github.com/yrlu/irl-imitation
* Gym environment and wrappers
  * https://github.com/ikostrikov/pytorch-ddpg-naf
  * https://github.com/vwxyzjn/cleanrl
  * https://github.com/openai/gym
* Normalizing flows
  * https://github.com/ikostrikov/pytorch-flows
  * https://github.com/tonyduan/normalizing-flows
  * https://github.com/VincentStimper/normalizing-flows
