# Language Conditioned Inverse RL 

## Setup 

- Core library install
    ```sh
    pip install -e . # at root directory of this repo
    ```
- [Dockerfile](../Dockerfile) for building the docker image for collecting data for training inverse RL reward function. 

- Install CMAES Lib : `pip install git+https://github.com/CMA-ES/pycma.git@master`

## Table Of Contents

1. [Model Training Notebooks](../Experiment-Notebooks/)
2. [Dataset and Data Loading Docs for Reward Model Training](./dataloading.md)
3. [Robotics Datacollection Docs](./robot-data-collect.md)
4. [Training with Reinforcement Learning](./rl.md)
5. [Docs for Omni-Channel Transformer Model](./transformer.md)

