# Language Conditioned Inverse Reinforcement Learning On Mountain Car

Core docs for the project can be found [here](Docs/README.md)

## Core Credits: 

1. https://arxiv.org/pdf/1906.00295.pdf
2. https://arxiv.org/abs/2009.01325
3. https://github.com/huggingface/transformers/tree/master/src/transformers
4. https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_pytorch.py
5. https://github.com/ZhiqingXiao/OpenAIGymSolution/tree/master/MountainCar-v0

## Disclaimer: 
1. This is a very old project made in 2021, while I was in grad school (before transformers were sexy and ChatGPT didn't exist) so running it and expecting it to work feels like wishful thinking. 
2. This project is a research project , most of it won't work today, BUT the ideas can still be useful.
3. The key idea behind this project was: can you learn a reward function that also takes language, states, actions and predicts a numeric reward based on how much the state-actions pairs correlate with the natural language input (higher the correlation, yields a large positive value vs lower the correlation, yields a large negative value); If such a reward function can exist then one can potentially train many different RL agents with different behaviors by only changing the language inputs to the function.
  - This project tried its luck with "mountain car" as the baseline example. It trained a Reward function (`R(state,action,text)`) that predicts different rewards based on the value of the sentence for the same state-action pairs.
  - For example (Assumed a trained reward function):
    - if a state-action trajectory contained a mountain car only moving at the bottom of the valley, and the sentence given to the reward function was "The car only moves around at the bottom of the valley", then the reward function would give a high positive reward.
    - if a state-action trajectory contained a mountain car only moving at the bottom of the valley, and the sentence given to the reward function was "The car reaches the top of the mountain", then the reward function would give a high negative reward.
  - Once the function is learned using the data, it is used in the RL training loop for training agents.
  - Also tried luck at robotics with this but I didn't endup having enough compute or enough data to tackle the problem well enough. 
4. See some of the testing results for : [Mountain car](./Experiment-Notebooks/LGR_OC_Transformer_MC_From_Git.ipynb) / [Robotics](./Experiment-Notebooks/LGR_Robot_Experiments_Final_Tests.ipynb); The plots in the notebooks are trying to help understand the reward model's behavior when a sentence matches the trajectory given to it and when it doesn't. 
