# Language Conditioned Inverse Reinforcement Learning On Mountain Car


# Running RL Training
```python
# coding: utf-8
# coding: utf-8
from language_conditioned_rl.models.reward_model import LGROmniChannelPureContrastiveRewardLearner
from language_conditioned_rl.models.rl_trainer import MountainCarTrainer,\
                                                    MOUNTAINCAR_DEFAULT_LOGGER_PROJECT_NAME,\
                                                    MOUNTAINCAR_DEFAULT_EXPERIMENT_NAME
REWARD_FN_PROJECT_NAME = 'valay/Language-Grounded-Rewards'
REWARD_FN_EXPERIMENT_NAME = 'LAN-114'
CHECKPOINT_PATH = 'checkpoints/epoch=01-val_loss=0.00.ckpt'
API_TOKEN = "<NEPTUNE_API_TOKEN>"
CHOSEN_TEXT = "The car is able to reach the top of the mountain"
# Instantiate Reward Function
REWARD_FN,config = LGROmniChannelPureContrastiveRewardLearner.from_neptune(REWARD_FN_PROJECT_NAME,REWARD_FN_EXPERIMENT_NAME,CHECKPOINT_PATH,api_token=API_TOKEN)
# Run RL training with Text bound reward function. 
trainer = MountainCarTrainer(
    text_context=CHOSEN_TEXT,
    num_eps=1000,
    model_hidden=256,\
    num_timesteps=200,\
    reward_scaleup=100,\
    project_name=MOUNTAINCAR_DEFAULT_LOGGER_PROJECT_NAME,\
    experiment_name=MOUNTAINCAR_DEFAULT_EXPERIMENT_NAME,\
    api_token=API_TOKEN,\
    video_save_freq=200,\
    video_save_dir='./video',\
    log_every=100,\
    reward_min=-12,\
    reward_max=12,\
)

trainer.run(REWARD_FN,render=True)
```

# Core Credits: 

1. https://arxiv.org/pdf/1906.00295.pdf
2. https://arxiv.org/abs/2009.01325
3. https://github.com/huggingface/transformers/tree/master/src/transformers
4. https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_pytorch.py
5. https://github.com/ZhiqingXiao/OpenAIGymSolution/tree/master/MountainCar-v0


# Model Docs 
Docs for the Model are present [here](Docs/transformer.md). 

# Data Loading Docs
Docs for the Dataloaders are present [here](Docs/dataloading.md). 
# Training Setup : 

## Mountain Car : 
- TODO 

## Robot Experiments
- TODO
- Install CMAES Lib : `pip install git+https://github.com/CMA-ES/pycma.git@master`
# Some Observations From Training 

1. Bigger Models are finding better decision boundaries with smaller Batchsizes
2. Smaller Models are also doing good with bigger batch sizes
3. Sentence Grounding examples based data-augmentation is extreamely benificial in boosting training results. 
    1. Sentence grounding means that we creating training tuples we create 
4. Size of transfomer's embeddings were tuned down to as small as 16 but it still finds pretty distinct boundaries. 

