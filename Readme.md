# Language Conditioned Inverse Reinforcement Learning On Mountain Car


## Running RL Training
```python
# coding: utf-8
from language_conditioned_rl.models.reward_model import LGRBehaviouralDiffLearnerInference
from language_conditioned_rl.models.rl_trainer import Trainer,DEFAULT_LOGGER_PROJECT_NAME,DEFAULT_EXPERIMENT_NAME
REWARD_FN_PROJECT_NAME = 'valay/Language-Grounded-Rewards'
REWARD_FN_EXPERIMENT_NAME = 'LAN-21'
CHECKPOINT_PATH = 'checkpoints/epoch=12-val_loss=0.98.ckpt'
API_TOKEN = "<NEPTUNE_API_TOKEN>"
CHOSEN_TEXT = "The car is able to reach the top of the mountain"
# Instantiate Reward Function
REWARD_FN,config = LGRBehaviouralDiffLearnerInference.from_neptune(REWARD_FN_PROJECT_NAME,REWARD_FN_EXPERIMENT_NAME,CHECKPOINT_PATH,api_token=API_TOKEN)

# Run RL training with Text bound reward function. 
trainer = Trainer(
    text_context=CHOSEN_TEXT,
    num_eps=10000,
    model_hidden=256,\
    num_timesteps=200,\
    reward_scaleup=100,\
    project_name=DEFAULT_LOGGER_PROJECT_NAME,\
    experiment_name=DEFAULT_EXPERIMENT_NAME,\
    api_token=None,\
    video_save_freq=200,\
    video_save_dir='./video',\
    log_every=100,\
    reward_min=-4,\
    reward_max=4,\
)

trainer.run(REWARD_FN)
```

## Model Docs 

- TODO 

