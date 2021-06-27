# Reinforcement Learning Using Learned Reward Function 


## Mountaincar RL Training

```python 
# coding: utf-8
from language_conditioned_rl.models.reward_model import LGROmniChannelPureContrastiveRewardLearner
from language_conditioned_rl.models.rl_trainer import MountainCarTrainer,\
                                                    MOUNTAINCAR_DEFAULT_LOGGER_PROJECT_NAME,\
                                                    MOUNTAINCAR_DEFAULT_EXPERIMENT_NAME
import random
REWARD_FN_PROJECT_NAME = 'valay/Language-Grounded-Rewards'
REWARD_FN_EXPERIMENT_NAME = 'LAN-114'
CHECKPOINT_PATH = 'checkpoints/epoch=01-val_loss=0.00.ckpt'
# Neptune API Token 
API_TOKEN = None
CHOSEN_TEXT = "The car swings around at the bottom of the valley"
# Instantiate Reward Function
REWARD_FN,config = LGROmniChannelPureContrastiveRewardLearner.from_neptune(REWARD_FN_PROJECT_NAME,REWARD_FN_EXPERIMENT_NAME,CHECKPOINT_PATH,api_token=API_TOKEN)
# Run RL training with Text bound reward function. 
trainer = MountainCarTrainer(
    num_eps=400,
    model_hidden=256,\
    num_timesteps=200,\
    reward_scaleup=100,\
    project_name=MOUNTAINCAR_DEFAULT_LOGGER_PROJECT_NAME,\
    experiment_name=MOUNTAINCAR_DEFAULT_EXPERIMENT_NAME,\
    api_token=API_TOKEN,\
    video_save_freq=20,\
    video_save_dir='./video',\
    log_every=100,\
    reward_min=-12,\
    reward_max=12,\
)

NUM_EXPERIMENT_PER_TYPE = 10
SENTENCES = [
    None,
    "The car swings around at the bottom of the valley.",
    "The car is able swing beyond the bottom of the valley but does not reach the top of the mountain",
    "The car is able to reach the top of the mountain",
]
RANDOM_SEEDS = [random.randint(0,1000) for _ in range(NUM_EXPERIMENT_PER_TYPE)]

for s in SENTENCES:
    for r in RANDOM_SEEDS:
        if s is None:
            trainer.run_native(render=False)
        else:
            trainer.run(REWARD_FN,render=False,text_context=s)
```

## Robotics RL Quick Starter 

### DDPG Agent 
```python 
from language_conditioned_rl.models.robotics.trainer import Simulator
from language_conditioned_rl.models.robotics.reward_model import make_model_from_checkpoint
from language_conditioned_rl.models.robotics.agent import RobotAgent,DDPGArgs
import os 
import signal
import neptune

# Neputune API key
API_KEY = None
VREP_SCENE = "./GDrive/NeurIPS2020.ttt"
VREP_SCENE = os.path.join(os.getcwd(),VREP_SCENE)
reward_fn = make_model_from_checkpoint(
    api_token=API_KEY,
    max_traj_length = 500,
    project_name='valay/Language-Grounded-Rewards-Robo',\
    # experiment_name='LRO-89',\
    experiment_name='LRO-113',\
    checkpoint_path='checkpoints/last.pt', \
)
pose_args = DDPGArgs()
pose_args.bounding_min = -1
pose_args.rate = 0.0001
pose_args.bounding_max= 1
agent = RobotAgent(pose_args=pose_args,learning_rate=0.0001,batch_size=128,)
sim = Simulator(
    scenepath=VREP_SCENE,
    headless=True,
    reward_scaleup=5,
    num_eps=2000,
    reward_max=0,
    video_save_freq=2,
    video_save_dir='./robo-videos',
    update_episodically=False,
    reward_min=-15,
    api_token=API_KEY,
)
filename = sim.get_picking_epsiode_file(path='./GDrive/testdata/*_1.json')
sim.picking_task_rl(
    filename,
    reward_fn,
    agent,
    train=True
)
sim.shutdown()
        
```

### CMAES Agent 

```python 
from language_conditioned_rl.models.robotics.cmaes_trainer import CMAESSimulation
from language_conditioned_rl.models.robotics.reward_model import make_model_from_checkpoint
from language_conditioned_rl.models.robotics.agent import RobotAgent,DDPGArgs
import os 
import signal
import neptune
import glob
import random 

# Neputune API key
API_KEY = None
VREP_SCENE = "./GDrive/NeurIPS2020.ttt"
VREP_SCENE = os.path.join(os.getcwd(),VREP_SCENE)

def get_picking_epsiode_file(path="../GDrive/testdata/*_1.json"):
    files = glob.glob(path)
    return random.choice(files)

reward_fn = make_model_from_checkpoint(
    api_token=API_KEY,
    max_traj_length = 500,
    project_name='valay/Language-Grounded-Rewards-Robo',\
    # experiment_name='LRO-89',\
    experiment_name='LRO-113',\
    checkpoint_path='checkpoints/last.pt', \
)
filename = get_picking_epsiode_file(path='./GDrive/testdata/*_1.json')
sim = CMAESSimulation(video_save_dir='./cmaes-videos',api_token=API_KEY)
sim.picking_task_cmaes(
    filename,
    reward_fn,
    num_gaussians=40,
    population_size=8,
    is_parallel=True,
    num_parallel_procs=2,
)
```