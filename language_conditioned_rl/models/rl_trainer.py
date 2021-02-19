import logging
from .reward_model import LGRInferenceMixin
from .rl_gym import get_env, RewardNormalizer,gym
from .rl_agent import SARSALambdaAgent

Formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

CHOSEN_TEXT = "The car is able to reach the top of the mountain"

DEFAULT_LOGGER_PROJECT_NAME = 'valay/Langauge-Grounded-Reward-RL-Implementation'
DEFAULT_EXPERIMENT_NAME = 'SARSA-Tiling-Rw-Fn'



def create_logger(logger_name:str,level=logging.INFO):
    custom_logger = logging.getLogger(logger_name)
    ch1 = logging.StreamHandler()
    ch1.setLevel(level)
    ch1.setFormatter(Formatter)
    custom_logger.addHandler(ch1)
    custom_logger.setLevel(level)    
    return custom_logger


class Trainer:
    def __init__(self,\
                text_context=CHOSEN_TEXT, \
                model_hidden=256,\
                num_eps=50000,\
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
                ) -> None:
        
        if api_token is None:
            print("No API Key Added : Will Start Offline Session")
        self.text_context = text_context
        self.model_hidden = model_hidden
        self.num_eps = num_eps
        self.num_timesteps = num_timesteps
        self.reward_scaleup = reward_scaleup
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.api_token = api_token
        self.video_save_freq = video_save_freq
        self.video_save_dir = video_save_dir
        self.reward_max = reward_max
        self.reward_min = reward_min
        self.reward_scaleup = reward_scaleup
        self.log_every=log_every


    def get_core_config(self,reward_fn:LGRInferenceMixin):
        ENV = "MountainCar-v0"
        config = dict(
            env=ENV,
            text_context=self.text_context,
            model_hidden=self.model_hidden,
            num_eps=self.num_eps,
            num_timesteps=self.num_timesteps,
            run_name=reward_fn.experiment_name,
            model_pth=reward_fn.loaded_checkpoint,
        )
        return config


    def run_episode(self,env:gym.Env,agent:SARSALambdaAgent):
        traj_tuples ,states ,actions = [],[],[]
        observation = env.reset()
        # $ Run Loop Over Timesteps
        for t in range(self.num_timesteps):
            states.append(observation)
            at = agent.act(observation)
            actions.append(at)
            st1, reward, done, info = env.step(at)
            traj_tuples.append((observation, at, st1, done))
            if done:
                break
            observation = st1
        return traj_tuples ,states ,actions


    def run(self,reward_fn:LGRInferenceMixin):
        import neptune
        config = self.get_core_config(reward_fn)
        # $ Create Logger. 
        if self.api_token is not None:
            neptune.init(self.project_name,api_token=self.api_token)
        else:
            neptune.init(self.project_name,backend=neptune.OfflineBackend())

        neptune.create_experiment(name=self.experiment_name, params=config)
        # $ Gets Mountain Car environment. 
        env = get_env(video_save_dir=self.video_save_dir,episode_save_freq=self.video_save_freq)
        # $ Create new Agent And Start the Episode.
        agent = SARSALambdaAgent(env)
        logger = create_logger(self.experiment_name)
        try:
            for e in range(self.num_eps):
                # $ run the episode 
                traj_tuples ,states ,actions = self.run_episode(env,agent)
                # $ Get the rewards from Trajectory. 
                reward = reward_fn.get_rewards(states, actions, self.text_context)
                # $ Normalize and Ammortize the rws
                reward = RewardNormalizer.normalize_reward(reward, \
                                            max_v=self.reward_max,\
                                            min_v=self.reward_min)
                reward *= self.reward_scaleup
                ammt_rw = RewardNormalizer.ammortized_rw(reward, len(traj_tuples))
                for idx, rw_tup in enumerate(zip(traj_tuples, ammt_rw)):
                    tup, rw = rw_tup
                    observation, at, st1, done = tup
                    next_act = None
                    if idx < len(traj_tuples)-1:
                        next_act = traj_tuples[idx+1][1]
                    agent.learn(observation, at, rw.item(),st1, done, action_next=next_act)
                
                neptune.log_metric('reward', e, y=reward)
                if e % self.log_every == 0:
                    logger.info(f'Completed Episode {e} With Reward {reward}')
        except KeyboardInterrupt as e:
            logger.error("Keyboard Interrupt Occured")
            env.close()
            neptune.stop()
            return 
        env.close()
        neptune.stop()
