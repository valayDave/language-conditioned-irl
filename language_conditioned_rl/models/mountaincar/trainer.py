import logging
from .gym import get_env, RewardNormalizer,gym
from .agent import SARSALambdaAgent
from ..reward_model import LGRMountainCarInferenceMixin

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


    def get_core_config(self,reward_fn:LGRMountainCarInferenceMixin,text_context:str=None):
        ENV = "MountainCar-v0"
        reward_ob = dict(used_rw_fn = False)
        if reward_fn is not None:
            reward_ob = dict(
                **reward_ob,
                run_name=reward_fn.experiment_name,
                model_pth=reward_fn.loaded_checkpoint,
            )
        config = dict(
            reward_scaleup=self.reward_scaleup,
            reward_min=self.reward_min,
            reward_max=self.reward_max,
            env=ENV,
            text_context=text_context,
            model_hidden=self.model_hidden,
            num_eps=self.num_eps,
            num_timesteps=self.num_timesteps,
            **reward_ob
        )
        return config


    def run_episode(self,env:gym.Env,agent:SARSALambdaAgent,render:bool=False):
        traj_tuples ,states ,actions = [],[],[]
        observation = env.reset()
        # $ Run Loop Over Timesteps
        for t in range(self.num_timesteps):
            states.append(observation)
            at = agent.act(observation)
            actions.append(at)
            st1, reward, done, info = env.step(at)
            if render:
                env.render()
            traj_tuples.append((observation, at, st1, done))
            if done:
                break
            observation = st1
        return traj_tuples ,states ,actions


    def run(self,reward_fn:LGRMountainCarInferenceMixin,render:bool=False,seed:int=None,text_context:str=CHOSEN_TEXT):
        import neptune
        config = self.get_core_config(reward_fn,text_context=text_context)
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
        if seed is not None:
            env.seed(seed=seed)

        try:
            for e in range(self.num_eps):
                # $ run the episode 
                traj_tuples ,states ,actions = self.run_episode(env,agent,render=render)
                # $ Get the rewards from Trajectory. 
                reward = reward_fn.get_rewards(states, actions, text_context)
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
            if self.video_save_dir:
                neptune.log_artifact(self.video_save_dir)
            neptune.stop()
            return 
        env.close()
        if self.video_save_dir:
            neptune.log_artifact(self.video_save_dir)
        neptune.stop()

    
    def play_sarsa_native(self,env:gym.Env,agent:SARSALambdaAgent,render:bool=False,train:bool=True):
        episode_reward = 0
        observation = env.reset()
        action = agent.act(observation)
        while True:
            if render:
                env.render()
            observation_next, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                if train:
                    agent.learn(observation, action, reward, observation_next, done)
                break
            action_next = agent.act(observation_next)
            if train:
                agent.learn(observation, action, reward, observation_next, done, action_next)
            observation, action = observation_next, action_next
        return episode_reward
        

    def run_native(self,train=True,render=False,seed:int=None):
        import neptune
        config = self.get_core_config(None,)
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
        if seed is not None:
            env.seed(seed=seed)
        try:
            for e in range(self.num_eps):
                episodereward = self.play_sarsa_native(env,agent,render=render,train=train)
                neptune.log_metric('reward', e, y=episodereward)
                if e % self.log_every == 0:
                    logger.info(f'Completed Episode {e} With Reward {episodereward}')

        except KeyboardInterrupt as e:
            logger.error("Keyboard Interrupt Occured")
            env.close()
            if self.video_save_dir:
                neptune.log_artifact(self.video_save_dir)
            neptune.stop()
            return 
        
        env.close()
        if self.video_save_dir:
            neptune.log_artifact(self.video_save_dir)
        neptune.stop()