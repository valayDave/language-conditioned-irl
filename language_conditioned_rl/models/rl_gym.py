import torch
import json
import os
import time
import gym
import math
from gym import error
from gym.utils import atomic_write
from gym.utils.json_utils import json_encode_np
from gym import envs

ENV = "MountainCar-v0"

class CustomStatsRecorder(object):
    def __init__(self, directory, file_prefix, autoreset=False, env_id=None):
        self.autoreset = autoreset
        self.env_id = env_id

        self.initial_reset_timestamp = None
        self.directory = directory
        self.file_prefix = file_prefix
        self.episode_lengths = []
        self.episode_rewards = []
        self.episode_types = [] # experimental addition
        self.current_trajectory = []
        self.collected_trajectories = []
        self._type = 't'
        self.timestamps = []
        self.steps = None
        self.total_steps = 0
        self.rewards = None

        self.done = None
        self.closed = False

        filename = '{}.stats.json'.format(self.file_prefix)
        self.path = os.path.join(self.directory, filename)

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, type):
        if type not in ['t', 'e']:
            raise error.Error('Invalid episode type {}: must be t for training or e for evaluation', type)
        self._type = type

    def before_step(self, action):
        assert not self.closed
        self.current_trajectory.append(
            dict(
                action = action,
            )
        )
        if self.done:
            raise error.ResetNeeded("Trying to step environment which is currently done. While the monitor is active for {}, you cannot step beyond the end of an episode. Call 'env.reset()' to start the next episode.".format(self.env_id))
        elif self.steps is None:
            raise error.ResetNeeded("Trying to step an environment before reset. While the monitor is active for {}, you must call 'env.reset()' before taking an initial step.".format(self.env_id))
    
    @classmethod
    def from_statsmonitor(cls,monitor):
      return cls(monitor.directory, 
           monitor.file_prefix,#'{}.episode_batch.{}'.format(self.file_prefix, self.file_infix), 
           autoreset= monitor.autoreset,
           env_id=monitor.env_id)

    def after_step(self, observation, reward, done, info):
        self.steps += 1
        self.total_steps += 1
        self.rewards += reward
        self.done = done
        
        data = dict(
                observation = observation,
                step = self.steps,
                rewards = reward,
                done = done,
            )
        self.current_trajectory[-1].update(data)
        if done:
            self.save_complete()

        if done:
            if self.autoreset:
                self.before_reset()
                self.after_reset(observation)

    def before_reset(self):
        assert not self.closed

        if self.done is not None and not self.done and self.steps > 0:
            raise error.Error("Tried to reset environment which is not done. While the monitor is active for {}, you cannot call reset() unless the episode is over.".format(self.env_id))

        self.done = False
        if self.initial_reset_timestamp is None:
            self.initial_reset_timestamp = time.time()

    def after_reset(self, observation):
        self.steps = 0
        self.rewards = 0
        # We write the type at the beginning of the episode. If a user
        # changes the type, it's more natural for it to apply next
        # time the user calls reset().
        self.episode_types.append(self._type)

    def save_complete(self):
        if self.steps is not None:
            self.episode_lengths.append(self.steps)
            self.episode_rewards.append(float(self.rewards))
            self.timestamps.append(time.time())
            self.collected_trajectories.append(
                self.current_trajectory
            )
            self.current_trajectory = []


    def close(self):
        self.flush()
        self.closed = True

    def flush(self):
        if self.closed:
            return

        with atomic_write.atomic_write(self.path) as f:
            json.dump({
                'initial_reset_timestamp': self.initial_reset_timestamp,
                'timestamps': self.timestamps,
                'episode_lengths': self.episode_lengths,
                'episode_rewards': self.episode_rewards,
                'episode_types': self.episode_types,
                'collected_trajectories':self.collected_trajectories
            }, f, default=json_encode_np)
            

class RewardNormalizer:

    @staticmethod
    def ammortized_rw(rw, num_items):
        rws = rw/num_items * torch.ones(num_items)
        ammot_rw = []
        for i in range(num_items):
            ammot_rw.append(rws[i:].sum())
        # print(ammot_rw)
        return torch.vstack(ammot_rw)

    @staticmethod
    def normalize_reward(x, max_v=8, min_v=-8):  # Normalizes
        abs_range = abs(max_v - min_v)
        used_v = (x - min_v)
        return (used_v)/(abs_range) - 1

    @staticmethod
    def normalize_reward_exp(x, max_v=6, min_v=-5):  # Normalizes exponentially
        if min_v <= x <= max_v:
            return - (1 - math.exp(x - max_v))

        abs_range = abs(max_v - min_v)
        return - (1 - math.exp(x - abs_range))


def get_env(video_save_dir='./video',episode_save_freq=20):
    from gym.wrappers import Monitor
    # lambda fn to see if episode should be saved. 
    save_fn = lambda epid : epid % episode_save_freq == 0
    env = Monitor(gym.make(ENV), video_save_dir, video_callable=save_fn,force=True)
    # set custom stats recorder. 
    new_statsmon = CustomStatsRecorder.from_statsmonitor(env.stats_recorder)
    env.stats_recorder = new_statsmon
    return env
