# @Refactor From: Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

import shutil
from PIL import Image
import matplotlib.pyplot as plt
import os
import os.path
import csv
import json
import glob
import pickle
import math
import cv2
import numpy as np
import select
import sys
from .utils import Voice
from pyrep.objects.vision_sensor import VisionSensor
import torch.nn as nn
from pyrep import PyRep
import matplotlib
matplotlib.use("TkAgg")
import neptune
import random
import signal
from .reward_model import RoboRewardFnMixin
from .agent import RobotAgent
from ..mountaincar.gym import RewardNormalizer


# Default robot position. You don't need to change this
DEFAULT_UR5_JOINTS = [105.0, -30.0, 120.0, 90.0, 60.0, 90.0]
# Evaluate headless or not
HEADLESS = False
# This is a debug variable...
USE_SHAPE_SIZE = True
# Run on the test data, or start the simulator in manual mode
# (manual mode will allow you to generate environments and type in your own commands)
RUN_ON_TEST_DATA = True
# How many of the 100 test-data do you want to test?
NUM_TESTED_DATA = 100
# Where to find the normailization?
NORM_PATH = "./GDrive/normalization_v2.pkl"
# Where to find the VRep scene file. This has to be an absolute path.
VREP_SCENE = "./GDrive/NeurIPS2020.ttt"

DEFAULT_LOGGER_PROJECT_NAME = 'valay/LGR-RL-Robot'
DEFAULT_EXPERIMENT_NAME = 'PICKING-Task-Agent'
import signal

def dir_exists(pth):
    try:
        os.stat(pth)
        return True
    except:
        return False

class Simulator(object):
    def __init__(self,\
                args=None,\
                scenepath=VREP_SCENE,\
                model_hidden=256,\
                num_eps=50000,\
                num_timesteps=200,\
                reward_scaleup=100,\
                project_name=DEFAULT_LOGGER_PROJECT_NAME,\
                experiment_name=DEFAULT_EXPERIMENT_NAME,\
                api_token=None,\
                update_episodically:bool=True,\
                update_frequency:int=30,\
                video_save_freq=200,\
                video_save_dir='./video',\
                log_every=100,\
                reward_min=-4,\
                reward_max=4,\
                headless=HEADLESS):
        self.pyrep = PyRep()
        self.pyrep.launch(scenepath, headless=headless)
        self.camera = VisionSensor("kinect_rgb_full")
        self.pyrep.start()
        self.update_episodically = update_episodically
        self.update_frequency = update_frequency
        self.model_hidden = model_hidden
        self.num_eps = num_eps
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
        signal.signal(signal.SIGINT,self._sigint_handler)
        self.trajectory = None
        self.global_step = 0
        self.normalization = pickle.load(
            open(NORM_PATH, mode="rb"), encoding="latin1")
        self.voice = Voice(load=False)
        self.shape_size_replacement = {}
        self.shape_size_replacement["58z29D2omoZ_2.json"] = "spill everything into the large curved dish"
        self.shape_size_replacement["P1VOZ4zk4NW_2.json"] = "fill a lot into the small square basin"
        self.shape_size_replacement["KOVJZ4Npy4G_2.json"] = "fill a small amount into the big round pot"
        self.shape_size_replacement["wjqQmB74rnr_2.json"] = "pour all of it into the large square basin"
        self.shape_size_replacement["LgVK8qXGowA_2.json"] = "fill a little into the big round bowl"
        self.shape_size_replacement["JZ90qm46ooP_2.json"] = "fill everything into the biggest rectangular bowl"
        self._setup_video_dir()

    def _setup_video_dir(self):
        if dir_exists(self.video_save_dir):
            shutil.rmtree(self.video_save_dir)
        os.mkdir(self.video_save_dir)


    def loadNlpCSV(self, path):
        self.nlp_dict = {}
        with open(path, "r") as fh:
            csvreader = csv.reader(fh, delimiter=",")
            for line in csvreader:
                if line[1] != "":
                    self.nlp_dict[line[0]+"_1.json"] = line[1]
                    self.nlp_dict[line[0]+"_2.json"] = line[2]

    def shutdown(self):
        self.pyrep.stop()
        self.pyrep.shutdown()

    def _getCameraImage(self):
        rgb_obs = self.camera.capture_rgb()
        rgb_obs = (np.asarray(rgb_obs) * 255).astype(dtype=np.uint8)
        rgb_obs = np.flip(rgb_obs, (2))
        return rgb_obs

    def _getSimulatorState(self):
        _, s, _, _ = self.pyrep.script_call(function_name_at_script_name="getState@control_script",
                                            script_handle_or_type=1,
                                            ints=(), floats=(), strings=(), bytes="")
        return s

    def _stopRobotMovement(self):
        _, _, _, _ = self.pyrep.script_call(function_name_at_script_name="stopRobotMovement@control_script",
                                            script_handle_or_type=1,
                                            ints=(), floats=(), strings=(), bytes="")

    def _getRobotState(self):
        _, s, _, _ = self.pyrep.script_call(function_name_at_script_name="getState@control_script",
                                            script_handle_or_type=1,
                                            ints=(), floats=(), strings=(), bytes="")
        state = np.take(np.asarray(s), indices=[0, 1, 2, 3, 4, 5, 30], axis=0)
        return state.tolist()

    def _setRobotJoints(self, joints):
        result = self.pyrep.script_call(function_name_at_script_name="setRobotJoints@control_script",
                                        script_handle_or_type=1,
                                        ints=(), floats=joints, strings=(), bytes="")

    def _setJointVelocityFromTarget(self, joints):
        _, s, _, _ = self.pyrep.script_call(function_name_at_script_name="setJointVelocityFromTarget@control_script",
                                            script_handle_or_type=1,
                                            ints=(), floats=joints, strings=(), bytes="")

    def _setJointVelocityFromTarget_Direct(self, joints):
        _, s, _, _ = self.pyrep.script_call(function_name_at_script_name="setJointVelocityFromTarget_Direct@control_script",
                                            script_handle_or_type=1,
                                            ints=(), floats=joints, strings=(), bytes="")

    def _dropBall(self, b_id):
        _, _, _, _ = self.pyrep.script_call(function_name_at_script_name="dropBall@control_script",
                                            script_handle_or_type=1,
                                            ints=(b_id,), floats=(), strings=(), bytes="")

    def _evalPouring(self):
        i, _, _, _ = self.pyrep.script_call(function_name_at_script_name="evalPouring@control_script",
                                            script_handle_or_type=1,
                                            ints=(), floats=(), strings=(), bytes="")
        return i

    def _graspedObject(self):
        i, _, _, _ = self.pyrep.script_call(function_name_at_script_name="graspedObject@control_script",
                                            script_handle_or_type=1,
                                            ints=(), floats=(), strings=(), bytes="")
        if i[0] >= 0:
            return True
        return False

    def _setRobotInitial(self, joints):
        _, _, _, _ = self.pyrep.script_call(function_name_at_script_name="setRobotJoints@control_script",
                                            script_handle_or_type=1,
                                            ints=(), floats=joints, strings=(), bytes="")

    def _graspClosestContainer(self):
        _, _, _, _ = self.pyrep.script_call(function_name_at_script_name="graspClosestContainer@control_script",
                                            script_handle_or_type=1,
                                            ints=(), floats=(), strings=(), bytes="")

    def _randomizeLight(self):
        _, _, _, _ = self.pyrep.script_call(function_name_at_script_name="randomizeLight@control_script",
                                            script_handle_or_type=1,
                                            ints=(), floats=(), strings=(), bytes="")

    def _resetEnvironment(self):
        self.pyrep.stop()
        self.pyrep.start()

    def _createEnvironment(self, ints, floats):
        result = self.pyrep.script_call(
            function_name_at_script_name="generateScene@control_script",
            script_handle_or_type=1,
            ints=ints,
            floats=floats,
            strings=(),
            bytes=""
        )

    def _getClosesObject(self):
        oid, dist, _, _ = self.pyrep.script_call(function_name_at_script_name="getClosesObject@control_script",
                                                 script_handle_or_type=1,
                                                 ints=(), floats=(), strings=(), bytes="")
        return oid, dist

    def dtype_with_channels_to_cvtype2(self, dtype, n_channels):
        numpy_type_to_cvtype = {'uint8': '8U', 'int8': '8S', 'uint16': '16U',
                                'int16': '16S', 'int32': '32S', 'float32': '32F',
                                'float64': '64F'}
        numpy_type_to_cvtype.up

    def predictTrajectory(self, voice, state, cnt):
        print(state)

        return trajectory, phase

    def normalize(self, value, v_min, v_max):
        if type(value) == list:
            value = np.asarray(value)
        if (value.shape[1] != v_min.shape[0] or v_min.shape[0] != v_max.shape[0] or
                len(value.shape) != 2 or len(v_min.shape) != 1 or len(v_max.shape) != 1):
            raise Exception("Shape Mismatch")
        value = np.copy(value)
        v_min = np.tile(np.expand_dims(v_min, 0), [value.shape[0], 1])
        v_max = np.tile(np.expand_dims(v_max, 0), [value.shape[0], 1])
        value = (value - v_min) / (v_max - v_min)
        return value

    def restoreValues(self, value, v_min, v_max):
        if (value.shape[1] != v_min.shape[0] or v_min.shape[0] != v_max.shape[0] or
                len(value.shape) != 2 or len(v_min.shape) != 1 or len(v_max.shape) != 1):
            print("Array dimensions are not matching!")

        value = np.copy(value)
        v_min = np.tile(np.expand_dims(v_min, 0), [value.shape[0], 1])
        v_max = np.tile(np.expand_dims(v_max, 0), [value.shape[0], 1])
        value = value * (v_max - v_min) + v_min
        return value

    def _generalizeVoice(self, voice):
        return voice

    def _mapObjectIDs(self, oid):
        if oid == 154:
            return 1
        elif oid == 155:
            return 2
        elif oid == 156:
            return 3
        elif oid == 113:
            return 1
        elif oid == 118:
            return 2
        elif oid == 124:
            return 3
        elif oid == 130:
            return 4
        elif oid == 136:
            return 5
        elif oid == 115:
            return 6
        elif oid == 119:
            return 7
        elif oid == 125:
            return 8
        elif oid == 131:
            return 9
        elif oid == 137:
            return 10
        elif oid == 148:
            return 11
        elif oid == 147:
            return 12
        elif oid == 146:
            return 13
        elif oid == 145:
            return 14
        elif oid == 143:
            return 15
        elif oid == 152:
            return 16
        elif oid == 151:
            return 17
        elif oid == 150:
            return 18
        elif oid == 149:
            return 19
        elif oid == 144:
            return 20

    def _getTargetPosition(self, data):
        state = self._getSimulatorState()
        tcp = state[12:14]
        target = data["target/id"]
        tp = data["target/type"]
        if tp == "cup":
            cups = data["ints"][2+data["ints"][0]:]
            t_id = [i for i in range(data["ints"][1])
                    if cups[i] == target][0] + data["ints"][0]
            t_pos = data["floats"][t_id*3:t_id*3+2]
        else:
            bowls = data["ints"][2:2+data["ints"][0]]
            t_id = [i for i in range(data["ints"][0]) if bowls[i] == target][0]
            t_pos = data["floats"][t_id*3:t_id*3+2]

        dist = np.sqrt(np.power(tcp[0] - t_pos[0], 2) +
                       np.power(tcp[1] - t_pos[1], 2))

        closest = list(self._getClosesObject())
        closest[0][0] = self._mapObjectIDs(closest[0][0])
        closest[0][1] = self._mapObjectIDs(closest[0][1])
        result = {}
        result["target"] = t_pos
        result["tid"] = target
        result["tid/actual"] = closest
        result["current"] = tcp
        result["distance"] = dist
        return result

    def _maybeDropBall(self, state):
        res = 0
        if state[5] > 3.0:
            self._dropBall(1)
            res = 1
        if state[5] > 3.0 and self.last_rotation > state[5]:
            self._dropBall(2)
            res = 2
        self.last_rotation = state[5]
        return res

    def _getLanguateInformation(self, voice, phs):
        def _quantity(voice):
            res = 0
            for word in self.voice.synonyms["little"]:
                if voice.find(word) >= 0:
                    res = 1
            for word in self.voice.synonyms["much"]:
                if voice.find(word) >= 0:
                    res = 2
            return res

        def _difficulty(voice):
            if phs == 2:
                voice = " ".join(voice.split()[4:])
            shapes = self.voice.synonyms["round"] + \
                self.voice.synonyms["square"]
            colors = self.voice.synonyms["small"] + \
                self.voice.synonyms["large"]
            sizes = self.voice.synonyms["red"] + self.voice.synonyms["green"] + \
                self.voice.synonyms["blue"] + \
                self.voice.synonyms["yellow"] + self.voice.synonyms["pink"]

            shapes_used = 0
            for word in shapes:
                if voice.find(word) >= 0:
                    shapes_used = 1
            colors_used = 0
            for word in colors:
                if voice.find(word) >= 0:
                    colors_used = 1
            sizes_used = 0
            for word in sizes:
                if voice.find(word) >= 0:
                    sizes_used = 1
            return shapes_used + colors_used + sizes_used

        data = {}
        data["original"] = voice
        data["features"] = _difficulty(voice)
        data["quantity"] = _quantity(voice)
        return data

    def valPhase1(self, files, feedback=True):
        successfull = 0
        val_data = {}
        nn_trajectory = []
        ro_trajectory = []
        for fid, fn in enumerate(files):
            print("Phase 1 Run {}/{}".format(fid, len(files)))
            eval_data = {}
            with open(fn + "1.json", "r") as fh:
                data = json.load(fh)

            gt_trajectory = np.asarray(data["trajectory"])
            self._resetEnvironment()
            self._createEnvironment(data["ints"], data["floats"])
            self._setRobotInitial(gt_trajectory[0, :])
            self.pyrep.step()

            eval_data["language"] = self._getLanguateInformation(
                data["voice"], 1)
            eval_data["trajectory"] = {"gt": [], "state": []}
            eval_data["trajectory"]["gt"] = gt_trajectory.tolist()

            cnt = 0
            phase = 0.0
            self.last_gripper = 0.0
            th = 1.0
            while phase < th and cnt < int(gt_trajectory.shape[0] * 1.5):
                state = self._getRobotState(
                ) if feedback else gt_trajectory[-1 if cnt >= gt_trajectory.shape[0] else cnt, :]
                cnt += 1
                tf_trajectory, phase = self.predictTrajectory(
                    data["voice"], state, cnt)
                r_state = tf_trajectory[-1, :]
                eval_data["trajectory"]["state"].append(r_state.tolist())
                r_state[6] = r_state[6]
                nn_trajectory.append(r_state)
                ro_trajectory.append(self._getRobotState())
                self.last_gripper = r_state[6]
                self._setJointVelocityFromTarget(r_state)
                self.pyrep.step()
                if r_state[6] > 0.5 and "locations" not in eval_data.keys():
                    eval_data["locations"] = self._getTargetPosition(data)

            eval_data["success"] = False
            if self._graspedObject():
                eval_data["success"] = True
                successfull += 1
            val_data[data["name"]] = eval_data

        return successfull, val_data

    def get_picking_epsiode_file(self,path="../GDrive/testdata/*_1.json"):
        files = glob.glob(path)
        return random.choice(files)
        
    
    def picking_task_rl(self,\
                        chosen_file:str,\
                        reward_fn:RoboRewardFnMixin,\
                        agent:RobotAgent,\
                        train:bool=False):
        with open(chosen_file, "r") as fh:
            data = json.load(fh)
        
        self.run_pick_rl_episodes(data,reward_fn,agent,train=train)

    def get_core_config(self,reward_fn:RoboRewardFnMixin,text_context:str=None):
        ENV = "Picking Task"
        reward_ob = dict(
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
            num_steps = reward_fn.max_traj_length,
            **reward_ob
        )
        return config
    
    def make_logger(self,reward_fn:RoboRewardFnMixin,text_context:str,train:bool=False):
        config = self.get_core_config(reward_fn,text_context=text_context)
        # $ Create Logger. 
        if self.api_token is not None:
            neptune.init(self.project_name,api_token=self.api_token)
        else:
            neptune.init(self.project_name,backend=neptune.OfflineBackend())

        exp_name = f'{self.experiment_name}'
        if not train:
            exp_name = f'{exp_name}-test'
        neptune.create_experiment(name=exp_name, params=config)
    
    
    def _reset_rl_environment(self,data:dict):
        gt_trajectory = np.asarray(data["trajectory"])
        self._resetEnvironment()
        self._createEnvironment(data["ints"], data["floats"])
        self._setRobotInitial(gt_trajectory[0, :])
        # $ Take step in stimultor when you resent environment
        self.pyrep.step()
        # $ todo return Final position of the objects. 
        return gt_trajectory

    def _get_rl_sim_state(self):
        data = {}
        array = self._getSimulatorState()
        data["joint_robot_position"] = array[0:6]
        data["joint_robot_velocity"] = array[6:12]
        data["tcp_position"] = array[12:15]
        data["tcp_orientation"] = array[15:18]
        data["tcp_linear_velocity"] = array[18:21]
        data["tcp_angular_veloctiy"] = array[21:24]
        data["tcp_target_position"] = array[24:27]
        data["tcp_target_orientation"] = array[27:30]
        data["joint_gripper"] = [array[30]]
        data["joint_gripper_velocity"] = [array[31]]
        return data
    
    def _get_rl_env_state(self):
        joint_state = self._get_rl_sim_state()
        robot_position =  joint_state['joint_robot_position']
        joint_gripper = joint_state['joint_gripper']
        return robot_position,joint_gripper,joint_state

    @staticmethod
    def get_object_target_position(ints,floats,targetid):
        numcup,numbowls = ints[0],ints[1]
        object_ids = ints[2:]
        final_objects = None
        assert len(floats) == len(object_ids)*3
        for idx,posidx in zip(range(int(len(floats)/3)),range(0,len(floats),3)):
            if object_ids[idx] == targetid:
                x,y,_ = floats[posidx:posidx+3]
                return (x,y)
            
        raise Exception("No Object Fouund from ints and floats")
            

    def _take_action_jv(self,joint_state,action):
        joint_vels = action[:6]    
        joint_pos = (joint_vels/20) + joint_state['joint_robot_position']
        gripper_pos = 1 if joint_vels[-1] > 0 else 0
        action_step = [*joint_pos,gripper_pos]
        self._setJointVelocityFromTarget_Direct(action_step)
        self.pyrep.step()

    def _is_terminal_state_pick_task(self):
        return self._graspedObject()
        
    
    def run_pick_rl_episodes(self,data,reward_fn:RoboRewardFnMixin,agent:RobotAgent,train:bool=True):
        try:
            text_input = data["voice"]
            # $ make logger to run episode 
            self.make_logger(reward_fn,text_input,train=train)
            # $ Set size of traj based on RW fn Constraints
            max_trajectory_size = reward_fn.max_traj_length
            # $ Get Metadata about text
            input_text_meta = self._getLanguateInformation(text_input, 1)

            final_position = self.get_object_target_position(
                data['ints'],data['floats'],data['target/id']
            )
            for episode in range(self.num_eps):
                # $ Reset environment
                print(f"Starting Episode {episode}  With Steps {max_trajectory_size}")
                gt_trajectory = self._reset_rl_environment(data)
                video_frames = []
                states = []
                actions = []
                reached_goal = False
                for step in range(max_trajectory_size):
                    # Get State of Robot
                    _,\
                    _,\
                    joint_state = self._get_rl_env_state()
                    video_frames.append(
                        self._getCameraImage()
                    )    
                    joint_state['final_postion'] = final_position
                    # $ predict action using policy IS JOINT velocity (JV)
                    action = agent.act(joint_state)
                    # $ JV is in rads/sec and PyRep runs at 20Hz so we can just simply divide by 20 and then add that to joint positions. 
                    # $ run action predicted by policy
                    actions.append(action)
                    self._take_action_jv(joint_state,action)
                    # $ append state to trajectory. 
                    states.append(joint_state)
                    if not self.update_episodically:
                        if step % self.update_frequency  == 0 and episode!=0:
                            agent.update()
                    # $ measure if the current state is teminal state 
                    if self._is_terminal_state_pick_task():
                        # $ if terminal state end episode
                        reached_goal = True
                        break
                if episode % self.video_save_freq ==0:
                    self._save_episode_video(episode,video_frames)
                episode_perf_stats = self._getTargetPosition(data)
                # $ run reward function and get reward values.
                reward = reward_fn.get_rewards(
                    text_input,dict(
                        image_sequence = video_frames,
                        trajectory = states
                    )
                )
                norm_reward = RewardNormalizer.normalize_reward(reward,\
                                                max_v=self.reward_max,\
                                                min_v=self.reward_min)
                scaledreward = norm_reward * self.reward_scaleup
                ammt_rw = RewardNormalizer.ammortized_rw(scaledreward, len(states))
                neptune.log_metric('is_success',episode,y=reached_goal)
                neptune.log_metric('reward', episode, y=reward)
                neptune.log_metric('scaled_reward', episode, y=scaledreward)
                neptune.log_metric('distance_to_obj',episode,y=episode_perf_stats['distance'])
                # $ update the policy
                if train:
                    self._observe_and_update_policy(agent,states,actions,ammt_rw,reached_goal)
        except KeyboardInterrupt as e:
            print("Keyboard Interrupt Occured")
            self.shutdown()
            neptune.log_artifact(self.video_save_dir)
            neptune.stop()
            return 
        
        neptune.log_artifact(self.video_save_dir)
    
    def _save_episode_video(self,episode:int,video_array:np.array,freq=20):
        width,height= video_array[0].shape[0], video_array[0].shape[1]
        path = os.path.join(self.video_save_dir,f'episode-{episode}.avi')
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        writer = cv2.VideoWriter(path, fourcc, 20, (640,480))
        for i in range(0,len(video_array),freq):
            x = video_array[i].astype('uint8')
            img= cv2.resize(x,(640,480))
            writer.write(img)
        writer.release()

    def _observe_and_update_policy(self,agent:RobotAgent,states,actions,rewards,reached_goal):
        # $ Make agent update its states and then update its policies in the training step.
        for idx, rw_tup in enumerate(zip(states, actions,rewards)):
            state,action, rw = rw_tup
            done=False
            if idx == len(states)-1:
                done = True
            agent.observe(state,action, rw.item(), done)
        agent.update()
        agent.reset()

    def evalDirect(self, runs, path="../GDrive/testdata/*_1.json"):
        files = glob.glob(path)
        files = files[:runs]
        files = [f[:-6] for f in files]
        data = {}
        s_p1, e_data = self.valPhase1(files)
        data["phase_1"] = e_data

        with open("val_result.json", "w") as fh:
            json.dump(data, fh)

    def _generateEnvironment(self):
        def genPosition(prev):
            px = 0
            py = 0
            done = False
            while not done:
                done = True
                px = np.random.uniform(-0.9, 0.35)
                py = np.random.uniform(-0.9, 0.35)
                dist = np.sqrt(px**2 + py**2)
                if dist < 0.5 or dist > 0.9:
                    done = False
                for o in prev:
                    if np.sqrt((px - o[0])**2 + (py - o[1])**2) < 0.25:
                        done = False
                if px > 0 and py > 0:
                    done = False
                angle = -45
                r_px = px * np.cos(np.deg2rad(angle)) + \
                    py * np.sin(np.deg2rad(angle))
                r_py = py * np.cos(np.deg2rad(angle)) - \
                    px * np.sin(np.deg2rad(angle))
                if r_py > 0.075:
                    done = False
            return [px, py]
        self._setRobotJoints(np.deg2rad(DEFAULT_UR5_JOINTS))

        ncups = np.random.randint(1, 3)
        nbowls = np.random.randint(ncups, 5)
        bowls = np.random.choice(20, size=nbowls, replace=False) + 1
        cups = np.random.choice(3, size=ncups, replace=False) + 1
        ints = [nbowls, ncups] + bowls.tolist() + cups.tolist()
        floats = []

        prev = []
        for i in range(nbowls + ncups):
            prev.append(genPosition(prev))
            floats += prev[-1]
            if i < nbowls and bowls[i] > 10:
                floats += [np.random.uniform(-math.pi/4.0,  math.pi/4.0)]
            else:
                floats += [0.0]

        self._createEnvironment(ints, floats)
        return ints, floats

    def simplifyVoice(self, voice):
        simple = []
        for word in voice.split(" "):
            if word in self.voice.basewords.keys():
                simple.append(self.voice.basewords[word])
        return " ".join(simple)

    def _sigint_handler(self,x,y):
        print("Sigint Received")
        print(x,y)
        neptune.log_artifact(self.video_save_dir)
        self.shutdown()
        neptune.stop()
        exit()
# if __name__ == "__main__":
#     sim = Simulator()
#     sim.evalDirect(runs=NUM_TESTED_DATA)
#     sim.shutdown()
