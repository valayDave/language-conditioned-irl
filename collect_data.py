"""
Script Purpose : Create the picking task demonstrations with Noise. 
"""
# Refactor from @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

from pyrep import PyRep
from pyrep.objects.vision_sensor import VisionSensor
from ros1compat import kdl_urdf_parser as kdl_parser_py
import PyKDL as kdl
import numpy as np
import random
import math
import scipy
from scipy.interpolate import interp1d
import hashids
import time
import json
from language_conditioned_rl.models.robotics.utils import Voice
from joblib import Parallel, delayed
import os
import click

# How many processes shold collect data in parallel? 
# A good measure is to put the number of CPU cores you have (Note that each process needs ~1GB of RAM also)
PROCESSES           = 1
# How many demonstrations (picking and pouring) should each process collect? 
SAMPLES_PER_PROCESS = 2
# Ever n demonstrations, VRep will be restarted entirely, not just the simulation. You don't need to change this
RESET_EACH          = 20
# If you run more than 1 process, you should run VRep headless
VREP_HEADLESS       = True
# Default position of the UR5 robot. You do not need to change this
DEFAULT_UR5_JOINTS  = [105.0, -30.0, 120.0, 90.0, 60.0, 90.0]
# Path to the UR5 URDF file
ROBOT_URDF          = "./GDrive/ur5_robot.urdf"
# General speed of the robot. Lower values will increase the robot's movement speed
TGEN_SPEED_FACTOR   = 150
# Height at which to grasp the cups. You do not need to change this.
GRASP_HEIGHT        = 0.115
# Output directory of the collected data
DATA_PATH           = "./freshly_collected_data/dataset/"
# Where to find the VRep scene file. This has to be an absolute path. 
VREP_SCENE          = "./GDrive/NeurIPS2020.ttt"
VREP_SCENE          = os.path.join(os.getcwd() , VREP_SCENE)

class SimulatorState(object):
    def __init__(self, array):
        self.array = array
        self.fromArray(array)
    
    def fromArray(self, array):
        if len(array) != 32:
            print("Expected state of length 32, but got state of length " + str(len(array)))
        self.array = array
        self.data = {}
        self.data["joint_robot_position"]   = array[0:6]
        self.data["joint_robot_velocity"]   = array[6:12]
        self.data["tcp_position"]           = array[12:15]
        self.data["tcp_orientation"]        = array[15:18]
        self.data["tcp_linear_velocity"]    = array[18:21]
        self.data["tcp_angular_veloctiy"]   = array[21:24]
        self.data["tcp_target_position"]    = array[24:27]
        self.data["tcp_target_orientation"] = array[27:30]
        self.data["joint_gripper"]          = [array[30]]
        self.data["joint_gripper_velocity"] = [array[31]]
    
    def toArray(self):
        return self.array

class Robot(object):
    def __init__(self):
        kdl_tree           = kdl_parser_py.treeFromFile(ROBOT_URDF)[1]
        self.offsets       = np.deg2rad([90.0, -90.0, 0.0, -90.0, 0.0, 0.0])
        self.base_joints   = np.deg2rad(DEFAULT_UR5_JOINTS)
        transform          = kdl_tree.getChain("world", "tcp")

        self.kdl_fk        = kdl.ChainFkSolverPos_recursive(transform)
        self.kdl_ik        = kdl.ChainIkSolverPos_LMA(transform)
        self.kdl_input     = kdl.JntArray(transform.getNrOfJoints())
        self.kdl_output    = kdl.JntArray(transform.getNrOfJoints())
        self.num_joints    = transform.getNrOfJoints()
        if self.num_joints != len(self.base_joints):
            print("Extracted chain has " + str(self.num_joints) + " joints, but " + len(self.base_joints) + " were expected.")
    
    def getJointAnglesFromCurrent(self, loc, rot, current):
        self._reset(current)
        return self._getJointAngles(loc, rot)        

    def getJointAngles(self, loc, rot):
        self._reset()
        return self._getJointAngles(loc, rot)
    
    def getTcpFromAngles(self, angles):
        self._reset(angles)
        goal = kdl.Frame()
        self.kdl_fk.JntToCart(self.kdl_input, goal)
        return goal

    def _reset(self, target=None):
        if target is None:
            target = self.base_joints

        for joint_idx in range(self.num_joints):
            self.kdl_input[joint_idx] = target[joint_idx] + self.offsets[joint_idx]

    def _getJointAngles(self, loc, rot):
        goal   = kdl.Frame()
        goal.p = kdl.Vector(*loc)
        goal.M = kdl.Rotation.EulerZYX(rot[0], rot[1], rot[2],)

        self.kdl_ik.CartToJnt(self.kdl_input, goal, self.kdl_output)
        return [v - self.offsets[i] for i, v in enumerate(self.kdl_output)]

def genPosition(prev):
    px = 0
    py = 0
    done = False        
    while not done:
        done = True        
        px = random.uniform(-0.9, 0.35)
        py = random.uniform(-0.9, 0.35)
        dist = np.sqrt(px**2 + py**2)
        if dist < 0.5 or dist > 0.9:
            done = False
        for o in prev:
            if np.sqrt((px - o[0])**2 + (py - o[1])**2) < 0.25:
                done = False
        if px > 0 and py > 0:
            done = False
        angle = -45
        r_px  = px * np.cos(np.deg2rad(angle)) + py * np.sin(np.deg2rad(angle))
        r_py  = py * np.cos(np.deg2rad(angle)) - px * np.sin(np.deg2rad(angle))
        if r_py > 0.075:
            done = False
    return [px, py]

def _getCameraImage(camera):
    rgb_obs = camera.capture_rgb()
    rgb_obs = (np.asarray(rgb_obs) * 255).astype(dtype=np.uint8)
    rgb_obs = np.flip(rgb_obs, (2))
    return rgb_obs

def _getSimulatorState(pyrep):
    _, s, _, _ = pyrep.script_call(function_name_at_script_name="getState@control_script",
                                    script_handle_or_type=1,
                                    ints=(), floats=(), strings=(), bytes="")
    state = SimulatorState(s)
    return state

def _setRobotJoints(pyrep, joints):
        result = pyrep.script_call(function_name_at_script_name="setRobotJoints@control_script",
                                        script_handle_or_type=1,
                                        ints=(), floats=joints, strings=(), bytes="")

def _setJointVelocityFromTarget(pyrep, joints):
    _, s, _, _ = pyrep.script_call(function_name_at_script_name="setJointVelocityFromTarget@control_script",
                                    script_handle_or_type=1,
                                    ints=(), floats=joints, strings=(), bytes="")

def _setJointVelocityFromTarget_Direct(pyrep, joints):
        _, s, _, _ = pyrep.script_call(function_name_at_script_name="setJointVelocityFromTarget_Direct@control_script",
                                       script_handle_or_type=1,
                                       ints=(), floats=joints, strings=(), bytes="")

def _moveL(robot, current, goal, pad_one=False):
    g_pos, g_rot   = goal
    if type(g_pos) == np.ndarray:
        g_pos = g_pos.tolist()
    if type(g_rot) == np.ndarray:
        g_rot = g_rot.tolist()
    if type(current) == np.ndarray:
        current = current.tolist()
    padding        = 1.0 if pad_one else 0.0
    kdl_frame      = robot.getTcpFromAngles(current)
    s_pos          = [kdl_frame.p.x(), kdl_frame.p.y(), kdl_frame.p.z()]
    s_rot          = kdlFrameToRot(kdl_frame).tolist()
    distance       = np.sqrt( np.power(s_pos[0] - g_pos[0], 2) + np.power(s_pos[1] - g_pos[1], 2) + np.power(s_pos[2] - g_pos[2], 2) )
    steps          = max(2, int(np.ceil(distance * TGEN_SPEED_FACTOR))) # at least 2 steps
    tcp_trj        = np.linspace(s_pos + s_rot, g_pos + g_rot, num=steps, endpoint=True, axis=0)
    joint_trj      = np.zeros((steps, 7), dtype=np.float32)
    joint_trj[0,:] = current + [padding]
    for i in range(1,steps,1):
        joint_trj[i] = robot.getJointAnglesFromCurrent(loc=tcp_trj[i,:3], rot=tcp_trj[i,3:], current=joint_trj[i-1,:6]) + [padding]
    return joint_trj

def _moveJ(robot, current, goal, pad_one=False):
    g_pos, g_rot   = goal
    if type(g_pos) == np.ndarray:
        g_pos = g_pos.tolist()
    if type(g_rot) == np.ndarray:
        g_rot = g_rot.tolist()
    if type(current) == np.ndarray:
        current = current.tolist()
    kdl_frame      = robot.getTcpFromAngles(current)
    s_pos          = [kdl_frame.p.x(), kdl_frame.p.y(), kdl_frame.p.z()]
    goal_joints    = robot.getJointAnglesFromCurrent(loc=g_pos, rot=g_rot, current=current)
    distance       = np.sqrt( np.power(s_pos[0] - g_pos[0], 2) + np.power(s_pos[1] - g_pos[1], 2) + np.power(s_pos[2] - g_pos[2], 2) )
    steps          = max(2, int(np.ceil(distance * TGEN_SPEED_FACTOR))) # at least 2 steps
    joint_trj      = np.linspace(current, goal_joints, num=steps, endpoint=True, axis=0)
    padding        = np.ones((joint_trj.shape[0], 1)) if pad_one else np.zeros((joint_trj.shape[0], 1))
    joint_trj      = np.hstack((joint_trj, padding))
    return joint_trj

def createEnvironment(pyrep):
    _setRobotJoints(pyrep, np.deg2rad(DEFAULT_UR5_JOINTS))

    ncups  = np.random.randint(1,3)
    nbowls = np.random.randint(ncups,5)
    bowls  = np.random.choice(20, size=nbowls, replace=False) + 1
    cups   = np.random.choice(3, size=ncups, replace=False) + 1
    ints   = [nbowls, ncups] + bowls.tolist() + cups.tolist()
    floats = []

    prev   = []
    for i in range(nbowls + ncups):
        prev.append(genPosition(prev))
        floats += prev[-1]
        if i < nbowls and bowls[i] > 10:
            floats += [random.uniform(-math.pi/4.0,  math.pi/4.0)]
        else:
            floats += [0.0]

    result = pyrep.script_call(
        function_name_at_script_name="generateScene@control_script",
        script_handle_or_type=1,
        ints=ints, 
        floats=floats, 
        strings=(), 
        bytes=""
    )
    return ints, floats

def kdlFrameToRot(frame):
    rotation = frame.M.GetEulerZYX()
    rotation = np.asarray(rotation, dtype=np.float32)
    # Weird fix...
    rotation[0] += np.deg2rad(360.0) if rotation[0] < 0 else 0.0
    return rotation

def _getObjectInfo(ints, floats, t_id, iscup):
    nbowls        = ints[0]
    ncups         = ints[1]
    bowl_ids      = ints[2:2+nbowls]
    cup_ids       = ints[2+nbowls:2+nbowls+ncups]

    index = None
    if iscup:
        index = np.argwhere(np.asarray(cup_ids)==t_id)[0,0] + nbowls
    else:
        index = np.argwhere(np.asarray(bowl_ids)==t_id)[0,0]
    
    data = floats[index * 3 : index * 3 + 3]
    return data

def _graspClosestObject():
    result = pyrep.script_call(
        function_name_at_script_name="graspObject@control_script",
        script_handle_or_type=1,
        ints=[], 
        floats=[], 
        strings=(), 
        bytes=""
    )

def getCollisionWaypoints(robot, start, goal):
    x1 = start[0]
    y1 = start[1]
    x2 = goal[0]
    y2 = goal[1]

    distance = np.abs(x2*y1 - y2*x1) / np.sqrt(np.power(y2-y1,2) + np.power(x2-x1,2))

    if distance < 0.15:
        kdl_frame       = robot.getTcpFromAngles(np.deg2rad(DEFAULT_UR5_JOINTS))
        original_rot    = kdlFrameToRot(kdl_frame)
        original_pos    = [kdl_frame.p.x(), kdl_frame.p.y(), kdl_frame.p.z()]
        original_pos[2] = goal[2]
        return [("J", original_pos, original_rot)]
    else:
        return []

def _generatePouring(robot, current, rot):
    trj          = np.tile(np.expand_dims(current, 0),reps=[int(np.floor(75*rot)),1])
    steps        = trj.shape[0]
    target       = trj[0,5] + rot
    kdl_frame    = robot.getTcpFromAngles(current)
    current_pos  = [kdl_frame.p.x(), kdl_frame.p.y(), kdl_frame.p.z()]

    inter    = interp1d([0.0, 0.4, 0.6, 1.0], [current_pos[2], current_pos[2] - 0.075, current_pos[2] - 0.075, current_pos[2]], kind='cubic')
    zpos     = inter(np.linspace(0.0, 1.0, num=steps, endpoint=True))

    for i in range(steps):
        kdl_frame = robot.getTcpFromAngles(trj[i,:6].tolist())
        step_pos  = [kdl_frame.p.x(), kdl_frame.p.y(), kdl_frame.p.z()]
        step_rot  = kdlFrameToRot(kdl_frame)

        step_pos[2] = zpos[i]
        trj[i,:6]   = robot.getJointAnglesFromCurrent(loc=step_pos, rot=step_rot, current=trj[0,:6])

    inter    = interp1d([0.0, 0.4, 0.6, 1.0], [trj[0,5], target, target, trj[-1,5]], kind='cubic')
    trj[:,5] = inter(np.linspace(0.0, 1.0, num=steps, endpoint=True))

    return trj

def _setupTask(phase, env, robot, current):
    task           = {}
    task["amount"] = np.random.choice([180, 110])
    ints, floats   = env
    nbowls         = ints[0]
    ncups          = ints[1]
    bowl_ids       = ints[2:2+nbowls]
    cup_ids        = ints[2+nbowls:2+nbowls+ncups]

    waypoints    = []
    kdl_frame    = robot.getTcpFromAngles(current)
    current_rot  = kdlFrameToRot(kdl_frame)
    current_pos  = [kdl_frame.p.x(), kdl_frame.p.y(), kdl_frame.p.z()]

    kdl_frame    = robot.getTcpFromAngles(np.deg2rad(DEFAULT_UR5_JOINTS))
    original_rot = kdlFrameToRot(kdl_frame)
    original_pos = [kdl_frame.p.x(), kdl_frame.p.y(), kdl_frame.p.z()]
    
    if phase == 0:  
        target    = np.random.choice(cup_ids)
        task["target/id"] = target
        task["target/type"] = "cup"
        target    = _getObjectInfo(ints, floats, target, iscup=True)
        target[2] = GRASP_HEIGHT
        rot       = [r for r in current_rot]
        rot[0]    += _calculateAngle(target[0], target[1])
        waypoints.append(("L", [t for t in target], rot))

        norm        = np.linalg.norm(target[:2], ord=2)
        factor      = norm / (norm * 0.85)
        target2     = [t for t in target]
        target2[0] /= factor
        target2[1] /= factor
        waypoints.insert(0, ("J", target2, rot))
        waypoints.append(("G", None, None))
        target3     = [t for t in target2]
        target3[2] += 0.10
        waypoints.append(("L", target3, rot))
    if phase == 1:
        target    = np.random.choice(bowl_ids)
        task["target/id"] = target
        task["target/type"] = "bowl"
        target    = _getObjectInfo(ints, floats, target, iscup=False)
        target[2] = current_pos[2]
        target    = adjustTargetForPouring(target[0], target[1], target[2])
        rot       = [r for r in original_rot]
        rot[0]    += _calculateAngle(target[0], target[1])
        waypoints += getCollisionWaypoints(robot, current_pos, target)
        waypoints.append(("J", target, rot))
        waypoints.append(("P", None, np.deg2rad(task["amount"])))
        waypoints.append(("I", 40, None))

    trajectory       = np.zeros((1,7), dtype=np.float32)
    trajectory[0,:6] = current
    grasp_active     = False if phase == 0 else True
    if grasp_active:
        trajectory[0,-1] = 1.0

    for i, wp in enumerate(waypoints):
        if wp[0] == "L": # Move linear in tool space
            part = _moveL(robot, trajectory[-1,:6], wp[1:], pad_one=grasp_active)
        elif wp[0] == "J": # Move linear in joint space
            part = _moveJ(robot, trajectory[-1,:6], wp[1:], pad_one=grasp_active)
        elif wp[0] == "G": # Close gripper
            part = np.tile(trajectory[-1,:],reps=[30,1])
            part[:,-1] = 1.0
            grasp_active = True
        elif wp[0] == "P": # Do pouring motion
            part = _generatePouring(robot, trajectory[-1,:], wp[2])
        elif wp[0] == "I": # Idle a little
            part = np.tile(np.expand_dims(trajectory[-1,:], 0),reps=[wp[1],1])
            
        trajectory = np.vstack((trajectory, part))
        
    
    task["trajectory"] = trajectory

    return task

def _calculateAngle(x, y):
    # Rotate:
    alpha = np.deg2rad(-135)
    point = np.dot( np.asarray([[np.cos(-alpha), np.sin(-alpha)],[-np.sin(-alpha), np.cos(-alpha)]]), np.asarray([x,y]) )

    angle = np.arctan(point[0] / point[1]) * -1.0
    return angle

def adjustTargetForPouring(x, y, z):
    for i in range(10):
        alpha    = np.deg2rad((i+1)/2.0)
        point    = np.dot( np.asarray([[np.cos(-alpha), np.sin(-alpha)],[-np.sin(-alpha), np.cos(-alpha)]]), np.asarray([x,y]) )
        distance = np.sqrt( np.power(x-point[0],2) + np.power(y-point[1],2) )
        if np.sqrt( np.power(x-point[0],2) + np.power(y-point[1],2) ) >= 0.025:
            break
    return [point[0], point[1], z]

def saveTaskToFile(path, task):
    for k, v in task.items():
        if type(v) == np.ndarray:
            task[k] = v.tolist()
        elif type(v) == np.int64:
            task[k] = int(v)
        elif type(v) in [str, list, int,bool]:
            pass
        else:
            print("Serializing unhandled type", k, type(v))

    with open(path, "w") as fh:
        json.dump(task, fh)

def _generateVoice(voice, task,add_noise):
    if not add_noise:
        sentence =  voice.generateSentence(task)
        print("-> " + sentence)
    else:
        sentence =  f"try to {voice.generateSentence(task)}"
        print("-> " + sentence)
    return sentence

def collectSingleSample(pyrep):
    robot = Robot()
    voice = Voice(load=False)
    print("Pyrep About to Start")
    rgb_camera = VisionSensor("kinect_rgb_full")
    pyrep.start()
    print("Pyrep About Started")

    frame       = 0
    done        = False
    task        = None
    environment = None
    phase       = 0 
    trj_step    = 0
    hid         = hashids.Hashids()
    task_name   = hid.encode(int(time.time() * 1000000))
    video_frames = []

    while not done:
        print("Running Loop Bro!")
        state     = _getSimulatorState(pyrep)
        if frame == 0:
            environment = createEnvironment(pyrep)
        elif task is None:
            task               = _setupTask(phase, environment, robot, state.data["joint_robot_position"])
            task["name"]       = task_name
            task["phase"]      = phase
            task["image"]      = _getCameraImage(rgb_camera)
            task["ints"]       = environment[0]
            task["floats"]     = environment[1]
            task["state/raw"]  = []
            task["state/dict"] = []
            task["voice"]      = _generateVoice(voice, task)
        else:
            video_frames.append(_getCameraImage(rgb_camera))
            task["state/raw"].append(state.toArray())
            task["state/dict"].append(state.data)
            try:
                angles    = task["trajectory"][trj_step,:]
                trj_step += 1
            except IndexError:
                
                angles   = task["trajectory"][-1,:]
                phase   += 1
                name     = task_name + "_" + str(phase) + ".json"
                saveTaskToFile(DATA_PATH + name, task)
                save_episode_video(random.randint(0,10000),video_frames)
                video_frames = []
                task     = None
                trj_step = 0
                if phase == 2:
                    done = True
            _setJointVelocityFromTarget(pyrep, angles)

        pyrep.step()        
        frame += 1

    pyrep.stop()


def save_episode_video(file_name,video_array:np.array,freq=3,video_dir='./freshly_collected_data/videos/'):
    import cv2
    width,height= video_array[0].shape[0], video_array[0].shape[1]
    path = os.path.join(video_dir,f'{file_name}.avi')
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter(path, fourcc, 20, (640,480))
    for i in range(0,len(video_array),freq):
        x = video_array[i].astype('uint8')
        img= cv2.resize(x,(640,480))
        writer.write(img)
    writer.release()

def _setup_picking_task(phase, env, robot, current,add_noise=True,noise_dampning_factor=0.3):
    task           = {}
    task["amount"] = np.random.choice([180, 110])
    ints, floats   = env
    nbowls         = ints[0]
    ncups          = ints[1]
    bowl_ids       = ints[2:2+nbowls]
    cup_ids        = ints[2+nbowls:2+nbowls+ncups]

    waypoints    = []
    kdl_frame    = robot.getTcpFromAngles(current)
    current_rot  = kdlFrameToRot(kdl_frame)
    current_pos  = [kdl_frame.p.x(), kdl_frame.p.y(), kdl_frame.p.z()]

    kdl_frame    = robot.getTcpFromAngles(np.deg2rad(DEFAULT_UR5_JOINTS))
    original_rot = kdlFrameToRot(kdl_frame)
    original_pos = [kdl_frame.p.x(), kdl_frame.p.y(), kdl_frame.p.z()]
    
    target    = np.random.choice(cup_ids)
    task["target/id"] = target
    task["target/type"] = "cup"
    task['add_noise'] = add_noise
    target    = _getObjectInfo(ints, floats, target, iscup=True)
    target[2] = GRASP_HEIGHT
    rot       = [r for r in current_rot]
    rot[0]    += _calculateAngle(target[0], target[1])
    waypoints.append(("L", [t for t in target], rot))

    norm        = np.linalg.norm(target[:2], ord=2)
    factor      = norm / (norm * 0.85)
    target2     = [t for t in target]
    target2[0] /= factor
    target2[1] /= factor
    waypoints.insert(0, ("J", target2, rot))
    waypoints.append(("G", None, None))
    target3     = [t for t in target2]
    target3[2] += 0.10
    waypoints.append(("L", target3, rot))

    trajectory       = np.zeros((1,7), dtype=np.float32)
    trajectory[0,:6] = current
    grasp_active     = False if phase == 0 else True
    if grasp_active:
        trajectory[0,-1] = 1.0

    # The waypoints are planned prehand with linear movements or grapsing status. 
    # The preplanned waypoints are converted to joint position trajectory. 
    for i, wp in enumerate(waypoints):
        if wp[0] == "L": # Move linear in tool space
            part = _moveL(robot, trajectory[-1,:6], wp[1:], pad_one=grasp_active)
        elif wp[0] == "J": # Move linear in joint space
            part = _moveJ(robot, trajectory[-1,:6], wp[1:], pad_one=grasp_active)
        elif wp[0] == "G": # Close gripper
            part = np.tile(trajectory[-1,:],reps=[30,1])
            part[:,-1] = 1.0
            grasp_active = True
        elif wp[0] == "P": # Do pouring motion
            part = _generatePouring(robot, trajectory[-1,:], wp[2])
        elif wp[0] == "I": # Idle a little
            part = np.tile(np.expand_dims(trajectory[-1,:], 0),reps=[wp[1],1])
        
        NOISE_DAMPING_FACTOR = noise_dampning_factor
        if add_noise:
            part_len,part_dims = part.shape
            traj_p = part[:,:-1]
            gripper_part = part[:,-1:]
            part_mean = traj_p.mean(axis=1)
            part_std = traj_p.std(axis=1)
            noise_min = part_mean.mean() - 2*part_std.mean()
            noise_max = part_mean.mean() - 1*part_std.mean()
            sampl = np.random.uniform(low=noise_min, high=noise_max, size=(part_len,part_dims-1))
            sampl = sampl*NOISE_DAMPING_FACTOR
            traj_p = traj_p + sampl
            part = np.concatenate((traj_p,gripper_part),axis=1)
            print("Part Modified with Noise of Mean",sampl.mean())

        trajectory = np.vstack((trajectory, part))
    
    task["trajectory"] = trajectory

    return task

def create_picking_episodes(pyrep,add_noise=False):
    robot = Robot()
    voice = Voice(load=False)
    rgb_camera = VisionSensor("kinect_rgb_full")
    pyrep.start()
    print("Pyrep Picking Task About To Start")
    frame       = 0
    done        = False
    task        = None
    environment = None
    phase       = 0 
    trj_step    = 0
    hid         = hashids.Hashids()
    task_name   = hid.encode(int(time.time() * 1000000))
    print("Running Loop Bro!")
    state     = _getSimulatorState(pyrep)
    environment = createEnvironment(pyrep)
    task               = _setup_picking_task(phase, environment, robot, state.data["joint_robot_position"],add_noise=add_noise,noise_dampning_factor=random.uniform(0.1,1))
    task["name"]       = task_name
    task["phase"]      = phase
    task["image"]      = _getCameraImage(rgb_camera)
    task["ints"]       = environment[0]
    task["floats"]     = environment[1]
    task["state/raw"]  = []
    task["state/dict"] = []
    task["voice"]      = _generateVoice(voice, task,add_noise)
    video_frames = []
    while not done:
        task["state/raw"].append(state.toArray())
        task["state/dict"].append(state.data)
        video_frames.append(
            _getCameraImage(rgb_camera)
        )
        try:
            angles    = task["trajectory"][trj_step,:]
            trj_step += 1
        except IndexError:
            angles   = task["trajectory"][-1,:]
            name     = task_name + "_" + str(phase) 
            js_name = f"{name}.json"
            save_episode_video(name,video_frames)
            saveTaskToFile(os.path.join(DATA_PATH,js_name), task)
            done = True
        _setJointVelocityFromTarget(pyrep, angles)
        pyrep.step()        
        frame += 1
    pyrep.stop()



def run(per_process_demos,without_noise):
    print("Starting BRO!")
    pyrep = None
    try:
        print("Collecting BRO!")
        pyrep = PyRep()
        print("Launched Pyrep!")
        pyrep.launch(VREP_SCENE, headless=VREP_HEADLESS)
        for i in range(per_process_demos):
            noise_bool = False
            if not without_noise:
                if i%2 != 0:
                    noise_bool = True
            create_picking_episodes(pyrep,add_noise=noise_bool)
        if pyrep is not None:
            pyrep.shutdown()
    except KeyboardInterrupt as e:
        if pyrep is not None:
            pyrep.shutdown()
        print("Shutdown Stuff!")




@click.command(help='Collect Demostrations For the Picking Task')
@click.option('--num-demos',default=10,help='Number of Demostrations to Create From Simulator')
@click.option('--without-noise',is_flag=True,help='Creates demostrations without noise')
@click.option('--num-procs',default=2,help='Number of Processes to create For Making Demostrations')
def make_picking_task(num_demos,without_noise,num_procs):
    per_process_demos = int(num_demos/num_procs)
    assert per_process_demos > 0
    Parallel(n_jobs=num_procs)(delayed(run)(per_process_demos,without_noise) for i in range(num_procs))
    


"""
TODO : 
    - Figure the Number Of samples to generate for the Trajectories. 
"""
if __name__ == "__main__":
   make_picking_task()