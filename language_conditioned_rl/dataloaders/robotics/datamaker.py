'''
This file holds all the core functions to create h5py dataset so that it can be used for fast data loading. 

Just creating index's and directly using the dataset with unzipping and unpacking is too slow. 

# Todo : Unpack entire files into h5py dataset.

# todo : Once the entire data is unpacked create the Mapping dataset which helps with the control parameters of the mapping function which makes this dataset.

'''
""" 
"""

from typing import List
import h5py
from torchvision import transforms
from dataclasses import dataclass
import torch
from PIL import Image
import base64
from torch.utils.data.dataset import Dataset
import numpy as np
import json
import zlib
import os
import random
import pandas
from metaflow import parallel_map
from .dataset import MAX_TRAJ_LEN,\
    USE_CHANNELS,\
    IMAGE_SIZE,\
    MAX_TEXT_LENGTH,\
    DISCRETE_CHANNELS,\
    load_json_from_file,\
    MAX_TRAJ_LEN,\
    is_present,\
    GROUPNAMES,\
    get_padding,\
    RoboDataUtils
    

import gc
import logging

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')



def create_logger(logger_name: str, level=logging.INFO):
    custom_logger = logging.getLogger(logger_name)
    ch1 = logging.StreamHandler()
    ch1.setLevel(level)
    ch1.setFormatter(formatter)
    custom_logger.addHandler(ch1)
    custom_logger.setLevel(level)
    return custom_logger


def get_hd5_base_sizes(img_size=IMAGE_SIZE,
                       batch_size=None,
                       max_traj_len=MAX_TRAJ_LEN,
                       max_txt_len=MAX_TEXT_LENGTH
                       ):
    return {
        'image': (batch_size, 3, *img_size),
        'joint_gripper': (batch_size, max_traj_len),
        'joint_gripper_velocity': (batch_size, max_traj_len, 1),
        'joint_robot_position': (batch_size, max_traj_len, 6),
        'joint_robot_velocity': (batch_size, max_traj_len, 6),
        'tcp_angular_veloctiy': (batch_size, max_traj_len, 3),
        'tcp_linear_velocity': (batch_size, max_traj_len, 3),
        'tcp_orientation': (batch_size, max_traj_len, 3),
        'tcp_position': (batch_size, max_traj_len, 3),
        'tcp_target_orientation': (batch_size, max_traj_len, 3),
        'tcp_target_position': (batch_size, max_traj_len, 3),
        'text': (batch_size, max_txt_len)
    }


def get_hd5_mask_sizes(batch_size=None,
                       max_traj_len=MAX_TRAJ_LEN,
                       max_txt_len=MAX_TEXT_LENGTH
                       ):
    return {
        'joint_gripper': (batch_size, max_traj_len),
        'joint_gripper_velocity': (batch_size, max_traj_len),
        'joint_robot_position': (batch_size, max_traj_len),
        'joint_robot_velocity': (batch_size, max_traj_len),
        'tcp_angular_veloctiy': (batch_size, max_traj_len),
        'tcp_linear_velocity': (batch_size, max_traj_len),
        'tcp_orientation': (batch_size, max_traj_len),
        'tcp_position': (batch_size, max_traj_len),
        'tcp_target_orientation': (batch_size, max_traj_len),
        'tcp_target_position': (batch_size, max_traj_len),
        'text': (batch_size, max_txt_len)
    }


def get_hd5_dtypes():
    return {
        'image': 'f',
        'joint_gripper': 'i',
        'joint_gripper_velocity': 'f',
        'joint_robot_position': 'f',
        'joint_robot_velocity': 'f',
        'tcp_angular_veloctiy': 'f',
        'tcp_linear_velocity': 'f',
        'tcp_orientation': 'f',
        'tcp_position': 'f',
        'tcp_target_orientation': 'f',
        'tcp_target_position': 'f',
        'text': 'i',
    }

class H5DataCreatorMainDataCreator:

    def __init__(self,
                 save_file_name,
                 sample_pths: List[str],
                 use_channels=USE_CHANNELS,
                 image_resize_dims=IMAGE_SIZE,
                 tokenizer=None,
                 chunk_size=256,
                 max_traj_len=MAX_TRAJ_LEN,
                 max_txt_len=MAX_TEXT_LENGTH):
        assert not is_present(save_file_name)
        assert tokenizer is not None
        self.roboutils = RoboDataUtils(tokenizer=tokenizer,
                                       use_channels=use_channels,
                                       max_traj_len=max_traj_len,
                                       max_txt_len=max_txt_len,
                                       image_resize_dims=image_resize_dims)
        self.file_name = save_file_name
        self.hf = None
        self.seq_grp = None
        self.mask_grp = None
        self.sample_pths = sample_pths
        self.chunk_size = chunk_size
        self.use_channels = use_channels
        self.final_id_list = []

    def build(self):
        logger = create_logger(self.__class__.__name__)
        for json_pth in self.sample_pths:
            logger.info(f"Writing Path {json_pth}")
            loaded_obj = load_json_from_file(json_pth)
            assert 'samples' in loaded_obj
            decompressd_objects = parallel_map(
                lambda x: self.decompress_json(loaded_obj[x]), loaded_obj['samples'])
            # Make the Chunked Tensors from this
            object_tuple = self.roboutils.make_saveable_chunk(
                decompressd_objects)
            sequence_dict, mask_dict, id_list = object_tuple
            if self.hf is None:
                self._create_core_structures(sequence_dict, mask_dict, id_list)
            else:
                self._append_to_file(sequence_dict, mask_dict, id_list)
            del loaded_obj
            del decompressd_objects
            del object_tuple
            gc.collect()
        self.close()

    def _append_to_file(self, sequence_dict, mask_dict, id_list: List[str]):
        if self.hf is None:
            raise Exception('No HDF5 File Open')
        # Add the new Ids of the documents to be appended.
        id_list_dataset = self.id_grp.get(GROUPNAMES.id_list)
        id_list_dataset.resize(len(id_list_dataset) + len(id_list), axis=0)
        id_list_dataset[-len(id_list):] = id_list

        # make channels for each sequence.
        for channel in self.use_channels:
            channel_seq_dataset = self.seq_grp.get(channel)
            old_coll_size = channel_seq_dataset.shape[0]
            new_coll_size = old_coll_size + len(id_list)
            # Resize data based on the new data
            channel_seq_dataset.resize(new_coll_size, axis=0)
            # Add the new Items to the dataset.
            channel_seq_dataset[-len(id_list):] = sequence_dict[channel]

            if channel in mask_dict and mask_dict[channel] is not None:
                channel_msk_dataset = self.mask_grp.get(channel)
                channel_msk_dataset.resize(new_coll_size, axis=0)
                channel_msk_dataset[-len(id_list):] = mask_dict[channel]

    def decompress_json(self, data):
        cache_val = json.loads(zlib.decompress(base64.b64decode(data)))
        return cache_val

    def close(self):
        if self.hf is not None:
            self.hf.close()
            self.hf = None
            self.seq_grp = None
            self.mask_grp = None

    def _create_core_structures(self, sequence_dict, mask_dict, id_list: List[str]):
        self.hf = h5py.File(self.file_name, 'w')
        self.seq_grp = self.hf.create_group(GROUPNAMES.sequences)
        self.mask_grp = self.hf.create_group(GROUPNAMES.masks)
        self.id_grp = self.hf.create_group(GROUPNAMES.id_list)
        # self.
        hd5sizes = get_hd5_base_sizes(img_size=self.roboutils.img_size,
                                      batch_size=None,
                                      max_traj_len=self.roboutils.max_traj_len,
                                      max_txt_len=self.roboutils.max_txt_len)
        data_types = get_hd5_dtypes()
        mask_sizes = get_hd5_mask_sizes()
        for channel in self.use_channels:
            self.seq_grp.create_dataset(
                channel,
                dtype=data_types[channel],
                chunks=True,
                data=sequence_dict[channel].numpy(),
                maxshape=hd5sizes[channel],
            )

        for channel in self.use_channels:
            if mask_dict[channel] is None:
                continue

            # print(len(mask_dict[channel].numpy().shape),len(hd5sizes[channel]))
            self.mask_grp.create_dataset(
                channel,
                dtype=data_types[channel],
                chunks=True,
                data=mask_dict[channel].numpy(),
                maxshape=mask_sizes[channel],

            )
        self.id_grp.create_dataset(GROUPNAMES.id_list, dtype=h5py.string_dtype(
        ), chunks=True, data=id_list, maxshape=(None,))
        # self.final_id_list.extend(id_list)
        return True

class HDF5ContrastiveSetCreator:
    pass
