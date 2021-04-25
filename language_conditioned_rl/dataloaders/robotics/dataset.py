from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from torchvision import transforms
from PIL import Image
import json
import numpy as np
import zlib
import itertools
import pandas
import base64
from functools import lru_cache
import os
import h5py
from typing import Dict, List,Tuple
import random
from ..channel import ChannelData, ChannelHolder, ContrastiveGenerator


# Text and Image Configuration.
MAX_TEXT_LENGTH = 25
IMAGE_SIZE = (256, 256)
PATCH_SIZE = 32
PATCH_EMBEDDING_DIMS = 128

# Configuration for Dataset size
MAX_TRAJ_LEN = 500


CONTRASTIVE_HDF5_DATASET_NAME = 'contrastive_ids'
CONTRASTIVE_HDF5_DATASET_NAME_MAIN_DEMO = f'{CONTRASTIVE_HDF5_DATASET_NAME}_main_demos'
CONTRASTIVE_HDF5_DATASET_NAME_CACHE_INDICES = f'{CONTRASTIVE_HDF5_DATASET_NAME}_cache_indices'

def is_present(pth):
    try:
        os.stat(pth)
        return True
    except:
        return False

class GROUPNAMES:
    id_list = 'id_list'
    sequences = 'sequences'
    masks = 'masks'


CONTINOUS_VALUE_DIMS = {
    'joint_gripper_velocity': 1,
    'joint_robot_position': 6,
    'joint_robot_velocity': 6,
    'tcp_angular_veloctiy': 3,
    'tcp_linear_velocity': 3,
    'tcp_orientation': 3,
    'tcp_position': 3,
    'tcp_target_orientation': 3,
    'tcp_target_position': 3,
}

DISCRETE_CHANNELS = {
    'joint_gripper': True
}
NO_PAD_CHANNELS = {
    'image': True,
}

ALL_CHANNELS = [
    'joint_gripper_velocity',
    'joint_robot_position',
    'joint_robot_velocity',
    'tcp_angular_veloctiy',
    'tcp_linear_velocity',
    'tcp_orientation',
    'tcp_position',
    'tcp_target_orientation',
    'tcp_target_position',
    'image',
    'text',
    'joint_gripper',
]
USE_CHANNELS = [
    # 'joint_gripper_velocity',
    'joint_robot_position',
    'joint_robot_velocity',
    'image',
    'text',
    'joint_gripper',
]

def get_padding():
    return {
        'image': 0,
        'joint_gripper': 3,
        'joint_gripper_velocity': 0,
        'joint_robot_position': 0,
        'joint_robot_velocity': 0,
        'tcp_angular_veloctiy': 0,
        'tcp_linear_velocity': 0,
        'tcp_orientation': 0,
        'tcp_position': 0,
        'tcp_target_orientation': 0,
        'tcp_target_position': 0,
        'text': 0,
    }


def collate_indices(dataframe: pandas.DataFrame, indices: List[Tuple[int, int]],identifier_name=None):
    """_collate_indices 
    Correlate the indices and store return the list of tuples with identifiers to the objects in its dataframe
    """
    assert identifier_name is not None and identifier_name in dataframe.columns
    collated_id_data = []
    for idx_tuple in indices:
        pos_idx, neg_idx = idx_tuple
        pos_obj, neg_obj = dataframe.iloc[dataframe.index.get_loc(pos_idx)],\
            dataframe.iloc[dataframe.index.get_loc(neg_idx)]

        posid, negid = pos_obj[identifier_name],\
            neg_obj[identifier_name]
        collated_id_data.append(
            (posid, negid)
        )

    return collated_id_data

def map_to_contrastive_indexes_to_ids(collated_ids: List[Tuple[str, str]], index_map: Dict[str, int]):
    """_map_to_demo_indexes 
    collated_ids : List of tuples with pos/neg ids in them 
    index_map : dictionary to map strings in `self.id_list` to index so that it can be used to help collate indexes for indexdata
    """
    mapped_arr = []
    for indexes in collated_ids:
        pid, nid = indexes
        mapped_arr.append([index_map[pid], index_map[nid]])
    return mapped_arr

class ContrastiveCollateFn:
    """ 
    This is the Collate function which will help make the batch of contrastive samples as one Object. 
    The `ContrastiveGenerator` object is an object which holds all the channel data and at the time of 
    learning it will use the `create_contrastive_inputs` function to help create the contrastive examples. 
    """
    def __call__(self,batch):
        core_d1_channels = ChannelHolder()
        core_d2_channels = ChannelHolder()
        for cont_dicts_tup in batch:
            d1_dict, d2_dict = cont_dicts_tup
            core_d1_channels = self.populate_channel(d1_dict, core_d1_channels)
            core_d2_channels = self.populate_channel(d2_dict, core_d2_channels)

        core_d1_channels = self.stack_channels(core_d1_channels)
        core_d2_channels = self.stack_channels(core_d2_channels)
        return ContrastiveGenerator(core_d1_channels, core_d2_channels)

    @staticmethod
    def stack_channels(core_channels:Dict[str,ChannelData]):
        for k in core_channels:
            core_channels[k].sequence = torch.stack(core_channels[k].sequence)
            if core_channels[k].mask is not None:
                if len(core_channels[k].mask) > 0 :
                    core_channels[k].mask = torch.stack(core_channels[k].mask)
            
        return core_channels

    @staticmethod
    def populate_channel(d_dict, core_channels):
        for k in d_dict:
            channel_obj = d_dict[k]
            if channel_obj.name not in core_channels:
                core_channels[channel_obj.name] = ChannelData(
                    name=channel_obj.name,
                    sequence=[channel_obj.sequence],
                    mask=None if channel_obj.mask is None else [channel_obj.mask]
                )
            else:
                core_channels[channel_obj.name].sequence.append(
                    channel_obj.sequence)
                if channel_obj.mask is not None and core_channels[channel_obj.name].mask is not None:
                    core_channels[channel_obj.name].mask.append(channel_obj.mask)
        return core_channels  # { k : list(tensor(1xsxd))}

class DemonstrationsDataset(Dataset):
    """DemonstrationsDataset 
    Dataset wrapper over the HDF5 dataset which is stored. 
    """
    def __init__(self, filename: str,) -> None:
        super().__init__()
        assert is_present(filename)
        self._open_dataset(filename)

    def _open_dataset(self,filename):
        self.h5 = h5py.File(filename,'r')
        self.id_list = self.h5.get(GROUPNAMES.id_list)[GROUPNAMES.id_list]
        self.sequences = self.h5.get(GROUPNAMES.sequences)
        self.masks = self.h5.get(GROUPNAMES.masks)

    def __getitem__(self, index) -> Dict[str,ChannelData]:
        """__getitem__ 
        returns dictionary of ChannelData
        """
        channel_dict = {}
        for k in self.sequences.keys():
            mask = None if k not in self.masks else torch.from_numpy(self.masks[k][index])
            channel_dict[k] = ChannelData(
                mask=mask,
                sequence=torch.from_numpy(self.sequences[k][index]),
                name=k
            )
        return channel_dict
    
    def __len__(self):
        return len(self.id_list)
    
    @property
    def is_closed(self):
        if self.h5 is None:
            return True
        else: 
            return False


    def close(self):
        self.h5.close()
        self.id_list = None
        self.sequences = None
        self.masks = None
        print("Dataset Closed")



class RandomPairwiseDataset(Dataset):
    """RandomPairwiseDataset 
    Dummy dataset to test functionality of contrastive colllaation. 
    """
    def __init__(self,filename:str,sample=20) -> None:
        super().__init__()
        self.demods = DemonstrationsDataset(filename)
        all_idxs = [i for i in range(len(self.demods))]
        if sample is not None:
            all_idxs = random.sample(all_idxs,sample)
        self.idxs = list(itertools.combinations(all_idxs,2))
    
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        idx_i,idx_j = self.idxs[idx]
        samp_i = self.demods[idx_i] 
        samp_j = self.demods[idx_j] 
        return samp_i,samp_j

    @staticmethod
    def collate_fn():
        return ContrastiveCollateFn()


class ContrastiveSampleGeneratedDataset(Dataset):
    """ContrastiveSampleGeneratedDataset 
    Dummy dataset to test functionality of contrastive collation. 

    Needs the `ContrastiveCollateFn` as a part of its data loader.
    """
    def __init__(self,\
                constrastive_set_hdf5_file:str,\
                use_channels=USE_CHANNELS) -> None:
        super().__init__()
        self.constrastive_set_hdf5_file=constrastive_set_hdf5_file
        self._open_dataset(self.constrastive_set_hdf5_file,use_channels=use_channels)

    def _open_dataset(self,filename,use_channels=USE_CHANNELS):
        assert is_present(filename), f"Contrastive Set {filename} should exist!"
        self.h5 = h5py.File(filename,'r')
        self.id_list = self.h5.get(GROUPNAMES.id_list)
        self.sequences = self.load_sequences(self.h5.get(GROUPNAMES.sequences),use_channels)
        self.masks = self.load_sequences(self.h5.get(GROUPNAMES.masks),use_channels,mask=True)
        self.contrastive_indices = list(self.h5.get(CONTRASTIVE_HDF5_DATASET_NAME_CACHE_INDICES))
    
    @staticmethod
    def load_sequences(seq,channels,mask=False):
        dd = {}
        for k in channels:
            assert mask or k in seq.keys(), f"Channel {k} not found in the HDF5 Dataset Sequence"
            if k not in seq.keys():
                continue
            dd[k] = np.array(seq[k])
        return dd
       
    def __len__(self):
        return len(self.contrastive_indices)

    def get_channel_data(self,index):
        """__getitem__ 
        returns dictionary of ChannelData
        """
        channel_dict = {}
        for k in self.sequences.keys():
            mask = None if k not in self.masks else torch.from_numpy(self.masks[k][index])
            channel_dict[k] = ChannelData(
                mask=mask,
                sequence=torch.from_numpy(self.sequences[k][index]),
                name=k
            )
        return channel_dict

    def __getitem__(self, idx):
        idx_i,idx_j = self.contrastive_indices[idx]
        samp_i = self.get_channel_data(idx_i) # Dictionary
        samp_j = self.get_channel_data(idx_j) # Dictionary
        return samp_i,samp_j
    
    @staticmethod
    def collate_fn():
        return ContrastiveCollateFn()