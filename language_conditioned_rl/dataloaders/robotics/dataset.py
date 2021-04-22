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
import os
import h5py
from typing import Dict, List
import random
from ..channel import ChannelData, ChannelHolder, ContrastiveGenerator


# Text and Image Configuration.
MAX_TEXT_LENGTH = 25
IMAGE_SIZE = (256, 256)
PATCH_SIZE = 32
PATCH_EMBEDDING_DIMS = 128

# Configuration for Dataset size
MAX_TRAJ_LEN = 500



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


def load_json_from_file(file_path):
    with open(file_path, 'r') as f:
        json_file = json.load(f)
    return json_file


def save_json_to_file(json_dict, file_path):
    with open(file_path, 'w') as f:
        json.dump(json_dict, f)



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


def pad_1d_tensors(sequences, max_len=None, padding_val=0):
    """
    :param sequences: list of tensors
      sequences = [torch(b),torch(k),....torch(z)] 


    :return:
      tuple(
        padded_seq : torch(len(sequences) x max_len),
        mask : torch(len(sequences))
      )
    """
    num = len(sequences)
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_dims = (num, max_len)
    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_val)
    mask = sequences[0].data.new(*out_dims).fill_(0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
        mask[i, :length] = 1
    return out_tensor, mask


def pad_2d_tensors(sequences, max_len=None, padding_val=0):
    """
    :param sequences: list of tensors
      sequences = [torch(b x d),torch(k x d),....torch(z x d)]
    :return:
      tuple(
        padded_seq : torch(len(sequences) x max_len x d),
        mask : torch(len(sequences))
      )
    """
    if len(set([s.size(1) for s in sequences])) > 1:
        raise Exception(
            "When Padding 2d tensored sequenced, dim=1 needs to be same for all items in the sequence. ")

    num = len(sequences)
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    _, dims = sequences[0].size()

    out_dims = (num, max_len, dims)
    out_dims_mask = (num, max_len)
    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_val)
    mask = sequences[0].data.new(*out_dims_mask).fill_(0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
        mask[i, :length] = 1
    return out_tensor, mask


class RoboDataUtils:
    def __init__(self,
                 tokenizer=None,
                 image_resize_dims=IMAGE_SIZE,
                 max_traj_len=MAX_TRAJ_LEN,
                 use_channels=USE_CHANNELS,
                 max_txt_len=MAX_TEXT_LENGTH) -> None:
        self.tokenizer = tokenizer
        self.max_txt_len = max_txt_len
        self.max_traj_len = max_traj_len
        self.use_channels = use_channels
        self.img_size = image_resize_dims
        self.resize_compose = transforms.Compose(
            [transforms.Resize(image_resize_dims), transforms.ToTensor()])

    def encode_sentence(self, sentence):
        data_dict = self.tokenizer.batch_encode_plus(
            [sentence],                      # Sentence to encode.
            add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            # Pad & truncate all sentences.
            max_length=self.max_txt_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',     # Return pytorch tensors.
        )
        return (data_dict['input_ids'].squeeze(0), data_dict['attention_mask'].squeeze(0))

    def make_tensor_from_image(self, img_arr):
        image_arr = np.asarray(img_arr, dtype=np.uint8)[:, :, ::-1]
        img = Image.fromarray(image_arr, 'RGB')
        resized_img_tensor = self.resize_compose(img)
        return resized_img_tensor

    @staticmethod
    def make_mask_from_len(len_tensor, max_size):
        '''
        len_tensor: 
        '''
        return (torch.arange(max_size)[None, :] < len_tensor[:, None]).float()

    @staticmethod
    def create_trajectory(state_dict_arr, use_channels=USE_CHANNELS):
        if len(state_dict_arr) == 0:
            return []
        channel_keys = list(state_dict_arr[0].keys())
        channels = {}
        state_df = pandas.DataFrame(state_dict_arr)
        for k in channel_keys:
            if k not in use_channels:
                continue
            channel_seq = torch.Tensor(state_df[k])
            if k in DISCRETE_CHANNELS:
                channel_seq = channel_seq.long().squeeze(1)
            channels[k] = channel_seq

        return channels

    @staticmethod
    def make_padded_sequence(sequences, max_len=None, padding_val=0):
        """
        INPUT
          sequences : [tensor.size(L1xd1),tensor.size(L2xd1),tensor.size(L3xd1)..] : b length
          OR 
          sequences : [tensor.size(L1),tensor.size(L2),tensor.size(L3)..]
        RETURNS:
          tuple(sequence_tensor,mask)
          if 1d Size sequences:
            tensor(b x max_len),tensor(max_len)
        """
        size_set = set([len(s.size()) for s in sequences])
        if len(size_set) > 1:
            raise Exception(
                "All tensors in the sequence have to be the same size")
        size = list(size_set)[0]
        if size == 1:
            return pad_1d_tensors(sequences, max_len=max_len, padding_val=padding_val)
        elif size == 2:
            return pad_2d_tensors(sequences, max_len=max_len, padding_val=padding_val)
        else:
            raise Exception("Cannot handle More than 2 mins of padding")

    def make_tensor_saveable_object(self, robo_data_dict: dict):
        required_keys = [
            'state/dict',
            'voice',
            'image',
            'main_name'
        ]
        assert len([k for k in required_keys if k in robo_data_dict]
                   ) == len(required_keys)
        traj_data_dict = self.create_trajectory(
            robo_data_dict['state/dict'], use_channels=self.use_channels)
        text_ids, txt_msk = self.encode_sentence(robo_data_dict['voice'])
        image_tensor = self.make_tensor_from_image(robo_data_dict['image'])

        input_sequence_dict = dict(
            image=image_tensor,
            text=text_ids,
            state=traj_data_dict
        )
        mask_dict = dict(
            image=None,
            text=txt_msk,
            state={k: None for k in traj_data_dict}
        )
        return input_sequence_dict, mask_dict

    def make_saveable_chunk(self, robo_data_dicts: List[dict]):
        sequences = []
        masks = []
        id_list = []
        for robo_data_dict in robo_data_dicts:
            # No masks except for text are added att start.
            seq_dict, msk_dict = self.make_tensor_saveable_object(
                robo_data_dict)
            id_list.append(robo_data_dict['main_name'])
            sequences.append(seq_dict)
            masks.append(msk_dict)

        text_ten = torch.stack([seq_dict['text'] for seq_dict in sequences])
        text_msk_tensor = torch.stack([msk_dict['text'] for msk_dict in masks])
        image_tensor = torch.stack([seq_dict['image']
                                    for seq_dict in sequences])

        # Mask state Tensors for this.
        # all sequences will be padded which are non text and mask is created.
        state_dict = dict()
        state_mask_dict = dict()
        padding_vals = get_padding()
        for traj_chan in sequences[0]['state']:
            padded_traj_seq, trj_mask = self.make_padded_sequence(
                [seq_dict['state'][traj_chan] for seq_dict in sequences], max_len=self.max_traj_len, padding_val=padding_vals[traj_chan])
            state_dict[traj_chan] = padded_traj_seq
            state_mask_dict[traj_chan] = trj_mask

        input_sequence_dict = dict(
            image=image_tensor,
            text=text_ten,
            **state_dict
        )
        mask_dict = dict(  # Normalized to ensure H5DataCreatorMainDataCreator works well
            image=None,
            text=text_msk_tensor,
            **state_mask_dict
        )
        return input_sequence_dict, mask_dict, id_list


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

    def __getitem__(self, index):
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


    def __len__(self):
        return len(self.id_list)


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