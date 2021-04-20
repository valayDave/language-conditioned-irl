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

from ..channel import ChannelData, ChannelHolder, ContrastiveGenerator


# Text and Image Configuration.
MAX_TEXT_LENGTH = 25
IMAGE_SIZE = (256, 256)
PATCH_SIZE = 32
PATCH_EMBEDDING_DIMS = 128


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


class RoboDataset(Dataset):
    def __init__(self, tokenizer=None, sampling_paths=[], image_resize_dims=IMAGE_SIZE, use_channels=USE_CHANNELS):
        super().__init__()
        self.sample_arr = []
        self.idx_combs = []
        self.tokenizer = tokenizer
        self.decompressed_cache = {}
        self.resize_compose = transforms.Compose(
            [transforms.Resize(image_resize_dims), transforms.ToTensor()])
        self.make_dataset_items(sampling_paths)
        self.use_channels = use_channels

        print("Using Channnels", use_channels)

    def make_dataset_items(self, sampling_paths,):
        self.st_dist = {}
        for p in sampling_paths:
            json_val = load_json_from_file(p)
            self.sample_arr.extend(
                self.decompress_json(x, json_val[x]) for x in json_val['samples']
            )
        self.idx_combs = list(itertools.combinations(
            range(len(self.sample_arr)), 2))

    def __len__(self):
        return len(self.idx_combs)

    def make_channel_sample(self, idx, with_cache=True):
        # self.decompress_json(idx,self.sample_arr[idx])
        decomp_obj = self.sample_arr[idx]
        traj_channel = self.create_trajectory(decomp_obj['state/dict'])
        sent_channel_tup = self.encode_sentence(decomp_obj['voice'])
        sent_channel = ChannelData(
            name='text',
            sequence=sent_channel_tup[0],
            mask=sent_channel_tup[1]
        )
        img_channel = ChannelData(
            name='image',
            sequence=self.make_tensor_from_image(decomp_obj['image']),
            mask=None
        )
        all_channel_dict = {
            'text': sent_channel,
            'image': img_channel,
            **traj_channel
        }
        if with_cache:
            self.cache_item(idx, all_channel_dict)
        return all_channel_dict

    def __getitem__(self, idx):
        idx1, idx2 = self.idx_combs[idx]
        # all_channel_dict1,all_channel_dict2 = self.get_cache(idx1),self.get_cache(idx2)
        # if all_channel_dict1 is None:
        all_channel_dict1 = self.make_channel_sample(idx1)
        # if all_channel_dict2 is None:
        all_channel_dict2 = self.make_channel_sample(idx2)

        return all_channel_dict1, all_channel_dict2

    def make_tensor_from_image(self, img_arr):
        image_arr = np.asarray(img_arr, dtype=np.uint8)[:, :, ::-1]
        img = Image.fromarray(image_arr, 'RGB')
        resized_img_tensor = self.resize_compose(img)
        return resized_img_tensor

    def encode_sentence(self, sentence):
        data_dict = self.tokenizer.batch_encode_plus(
            [sentence],                      # Sentence to encode.
            add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            # Pad & truncate all sentences.
            max_length=MAX_TEXT_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',     # Return pytorch tensors.
        )
        return (data_dict['input_ids'].squeeze(0), data_dict['attention_mask'].squeeze(0))

    def cache_item(self, id, item):
        self.decompressed_cache[id] = item

    def get_cache(self, idx):
        if idx in self.decompressed_cache:
            return self.decompressed_cache[idx]
        return None

    def decompress_json(self, msg_idx, data):
        cache_val = json.loads(zlib.decompress(base64.b64decode(data)))
        return cache_val

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
            # pad_sequence([zs_tensor,zt_tensor],batch_first=True)
            channels[k] = ChannelData(name=k,
                                      sequence=channel_seq,  # (l x d)
                                      )

        return channels

    @staticmethod
    def loading_collate_fn(batch):
        core_d1_channels = ChannelHolder()
        core_d2_channels = ChannelHolder()
        for cont_dicts_tup in batch:
            d1_dict, d2_dict = cont_dicts_tup
            core_d1_channels = populate_channel(d1_dict, core_d1_channels)
            core_d2_channels = populate_channel(d2_dict, core_d2_channels)

        core_d1_channels = set_masks_and_padding(core_d1_channels)
        core_d2_channels = set_masks_and_padding(core_d2_channels)
        return ContrastiveGenerator(core_d1_channels, core_d2_channels)


def set_masks_and_padding(core_channels):
    for k in core_channels:
        kd1_s = core_channels[k].sequence
        size_arr = [len(s) for s in core_channels[k].sequence]
        if k in NO_PAD_CHANNELS:
            core_channels[k].sequence = torch.stack(core_channels[k].sequence)
            continue
        core_channels[k].sequence = pad_sequence(kd1_s, batch_first=True)

        if len(core_channels[k].sequence.size()) == 3:
            _, mx_size, _ = core_channels[k].sequence.size()
        elif len(core_channels[k].sequence.size()) == 2:
            _, mx_size = core_channels[k].sequence.size()

        if mx_size is not None:
            # Create a mask if there is none.
            if core_channels[k].mask is None:
                core_channels[k].mask = make_mask_from_len(
                    torch.Tensor(size_arr), max_size=mx_size)
            else:
                core_channels[k].mask = pad_sequence(
                    core_channels[k].mask, batch_first=True)

    return core_channels


def make_mask_from_len(len_tensor, max_size):
    '''
    len_tensor: 
    '''
    return (torch.arange(max_size)[None, :] < len_tensor[:, None]).float()


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
