'''
This file holds all the core functions to create h5py dataset so that it can be used for fast data loading. 

Just creating index's and directly using the dataset with unzipping and unpacking is too slow. 
'''
from typing import List
import h5py
from torchvision import transforms
from dataclasses import dataclass
import torch
from PIL import Image
import base64

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
                    load_json_from_file

import gc
import logging

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')



def create_logger(logger_name:str,level=logging.INFO):
    custom_logger = logging.getLogger(logger_name)
    ch1 = logging.StreamHandler()
    ch1.setLevel(level)
    ch1.setFormatter(formatter)
    custom_logger.addHandler(ch1)
    custom_logger.setLevel(level)    
    return custom_logger
# Todo : Unpack entire files into h5py dataset.

# todo : Once the entire data is unpacked create the Mapping dataset which helps with the control parameters of the mapping function which makes this dataset.
def is_present(pth):
  try:
    os.stat(pth)
    return True
  except:
    return False

def is_present(pth):
  try:
    os.stat(pth)
    return True
  except:
    return False

def get_hd5_base_sizes(img_size=IMAGE_SIZE,\
                        batch_size=None,\
                        max_traj_len=MAX_TRAJ_LEN,\
                        max_txt_len=MAX_TEXT_LENGTH
                        ):
    return {
        'image': (batch_size, 3,*img_size),
        'joint_gripper': (batch_size,max_traj_len),
        'joint_gripper_velocity': (batch_size,max_traj_len, 1),
        'joint_robot_position': (batch_size,max_traj_len, 6),
        'joint_robot_velocity': (batch_size,max_traj_len, 6),
        'tcp_angular_veloctiy': (batch_size,max_traj_len, 3),
        'tcp_linear_velocity': (batch_size,max_traj_len, 3),
        'tcp_orientation': (batch_size,max_traj_len, 3),
        'tcp_position': (batch_size,max_traj_len, 3),
        'tcp_target_orientation': (batch_size,max_traj_len, 3),
        'tcp_target_position': (batch_size,max_traj_len, 3),
        'text': (batch_size, max_txt_len)
    }

def get_hd5_mask_sizes(batch_size=None,\
                       max_traj_len=MAX_TRAJ_LEN,\
                       max_txt_len=MAX_TEXT_LENGTH
                      ):
    return {
        'joint_gripper': (batch_size,max_traj_len),
        'joint_gripper_velocity': (batch_size,max_traj_len),
        'joint_robot_position': (batch_size,max_traj_len),
        'joint_robot_velocity': (batch_size,max_traj_len),
        'tcp_angular_veloctiy': (batch_size,max_traj_len),
        'tcp_linear_velocity': (batch_size,max_traj_len),
        'tcp_orientation': (batch_size,max_traj_len),
        'tcp_position': (batch_size,max_traj_len),
        'tcp_target_orientation': (batch_size,max_traj_len),
        'tcp_target_position': (batch_size,max_traj_len),
        'text': (batch_size, max_txt_len)
    }




def get_hd5_dtypes():
    return {
        'image': 'f',
        'joint_gripper': 'f',
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



def pad_1d_tensors(sequences, max_len=None,padding_val=0):
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


def pad_2d_tensors(sequences, max_len=None,padding_val=0):
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
                 tokenizer=None,\
                 image_resize_dims=IMAGE_SIZE,\
                 max_traj_len=MAX_TRAJ_LEN,\
                 use_channels=USE_CHANNELS,\
                 max_txt_len=MAX_TEXT_LENGTH) -> None:
        self.tokenizer = tokenizer
        self.max_txt_len = max_txt_len
        self.max_traj_len = max_traj_len
        self.use_channels=use_channels
        self.img_size=image_resize_dims
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
    def make_padded_sequence(sequences, max_len=None,padding_val=0):
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
    

    def make_tensor_saveable_object(self,robo_data_dict:dict):
      required_keys = [
        'state/dict',
        'voice',
        'image'
      ]
      assert len([k for k in required_keys if k in robo_data_dict]) == len(required_keys)
      traj_data_dict = self.create_trajectory(robo_data_dict['state/dict'],use_channels=self.use_channels)
      text_ids,txt_msk = self.encode_sentence(robo_data_dict['voice'])
      image_tensor = self.make_tensor_from_image(robo_data_dict['image'])

      input_sequence_dict = dict(
        image = image_tensor,
        text = text_ids,
        state=traj_data_dict
      )
      mask_dict = dict(
        image = None,
        text = txt_msk,
        state={k:None for k in traj_data_dict} 
      )
      return input_sequence_dict,mask_dict

    def make_saveable_chunk(self,robo_data_dicts:List[dict]):
      sequences = []
      masks = []
      id_list = []
      for robo_data_dict in robo_data_dicts:
        # No masks except for text are added att start. 
        id_list.append(robo_data_dict['name'])
        seq_dict,msk_dict = self.make_tensor_saveable_object(robo_data_dict)
        sequences.append(seq_dict)
        masks.append(msk_dict)
      
      
      text_ten = torch.stack([seq_dict['text'] for seq_dict in sequences])
      text_msk_tensor = torch.stack([msk_dict['text'] for msk_dict in masks])
      image_tensor = torch.stack([seq_dict['image'] for seq_dict in sequences])

      # Mask state Tensors for this. 
      # all sequences will be padded which are non text and mask is created. 
      state_dict = dict()
      state_mask_dict = dict()
      padding_vals = get_padding()
      for traj_chan in sequences[0]['state']:
        padded_traj_seq,trj_mask = self.make_padded_sequence([seq_dict['state'][traj_chan] for seq_dict in sequences],max_len=self.max_traj_len,padding_val=padding_vals[traj_chan])
        state_dict[traj_chan] = padded_traj_seq
        state_mask_dict[traj_chan] = trj_mask

      input_sequence_dict = dict(
          image = image_tensor,
          text = text_ten,
          **state_dict
        )
      mask_dict = dict( # Normalized to ensure H5DataCreatorMainDataCreator works well
        image = None, 
        text = text_msk_tensor,
        **state_mask_dict
      )
      return input_sequence_dict,mask_dict,id_list

class H5DataCreatorMainDataCreator:
    """ 
    There are three main datasets. :

    1. Demonstrations : `demo_set` (HDF5 dataset) with all demos and associated index `id_list` 
      - We create this dataset if there is a change in the channels the transformer will attend to.  

    2. Metadata : `metadata` about each id in the `id_list`. Ideallly as a CSV dataframe
      - We create this dataset once. 

    3. Contrastive Pairs : using the `demo_set` and `metadata` we create the `contrastive_set` for training and testing. 
      - We create this dataset using control parameters (`CP`). 
        - These control parameters determine the distribution over different forms of contrasting behaviours. Like Picking Pouring. ETC
        - They will help make a train and test distribution. 
    """
    def __init__(self,
                 save_file_name,\
                 sample_pths: List[str],\
                 use_channels=USE_CHANNELS,\
                 image_resize_dims=IMAGE_SIZE,\
                 tokenizer=None,\
                 chunk_size=256,\
                 max_traj_len=MAX_TRAJ_LEN,\
                 max_txt_len=MAX_TEXT_LENGTH):
        assert not is_present(save_file_name )
        assert tokenizer is not None
        self.roboutils = RoboDataUtils(tokenizer=tokenizer,\
                                  use_channels=use_channels,\
                                  max_traj_len=max_traj_len,\
                                  max_txt_len=max_txt_len,\
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
          decompressd_objects = parallel_map(lambda x : self.decompress_json(loaded_obj[x]), loaded_obj['samples'])
          # Make the Chunked Tensors from this 
          object_tuple = self.roboutils.make_saveable_chunk(decompressd_objects)
          sequence_dict,mask_dict,id_list = object_tuple
          if self.hf is None:
            self._create_core_structures(sequence_dict,mask_dict,id_list)
          else:
            self._append_to_file(sequence_dict,mask_dict,id_list)
          del loaded_obj
          del decompressd_objects
          del object_tuple
          gc.collect()
        self.close()
          
    def _append_to_file(self,sequence_dict,mask_dict,id_list:List[str]):
      if self.hf is None:
        raise Exception('No HDF5 File Open')
      for channel in self.use_channels:
        channel_seq_dataset = self.seq_grp.get(channel)
        id_list_dataset = self.id_grp.get('id_list')
        old_coll_size = channel_seq_dataset.shape[0]
        new_coll_size =  old_coll_size + len(id_list)        
        # Resize data based on the new data
        channel_seq_dataset.resize(new_coll_size,axis=0)
        id_list_dataset.resize(new_coll_size,axis=0)
        # Add the new Items to the dataset. 
        channel_seq_dataset[-len(id_list):] = sequence_dict[channel]
        if channel in mask_dict and mask_dict[channel] is not None:
          channel_msk_dataset = self.mask_grp.get(channel)
          channel_msk_dataset.resize(new_coll_size,axis=0)
          channel_msk_dataset[-len(id_list):] = mask_dict[channel]
        
    def decompress_json(self, data):
        cache_val = json.loads(zlib.decompress(base64.b64decode(data)))
        return cache_val

    def close(self):
        if self.hf is not None:
            self.hf.close()
            self.hf = None
            self.seq_grp=None
            self.mask_grp=None

    def _create_core_structures(self,sequence_dict,mask_dict,id_list:List[str]):
        self.hf = h5py.File(self.file_name, 'w')
        self.seq_grp = self.hf.create_group('sequences')
        self.mask_grp = self.hf.create_group('mask')
        self.id_grp = self.hf.create_group('id_list')
        # self.
        hd5sizes = get_hd5_base_sizes(img_size=self.roboutils.img_size,\
                                      batch_size=None,
                                      max_traj_len=self.roboutils.max_traj_len,\
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
        self.id_grp.create_dataset('id_list',dtype=h5py.string_dtype(),chunks=True,data=id_list,maxshape=(None,))
        # self.final_id_list.extend(id_list)
        return True



class HDF5ContrastiveSetCreator:
  pass