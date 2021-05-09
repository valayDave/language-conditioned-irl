'''
This file holds all the core functions to create h5py dataset so that it can be used for fast data loading. 

Just creating index's and directly using the dataset with unzipping and unpacking is too slow. 

# Todo : Unpack entire files into h5py dataset.

# todo : Once the entire data is unpacked create the Mapping dataset which helps with the control parameters of the mapping function which makes this dataset.

'''
from typing import Dict, List, Tuple
import h5py
from torchvision import transforms
from dataclasses import dataclass, field
import datetime
import abc
import itertools
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
import gc
import logging
from .dataset import MAX_TRAJ_LEN,\
    USE_CHANNELS,\
    IMAGE_SIZE,\
    MAX_TEXT_LENGTH,\
    DISCRETE_CHANNELS,\
    MAX_TRAJ_LEN,\
    MAX_VIDEO_FRAMES,\
    is_present,\
    GROUPNAMES,\
    DemonstrationsDataset,\
    CONTRASTIVE_HDF5_DATASET_NAME_MAIN_DEMO,\
    CONTRASTIVE_HDF5_DATASET_NAME_CACHE_INDICES,\
    collate_indices,\
    map_to_contrastive_indexes_to_ids

from .utils import \
    RoboDataUtils,\
    load_json_from_file,\
    save_json_to_file



class HDF5ContrastiveSetCreator:
    pass


def safe_mkdir(pth):
    try:
        os.makedirs(pth)
    except:
        pass


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
                       max_txt_len=MAX_TEXT_LENGTH,
                       max_video_frames = MAX_VIDEO_FRAMES,
                       ):
    return {
        'image': (batch_size, 3, *img_size),
        'image_sequence': (batch_size,max_video_frames, 3, *img_size),
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
        'image_sequence':'f',
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
        self.logger = create_logger(self.__class__.__name__)
        for json_pth in self.sample_pths:
            self.logger.info(f"Writing Path {json_pth}")
            loaded_obj = load_json_from_file(json_pth)
            assert 'samples' in loaded_obj
            decompressd_objects = parallel_map(
                lambda x: self.decompress_json(loaded_obj[x], x), loaded_obj['samples'])
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
        id_list = [n.encode("ascii", "ignore") for n in id_list]
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

    def decompress_json(self, data, key):
        cache_val = json.loads(zlib.decompress(base64.b64decode(data)))
        cache_val['main_name'] = key
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
        id_list = [n.encode("ascii", "ignore") for n in id_list]
        self.id_grp.create_dataset(
            GROUPNAMES.id_list, dtype='S20', chunks=True, data=id_list, maxshape=(None,))
        # self.final_id_list.extend(id_list)
        return True

class HDF5VideoDatasetCreator(H5DataCreatorMainDataCreator):

    META_EXTRACTION_KEYS = [
        'phase',
        'ints',
        'voice',
        'target/id',
        'target/type',
        'amount',
        'main_name'
    ]
    
    def decompress_json(self, data, key):
        cache_val = json.loads(zlib.decompress(base64.b64decode(data)))
        cache_val['main_name'] = key
        return cache_val

    def load_and_decompress(self,json_pth):
        loaded_obj = load_json_from_file(json_pth)
        file_name = json_pth.split('/')[-1].split('.json')[0]
        decompressed_object = self.decompress_json(list(loaded_obj.values())[0], file_name)
        return decompressed_object

    @staticmethod
    def _make_list_chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def _make_metadf(self,decomp_objs):
        data_df = pandas.DataFrame(decomp_objs)
        data_df = data_df[self.META_EXTRACTION_KEYS]
        data_df['num_cups'] = data_df['ints'].apply(lambda x:x[1])
        data_df['num_bowls'] = data_df['ints'].apply(lambda x:x[0])
        data_df['comes_from'] = data_df['main_name']
        return data_df
    
    
    def build(self,chunk_size=20):
        self.logger = create_logger(self.__class__.__name__)
        file_chunks = self._make_list_chunks(self.sample_pths,chunk_size)
        main_meta_df = []
        for idx,json_pth_chunk in enumerate(file_chunks):

            decompressd_objects = parallel_map(lambda x : self.load_and_decompress(x),json_pth_chunk)
            self.logger.info(f"Completed Loading Path Chunk {idx}")
            # Make the Chunked Tensors from this
            object_tuple = self.roboutils.make_saveable_chunk(decompressd_objects)
            sequence_dict, mask_dict, id_list = object_tuple
            if self.hf is None:
                self._create_core_structures(sequence_dict, mask_dict, id_list)
            else:
                self._append_to_file(sequence_dict, mask_dict, id_list)
            main_meta_df.append(
                self._make_metadf(decompressd_objects)
            )
            del decompressd_objects
            del object_tuple
            gc.collect()
        self.close()
        concat_df = pandas.concat(main_meta_df)
        concat_df.to_csv(
            f'{self.file_name}.meta.csv'
        )

FORBIDDEN_SENTENCES = [
    "dump some into the bowl",
    'pour some into the container',
    'pour some into the round bowl',
    'pour some into the small vessel',
    'dump some into the circular vessel',
    'dump some into the bowl',
    'carefully dump some into the circular container',
    'briskly dump some material into the pink cup',
    'quickly dump some into the square container',
    'dump some into the circular vessel',
    'in a fast manner dump some into the tiny bin',
    'put a portion into the box',
    'place some material into the bowl',
    'put a part in the mug into the tiny bowl',
]
OBJECT_IDENTIFIERS = [
    {
        "ID": 1,
        "Type": "cup",
        "Color": "red",
        "Size": "n/a",
        "Shape": "n/a"
    },
    {
        "ID": 2,
        "Type": "cup",
        "Color": "green",
        "Size": "n/a",
        "Shape": "n/a"
    },
    {
        "ID": 3,
        "Type": "cup",
        "Color": "blue",
        "Size": "n/a",
        "Shape": "n/a"
    },
    {
        "ID": 1,
        "Type": "bowl",
        "Color": "yellow",
        "Size": "small",
        "Shape": "round"
    },
    {
        "ID": 2,
        "Type": "bowl",
        "Color": "red",
        "Size": "small",
        "Shape": "round"
    },
    {
        "ID": 3,
        "Type": "bowl",
        "Color": "green",
        "Size": "small",
        "Shape": "round"
    },
    {
        "ID": 4,
        "Type": "bowl",
        "Color": "blue",
        "Size": "small",
        "Shape": "round"
    },
    {
        "ID": 5,
        "Type": "bowl",
        "Color": "pink",
        "Size": "small",
        "Shape": "round"
    },
    {
        "ID": 6,
        "Type": "bowl",
        "Color": "yellow",
        "Size": "large",
        "Shape": "round"
    },
    {
        "ID": 7,
        "Type": "bowl",
        "Color": "red",
        "Size": "large",
        "Shape": "round"
    },
    {
        "ID": 8,
        "Type": "bowl",
        "Color": "green",
        "Size": "large",
        "Shape": "round"
    },
    {
        "ID": 9,
        "Type": "bowl",
        "Color": "blue",
        "Size": "large",
        "Shape": "round"
    },
    {
        "ID": 10,
        "Type": "bowl",
        "Color": "pink",
        "Size": "large",
        "Shape": "round"
    },
    {
        "ID": 11,
        "Type": "bowl",
        "Color": "yellow",
        "Size": "small",
        "Shape": "square"
    },
    {
        "ID": 12,
        "Type": "bowl",
        "Color": "red",
        "Size": "small",
        "Shape": "square"
    },
    {
        "ID": 13,
        "Type": "bowl",
        "Color": "green",
        "Size": "small",
        "Shape": "square"
    },
    {
        "ID": 14,
        "Type": "bowl",
        "Color": "blue",
        "Size": "small",
        "Shape": "square"
    },
    {
        "ID": 15,
        "Type": "bowl",
        "Color": "pink",
        "Size": "small",
        "Shape": "square"
    },
    {
        "ID": 16,
        "Type": "bowl",
        "Color": "yellow",
        "Size": "large",
        "Shape": "square"
    },
    {
        "ID": 17,
        "Type": "bowl",
        "Color": "red",
        "Size": "large",
        "Shape": "square"
    },
    {
        "ID": 18,
        "Type": "bowl",
        "Color": "green",
        "Size": "large",
        "Shape": "square"
    },
    {
        "ID": 19,
        "Type": "bowl",
        "Color": "blue",
        "Size": "large",
        "Shape": "square"
    },
    {
        "ID": 20,
        "Type": "bowl",
        "Color": "pink",
        "Size": "large",
        "Shape": "square"
    }
]

class SampleContrastingRule(metaclass=abc.ABCMeta):
    """SampleContrastingRule 
    The Goal of this class is to apply the rules of contrasting on the Metadata 
    DataFrame and extract the contrasting indices based on the rule. 
    `num_samples_per_rule` controls the number of indices that will get extracted using this contrastive matching rule. 

    Implement the Following Methods By Inheritance.:
        : _execute_rule : Filter a data frame with indices that belong to that contrastive samples of the rules. 
        : _validate_rule : Create a function to validate if two sample follow this rule or not. 
    
    Implement Following Properties for Testing Plots : 
        : `self.rule_map` : {
            "pos_traj_rw":"", # Based on contrasting rule what is meaning of the postive sample
            "neg_traj_rw":"", # Based on contrasting rule what is meaning of the negative sample
            "plot_title": "" # Title of the plot when plotting the postive negative samples
        }
    """

    def __init__(self, description=None):
        self.description = description
        self.rule_name = self.__class__.__name__
        self.rule_map = {
            "pos_traj_rw":"",
            "neg_traj_rw":"",
            "plot_title": ""
        }

    def __call__(self, metadf: pandas.DataFrame, num_samples_per_rule: int = 1000) -> List[Tuple[str, str]]:
        required_columns = HDF5ContrastiveSetCreator.MAPPING_COLUMNS.values()
        assert set(required_columns).issubset(set(metadf.columns)), \
                f'{set(metadf.columns)} Not a Part of {set(required_columns)}'
        return self._execute_rule(metadf, num_samples_per_rule=num_samples_per_rule)

    def _execute_rule(self, metadf: pandas.DataFrame, num_samples_per_rule: int = 100) -> List[Tuple[str, str]]:
        raise NotImplementedError()
    
    def validate_rule(self,demo_a:pandas.Series,demo_b:pandas.Series):
        required_columns = HDF5ContrastiveSetCreator.MAPPING_COLUMNS.values()
        assert set(required_columns).issubset(set(demo_a.keys()))
        assert set(required_columns).issubset(set(demo_b.keys()))
        self._validate_rule(demo_a,demo_b)
        
    def _validate_rule(self,demo_a:pandas.Series,demo_b:pandas.Series):
        raise NotImplementedError()

    def get_contrasting_attribute_pairs(self,metadf: pandas.DataFrame,column_name:str,num_samples_per_rule: int = 100):
        assert column_name in metadf.columns
        contrasiting_indices = self(metadf,num_samples_per_rule=num_samples_per_rule)
        filter_df = metadf[column_name]
        return_pairs =[]
        for idxtup in contrasiting_indices:
            return_pairs.append((filter_df.iloc[idxtup[0]],filter_df.iloc[idxtup[1]]))
        return return_pairs


class ContrastingActionsRule(SampleContrastingRule):
    """ContrastingActionsRule 
    Rule creates contrasting indexes based on examples which are different by task_type : Picking Vs Pouring
    """

    def __init__(self,):
        super().__init__(description="Variation in the type of task creates contrastive samples")
        self.rule_map = {
            "pos_traj_rw":"The sentence is describing task X and the trajectory followed task X",
            "neg_traj_rw":"The sentence is describing task Y and the trajectory followed task X",
            "plot_title":"Reward Distribution when the tasks are different "
        }

    def _execute_rule(self, metadf: pandas.DataFrame, num_samples_per_rule: int = 1000) -> List[Tuple[str, str]]:
        sets = []
        non_sampled_df = metadf.groupby(['demo_type'])
        for _, idxs in non_sampled_df.groups.items():
            sets.append(idxs)

        return_indexs = []
        set0, set1 = sets
        for i in range(num_samples_per_rule):
            return_indexs.append((random.choice(set0), random.choice(set1)))

        return return_indexs
    
    def _validate_rule(self,demo_a:pandas.Series,demo_b:pandas.Series):
        return demo_a['demo_type'] != demo_b['demo_type']



class PouringShapeSizeContrast(SampleContrastingRule):
    """PouringShapeSizeContrastPouringShapeSizeContrast 
    Rule creating contrasting indices based on shapes and sizes of the objects for pouring.

    This requires a little complex rule. 
        - Every object contains the folloowing attributes: 
            - shape
            - size
            - color
        - Contrastive example created from following rule : 
            - Pick some object of T1 based on shape and size.
            - Pick T2 where T2_shape != T1_shape and T2_size != T1_size and T2_color != T1_color

    """

    def __init__(self,):
        super().__init__(description="Rule creating contrasting indices based on shapes and sizes of the objects for pouring. Picking will be done based on the object itself. ")
        self.FORBIDDEN_SENTENCES = FORBIDDEN_SENTENCES
        self.OBJECT_IDENTIFIERS = pandas.DataFrame(OBJECT_IDENTIFIERS)
        self.rule_map = {
            "pos_traj_rw":"The sentence is describing object X in environment and the trajectory interacted with object X",
            "neg_traj_rw":"The sentence is describing object Y in environment and the trajectory interacted with object X",
            "plot_title":"Reward Distribution when the task is the same but the object is changed in the contrastive sentence For Pouring Task"
        }

    def make_pouring_data(self,dataframe:pandas.DataFrame,size):
        bowl_types = self.OBJECT_IDENTIFIERS[self.OBJECT_IDENTIFIERS['Type'] =='bowl']
        return_data = []
        
        presentids = dataframe['target_id'].unique()
        bowl_types = bowl_types[bowl_types['ID'].apply(lambda x: x in presentids)]

        while len(return_data) < size:
            # Pick some pouring demo
            t1obj = dataframe.sample(1)
            idxa = t1obj.index[0]
            t1_targid = t1obj.iloc[0]['target_id']
            t1_obj = bowl_types[bowl_types['ID'] == t1_targid]
            # Find all other objects which don't match the demo object's shape size and color  
            not_t1_mask = bowl_types.apply(lambda x: x['Shape']!= t1_obj.iloc[0]['Shape'] and x['Size']!= t1_obj.iloc[0]['Size'] and x['Color']!=t1_obj.iloc[0]['Color'],axis=1)
            if len(bowl_types[not_t1_mask]) == 0:
                print(f"Category {t1_targid} Has No Counter IN List")
                continue
            t2_targid = bowl_types[not_t1_mask].sample(1).iloc[0]['ID']
            # Filter data based on the object's which didn't match demo 1's criterea and make training tuple. 
            t2obj = dataframe[dataframe['target_id'] == t2_targid].sample(1)
            idxb = t2obj.index[0]
            return_data.append(
                (idxa,idxb)
            )

        return return_data



    def _execute_rule(self, metadf: pandas.DataFrame, num_samples_per_rule: int = 1000) -> List[Tuple[str, str]]:
        ddf = metadf[metadf['voice'].apply(lambda x:x not in self.FORBIDDEN_SENTENCES)]
        demo_grp = ddf[ddf['demo_type'] ==1] 
        return self.make_pouring_data(demo_grp,num_samples_per_rule)

class PickingObjectContrastingRule(SampleContrastingRule):
    """PickingObjectContrastingRule 
    Rule creating contrasting indices based object differences in picking task
    """

    def __init__(self,):
        super().__init__(description="Rule creating contrasting indices based on shapes and sizes of the objects for pouring. Picking will be done based on the object itself. ")
        self.FORBIDDEN_SENTENCES = FORBIDDEN_SENTENCES
        self.OBJECT_IDENTIFIERS = pandas.DataFrame(OBJECT_IDENTIFIERS)
        self.rule_map = {
            "pos_traj_rw":"The sentence is describing object X in environment and the trajectory interacted with object X",
            "neg_traj_rw":"The sentence is describing object Y in environment and the trajectory interacted with object X",
            "plot_title":"Reward Distribution when the task is the same but the object is changed in the contrastive sentence For Picking Task"
        }

    def make_picking_data(self,demo_grp,size):
      return_indexs = []
      sets=[]
      non_sampled_df = demo_grp.groupby(['target_id'])
      for _, idxs in non_sampled_df.groups.items():
          sets.append(idxs)
      for i in range(size):
        # Select any two group of indexes with different target_id
        set0, set1 = random.sample(sets, 2)
        # Select any two indexes from the selected group of indexes.
        return_indexs.append((random.choice(set0), random.choice(set1)))

      return return_indexs
    

    def _execute_rule(self, metadf: pandas.DataFrame, num_samples_per_rule: int = 1000) -> List[Tuple[str, str]]:
        ddf = metadf[metadf['voice'].apply(lambda x:x not in self.FORBIDDEN_SENTENCES)]
        demo_grp = ddf[ddf['demo_type'] ==0] 
        return self.make_picking_data(demo_grp,num_samples_per_rule)


class SameObjectPouringIntensityRule(SampleContrastingRule):
    """SameObjectPouringIntensityRule 
    Rule creates contrasting indexes for the pouring task with Little/Lot variations. 
    This is for same object! as ContrastingObjectRule already contrasts different
    `target_id`s. 
    """

    def __init__(self,):
        super().__init__(description="Variation in the intensity of pouring to create contrastive samples")
        self.FORBIDDEN_SENTENCES = FORBIDDEN_SENTENCES
        self.rule_map = {
            "pos_traj_rw":"The sentence is telling the robot to pour little and the robot did the same. Same for pouring a lot.",
            "neg_traj_rw":"The sentence is telling the robot to pour little but the robot pours a lot. Visa-Versa for little",
            "plot_title":"Reward Distribution in the pour Little vs lot case"
        }

    def _execute_rule(self, metadf: pandas.DataFrame, num_samples_per_rule: int = 1000) -> List[Tuple[str, str]]:
        sets = []
        ddf = metadf[metadf['voice'].apply(lambda x:x not in self.FORBIDDEN_SENTENCES)]        
        process_df = ddf[ddf['demo_type'] == 1]
        # num_top_type_grp = len(metadf.groupby(['demo_type']))
        num_top_type_grp = len(process_df.groupby(['target_id']))
        all_idxs = []
        sets = []
        
        for tgid, tgidgroup in process_df.groupby(['target_id']):
          # Make contrastive samples based on different pouring amounts for same type of object
          amount_sets = []
          for pouramt, pagroup_idxs in tgidgroup.groupby(['pouring_amount']).groups.items():
            amount_sets.append(pagroup_idxs)
            # sets.append(grpidxs)
          
          for i in range(int(num_samples_per_rule/num_top_type_grp)):
              # Select any of indexes with different pouring_amount for same target
              set0, set1 = amount_sets
              # Select any two indexes from the selected group of indexes.
              all_idxs.append((random.choice(set0), random.choice(set1)))

        return all_idxs
    
    def _validate_rule(self,demo_a:pandas.Series,demo_b:pandas.Series):
        return demo_a['demo_type'] == 1 and demo_a['pouring_amount'] != demo_b['pouring_amount'] and demo_a['target_id'] == demo_b['target_id']




POSSIBLE_RULES = [
    ContrastingActionsRule,
    PouringShapeSizeContrast,
    PickingObjectContrastingRule,
    SameObjectPouringIntensityRule,
]


@dataclass
class ContrastiveControlParameters:
    """ [summary]
    num_train_samples : size of the contrastive indices based training set to generate. if `train_rule_size_distribution` is not provided then it will divide the samples of each rule evenly. 
    num_test_samples  : size of the contrastive indices based testing set to generate. if `test_rule_size_distribution` is not provided then it will divide the samples of each rule evenly. 

    test_rule_size_distribution : If empty then num_train_samples are used else need to be the size of the `rules` array
    train_rule_size_distribution : If empty then num_test_samples are used else need to be the size of the `rules` array

    total_train_demos : Number of demonstration from which to create the contrastive `num_train_samples`. 
    total_test_demos : Number of demonstration from which to create the contrastive `num_test_samples`. 

    rules:List[SampleContrastingRule] : Instantiated rules for creating the contrasting dataset. 

    cache_main: boolean to signify if we should cache records from the main dataset. 

    """
    num_train_samples: int = 10000
    num_test_samples: int = 10000
    rules: List[SampleContrastingRule] = field(default_factory=lambda: [r() for r in POSSIBLE_RULES])
    train_rule_size_distribution: List[int] = field(default_factory=lambda: [])
    test_rule_size_distribution: List[int] = field(default_factory=lambda: [])
    total_train_demos: int = 12000
    total_test_demos: int = 4000
    cache_main:bool = True
    created_on:str = None

    def __post_init__(self):

        if len(self.train_rule_size_distribution) > 0:
            assert len(self.train_rule_size_distribution) == len(self.rules)

        if len(self.test_rule_size_distribution) > 0:
            assert len(self.test_rule_size_distribution) == len(self.rules)
        self.created_on = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def to_json(self):
        data_dict = {**self.__dict__}
        data_dict['rules'] = [r.rule_name for r in data_dict['rules']]
        return data_dict

    @classmethod
    def from_json(cls,json_dict:dict):
        assert 'rules' in json_dict
        rules = []
        for rule in json_dict['rules']:
            chosen_rule = [x for x in POSSIBLE_RULES if x.__name__ == rule]
            if len(chosen_rule) == 0:
                print(f"Couldn't Find Rule : {rule}")
                continue
            rules.append(chosen_rule[0]())
        
        json_dict['rules'] = rules
        return cls(**json_dict)


DEFAULT_CONTROL_PARAMS = ContrastiveControlParameters(
    rules= [r() for r in POSSIBLE_RULES],
    num_train_samples=50000,
    num_test_samples=1000
)


class HDF5ContrastiveSetCreator:
    """ 
    HDF5ContrastiveSetCreator : Creates the dataset holding the contrasting indices. \n
    CORE Rules to Contrast two examples i,j are below and attributes are derived from MAIN_KEYS:
      1. `i_demo_type` != `j_demo_type`
        1. Variation in the type of task creates contrastive samples
      2. `i_demo_type` == `j_demo_type` && `i_target_id` != `j_target_id`
        1. If the task is the same the variation in the target object creates contrastive examples
      3. `i_demo_type` == `j_demo_type` && `demo_type` == `pouring` && `i_target_id` == `j_target_id` && `i_amount` != `j_amount`
        1. Variation in the amounts volume in pour creates contrastive samples

    USAGE : 
    ```
        MAIN_META_PTH = "<PATH_TO_METADATA_CSV>"
        DEMO_DS_FILE = "<PATH_TO_MASSIVE_HDF5_DEMOSTRATION_DATASET>"
        STORING_FOLDER = "<PATH_TO_FOLDER_WHERE_WE_STORE_DATASET>"
        # STORING_FOLDER shouldn't exist
        contrastive_datamaker = HDF5ContrastiveSetCreator(MAIN_META_PTH,DEMO_DS_FILE,)
        contrastive_datamaker.make_dataset(STORING_FOLDER)
    ```
    """

    INIT_META_COLUMNS = [
        'main_name',
        'phase',
        'ints',
        'num_bowls',
        'num_cups',
        'comes_from',
        'voice',
        'target/id',
        'target/type',
        'amount'
    ]

    MAPPING_COLUMNS = {
        'phase': 'demo_type',
        'ints': 'object_meta',
        'num_bowls': 'num_bowls',
        'num_cups': 'num_cups',
        'comes_from': 'demo_name',
        'voice': 'voice',
        'target/id': 'target_id',
        'target/type': 'target_type',
        'amount': 'pouring_amount'
    }

    MAIN_KEYS = [
        'chunk_name',
        'demo_type',
        'object_meta',
        'num_bowls',
        'num_cups',
        'demo_name',
        'voice',
        'target_id',
        'target_type',
        'pouring_amount'
    ]

    MAIN_IDENTIFIER_NAME = 'demo_name'

    CONTRASITIVE_SET_FILENAME = 'contrastiveset.hdf5'
    CONTRASITIVE_META_FILENAME = 'contrastiveset.csv'
    CREATION_PARAMS_FILENAME = 'creation_params.json'

    TRAIN_PREFIX = 'train_'
    TEST_PREFIX = 'test_'

    HDF5_DATASET_NAME_MAIN_DEMO = CONTRASTIVE_HDF5_DATASET_NAME_MAIN_DEMO
    HDF5_DATASET_NAME_CACHE_IDX = CONTRASTIVE_HDF5_DATASET_NAME_CACHE_INDICES

    def __init__(self,
                 metafile_path: str,
                 core_demostrations_hdf5pth: str,
                 chunk_size=512,
                 control_params: ContrastiveControlParameters = DEFAULT_CONTROL_PARAMS) -> None:

        assert is_present(metafile_path)
        assert is_present(core_demostrations_hdf5pth)
        metadf = pandas.read_csv(metafile_path)
        print(set(self.INIT_META_COLUMNS) - set(metadf.columns))
        assert set(self.INIT_META_COLUMNS).issubset(metadf.columns)
        self.metadf: pandas.DataFrame = metadf.rename(
            columns=self.MAPPING_COLUMNS)
        self.demo_dataset = DemonstrationsDataset(core_demostrations_hdf5pth)
        self.id_list = self.demo_dataset.id_list
        self.control_params = control_params
        self.chunk_size=chunk_size
        self.logger = create_logger(self.__class__.__name__)

    def _partition_train_test(self):
        # $ create train test partition by using completely unseen demonstration in the test set.
        train_filter_rows = random.sample(
            list(self.metadf.index), self.control_params.total_train_demos)
        train_rows = self.metadf.iloc[train_filter_rows]
        left_rows = self.metadf.drop(index=train_filter_rows)
        test_rows = left_rows.sample(self.control_params.total_test_demos)
        return train_rows, test_rows

    def _create_distributon(self, dataframe: pandas.DataFrame, train=False):
        """create_distributon [summary]
        Create the distribution of contrastive samples index from metadata dataframe based on the rules stored in 
        `self.control_params.rules` and `self.control_params.num_train_samples`
        :param dataframe: the core metadata-dataframe from which indexes of the contrastive samples will be extracted
        :type dataframe: pandas.DataFrame
        """
        num_rules = len(self.control_params.rules)
        # $ If we have `train_rule_size_distribution` or `test_rule_size_distribution` set then use them.
        if train and len(self.control_params.train_rule_size_distribution) > 0:
            size_dis = self.control_params.train_rule_size_distribution
        elif not train and len(self.control_params.test_rule_size_distribution) > 0:
            size_dis = self.control_params.test_rule_size_distribution
        else:
            # $ If we dont have `train_rule_size_distribution` or `test_rule_size_distribution` set then use balanced distribution based on rules.
            if train:
                per_rule_size = int(
                    self.control_params.num_train_samples/num_rules)
            else:
                per_rule_size = int(
                    self.control_params.num_test_samples/num_rules)
            size_dis = [per_rule_size for _ in range(num_rules)]

        all_indices = []
        # $ Use rule to create contrastive indices
        for size, rule in zip(size_dis, self.control_params.rules):
            created_indices = rule(dataframe, num_samples_per_rule=size)
            all_indices.extend(created_indices)

        return all_indices


    def _make_join_data_for_indices(self,sample_indices:List[List[int]],chunk_size=256):
        unique_idxes = list(set([x for i in sample_indices for x in i]))
        chunked_demo_indexes = self.make_chunks_based_indexes(unique_idxes,chunk_size=chunk_size)
        # $ Extract the ids,seqs and msk from the main dataset and filter the chunks using the `chunked_demo_indexes` 
        self.logger.info("Retrieving Chunks")
        id_chunks,seq_chunks,msk_chunks = self._retrieve_sequence_and_masks(chunked_demo_indexes)
        self.logger.info("Retrieved Chunks")
        concat_seq_dict = {k:[] for k in seq_chunks[0].keys()}
        concat_msk_dict = {k:[] for k in msk_chunks[0].keys()}
        for chk in seq_chunks:
            for k in chk:
                concat_seq_dict[k].append(chk[k])

        for chk in msk_chunks:
            for k in chk:
                concat_msk_dict[k].append(chk[k])
        # $ Concatenate all the data. 
        for k in concat_seq_dict:
            concat_seq_dict[k] = np.concatenate(concat_seq_dict[k])
        
        for k in concat_msk_dict:
            concat_msk_dict[k] = np.concatenate(concat_msk_dict[k])
        
        id_list = [i for ix in id_chunks for i in  ix]
        return concat_seq_dict,concat_msk_dict,sorted(unique_idxes),id_list

    def _save_hdf5_file(self, save_path: str, sample_indices: List[List[int]],cache_main=True,chunk_size=256):
        with h5py.File(save_path, 'w') as contrastive_ds_file:
            # $ Store info from main demonstration store
            contrastive_ds_file.create_dataset(f'{self.HDF5_DATASET_NAME_MAIN_DEMO}',
                             data=np.array(sample_indices), dtype='i')
            if not cache_main:
                return 
            
            self.logger.info("Caching Records From Main Dataset")
            # $ Create a cache of the samples from the demonstration dataset by filtering via chunks
            concat_seq_dict,\
            concat_msk_dict,\
            sorted_demo_idxs,\
            id_list = self._make_join_data_for_indices(sample_indices,chunk_size=chunk_size)
            self.logger.info(f"Data is Joined Saving in {contrastive_ds_file}")

            seq_grp = contrastive_ds_file.create_group(GROUPNAMES.sequences)
            msk_grp = contrastive_ds_file.create_group(GROUPNAMES.masks)
            data_types = get_hd5_dtypes()
        
            for channel in concat_seq_dict:
                seq_grp.create_dataset(
                    channel,
                    dtype=data_types[channel],
                    chunks=True,
                    data=concat_seq_dict[channel],
                )

            for channel in concat_msk_dict:
                msk_grp.create_dataset(
                    channel,
                    dtype=data_types[channel],
                    chunks=True,
                    data=concat_msk_dict[channel],
                )

            # Remap the indexs from the demonstration to the actual indexes in the cache.
            demo_to_cache_map = {dem_idx:idx for idx,dem_idx in enumerate(sorted_demo_idxs)}
            cache_indices = []
            for idx_pair in sample_indices:
                p1,p2 = idx_pair
                cache_indices.append(
                    [demo_to_cache_map[p1],demo_to_cache_map[p2]]
                )
            # $ This creates a new mapping of the contrastive indices based on the cache.
            contrastive_ds_file.create_dataset(f'{self.HDF5_DATASET_NAME_CACHE_IDX}',
                             data=np.array(cache_indices), dtype='i')

            contrastive_ds_file.create_dataset(GROUPNAMES.id_list,data=id_list,chunks=True,dtype='S20')
            

    def _retrieve_sequence_and_masks(self,chunked_indexes:List[List[int]]):
        id_chunks,msk_chunks,seq_chunks = [],[],[]
        for idx,chunk in enumerate(chunked_indexes):
            id_list_chunk = self._get_ids(chunk)
            mask_chunk_dict = self._get_mask(chunk)
            sequence_chunk_dict = self._get_sequence(chunk)
            self.logger.info(f"Completed Extracting Chunk {idx} Of Size {len(id_list_chunk)} ")

            id_chunks.append(id_list_chunk)    
            msk_chunks.append(mask_chunk_dict)
            seq_chunks.append(sequence_chunk_dict)
        return id_chunks,seq_chunks,msk_chunks


    def _get_ids(self,chunk) -> np.ndarray:
        return_chunk = self.demo_dataset.id_list[chunk]
        return return_chunk

    def _get_sequence(self,chunk) -> Dict[str,np.ndarray]:
        ret_dict = {}
        for dataset_name in self.demo_dataset.sequences.keys():
            return_chunk = self.demo_dataset.sequences[dataset_name][chunk]
            ret_dict[dataset_name] = return_chunk
        return ret_dict

    def _get_mask(self,chunk) -> Dict[str,np.ndarray]:
        ret_dict = {}
        for dataset_name in self.demo_dataset.masks.keys():
            return_chunk = self.demo_dataset.masks[dataset_name][chunk]
            ret_dict[dataset_name] = return_chunk
        return ret_dict
    
    @staticmethod
    def make_chunks_based_indexes(index_list: List[int], chunk_size=256):
        # This will first make the index chunks with a range of 256 so that we can efficiently extract data from hdf5
        # Sorting so indexing chunks is easy
        sorted_idxs = sorted(index_list)
        curr_chunk_multiple = 1
        chunked_arr = []
        curr_arr = []
        while len(sorted_idxs) > 0:
            poped_idx = sorted_idxs.pop(0)
            # Make new chunk if exceed current chunk index else add to current chunk 
            if poped_idx > chunk_size * curr_chunk_multiple:
                curr_chunk_multiple += 1
                chunked_arr.append(curr_arr)
                curr_arr = [poped_idx]
            else:
                curr_arr.append(poped_idx)
        if len(curr_arr) > 0:
            chunked_arr.append(curr_arr)
        return chunked_arr

    def _save_contrastive_samples(self,
                                  save_path: str,
                                  dataframe: pandas.DataFrame,
                                  sample_indices: List,
                                  chunk_size=256,
                                  train=False):
        meta_path = None
        datasetpth = None
        if train:
            datasetpth = self.TRAIN_PREFIX + self.CONTRASITIVE_SET_FILENAME
            meta_path = self.TRAIN_PREFIX + self.CONTRASITIVE_META_FILENAME
        else:
            datasetpth = self.TEST_PREFIX + self.CONTRASITIVE_SET_FILENAME
            meta_path = self.TEST_PREFIX + self.CONTRASITIVE_META_FILENAME

        dataframe.to_csv(os.path.join(save_path, meta_path))
        dataset_savepth = os.path.join(save_path, datasetpth)
        self._save_hdf5_file(dataset_savepth,\
                                    sample_indices,\
                                    chunk_size=chunk_size,\
                                    cache_main=self.control_params.cache_main)

    def make_dataset(self, save_path: str,chunk_size=256):
        """make_dataset [summary]
        DemonstrationDataset : Dataset with all the demos. HDF5 File created by `H5DataCreatorMainDataCreator`.
        Function does following steps : 
            - `_partition_train_test` : partition the dataset to train/test via metadata dataframe
            - `_create_distributon` : create the distriubtion of contrastive samples based on the `SampleContrastingRule`. Returns the indices of the data found in dataframes
            - `collate_indices` : Correlate the indices of the dataframe with the ID values within the dataframe and make list of id pairs of contrastives samples.
            - `map_to_contrastive_indexes_to_ids` : Maps the ID values of filtered contrastive pairs to Indexes in the `DemonstrationDataset` using an index map
            - `_save_contrastive_samples` : Saves the dataset to a HDF5 File along with thier metadata Files to the `save_path`
        :param save_path: Path to folder. Has to not exist so dataset is created and populated within that.
        :type save_path: str
        """
        assert not is_present(save_path)
        # $ First find partitions for the train and the test set samples from the Meta-dataframe.
        train_df, test_df = self._partition_train_test()
        # $ Apply rules to create the contrastive indices of individual train/test meta dataframes
        train_indices = self._create_distributon(train_df, train=True)
        test_indices = self._create_distributon(test_df, train=False)
        # $ Using indices created by the rules, Find the Ids' they Belong to in their dataframes .
        train_collated_ids = collate_indices(train_df, train_indices,identifier_name=self.MAIN_IDENTIFIER_NAME)
        test_collated_ids = collate_indices(test_df, test_indices,identifier_name=self.MAIN_IDENTIFIER_NAME)
        # $ Find the mapping index of the ids in the `self.id_list`. It will provide indexes in the main demonstration dataset
        # $ Make a dictionary which hold the index positions to use it .
        index_map = {k.decode('utf-8'): idx for idx,
                     k in enumerate(self.demo_dataset.id_list)}
        train_sample_indexes = map_to_contrastive_indexes_to_ids(
            train_collated_ids, index_map)
        test_sample_indexes = map_to_contrastive_indexes_to_ids(
            test_collated_ids, index_map)
        # $ Save the contrastive indices of the into last hdf5 file and
        # $ also save the Metadata about the extracted Meta data of the test/trainset.
        safe_mkdir(save_path)
        self._save_contrastive_samples(
            save_path, train_df, train_sample_indexes, train=True,chunk_size=chunk_size)
        self._save_contrastive_samples(
            save_path, test_df, test_sample_indexes, train=False,chunk_size=chunk_size)
        # $ Save the control parameters which created the dataset.
        save_json_to_file(self.control_params.to_json(), os.path.join(
            save_path, self.CREATION_PARAMS_FILENAME))
