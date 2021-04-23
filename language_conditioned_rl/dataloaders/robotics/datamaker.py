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
    RoboDataUtils,\
    DemonstrationsDataset
import gc
import logging


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


class SampleContrastingRule(metaclass=abc.ABCMeta):
    """SampleContrastingRule 
    The Goal of this class is to apply the rules of contrasting on the Metadata 
    DataFrame and extract the contrasting indices based on the rule. 
    `num_samples_per_rule` controls the number of indices that will get extracted using this contrastive matching rule. 
    """

    def __init__(self, description=None):
        self.description = description
        self.rule_name = self.__class__.__name__

    def __call__(self, metadf: pandas.DataFrame, num_samples_per_rule: int = 1000) -> List[Tuple[str, str]]:
        assert set(HDF5ContrastiveSetCreator.MAPPING_COLUMNS.values()).issubset(set(metadf.columns)), f'{set(metadf.columns)} Not a Part of {set(HDF5ContrastiveSetCreator.MAPPING_COLUMNS.values())}'
        return self._execute_rule(metadf, num_samples_per_rule=num_samples_per_rule)

    def _execute_rule(self, metadf: pandas.DataFrame, num_samples_per_rule: int = 100) -> List[Tuple[str, str]]:
        raise NotImplementedError()


class ContrastingActionsRule(SampleContrastingRule):
    """ContrastingActionsRule 
    Rule creates contrasting indexes based on examples which are different by task_type : Picking Vs Pouring
    """

    def __init__(self,):
        super().__init__(description="Variation in the type of task creates contrastive samples")

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


class ContrastingObjectRule(SampleContrastingRule):
    """ContrastingObjectRule 
    Rule creates contrasting indexes based on examples which are of same task but different objects
    """

    def __init__(self,):
        super().__init__(description="Same Task But Variation in the type of target object is used to create contrastive samples")

    def _execute_rule(self, metadf: pandas.DataFrame, num_samples_per_rule: int = 1000) -> List[Tuple[str, str]]:
        num_top_type_grp = len(metadf.groupby(['demo_type']))
        all_idxs = []
        for grp, df in metadf.groupby(['demo_type']):
            sets = []
            # Group by target id and collect all indexes of groups as a list
            type_group = df.groupby(['target_id'])
            for tid, idxs in type_group.groups.items():
                sets.append(idxs)
            # sets contains the lists of list. Each item in sets is a list of indexes with a specific target_id
            for i in range(int(num_samples_per_rule/num_top_type_grp)):
                # Select any two group of indexes with different target_id
                set0, set1 = random.sample(sets, 2)
                # Select any two indexes from the selected group of indexes.
                all_idxs.append((random.choice(set0), random.choice(set1)))
        return all_idxs


class PouringIntensityRule(SampleContrastingRule):
    """PouringIntensityRule 
    Rule creates contrasting indexes for the pouring task with Little/Lot variations. 
    """

    def __init__(self,):
        super().__init__(description="Variation in the intensity of pouring to create contrastive samples")

    def _execute_rule(self, metadf: pandas.DataFrame, num_samples_per_rule: int = 1000) -> List[Tuple[str, str]]:
        process_df = metadf[metadf['demo_type'] == 1]
        # num_top_type_grp = len(metadf.groupby(['demo_type']))
        num_top_type_grp = len(process_df.groupby(['pouring_amount']))
        all_idxs = []
        sets = []
        for grp, grpidxs in process_df.groupby(['pouring_amount']).groups.items():
            sets.append(grpidxs)

        for i in range(int(num_samples_per_rule)):
            # Select any two group of indexes with different pouring_amount
            set0, set1 = random.sample(sets, 2)
            # Select any two indexes from the selected group of indexes.
            all_idxs.append((random.choice(set0), random.choice(set1)))

        return all_idxs


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


    """
    num_train_samples: int = 10000
    num_test_samples: int = 10000
    rules: List[SampleContrastingRule] = field(default_factory=lambda:[])
    train_rule_size_distribution: List[int] = field(default_factory=lambda:[])
    test_rule_size_distribution: List[int] = field(default_factory=lambda:[])

    total_train_demos: int = 12000
    total_test_demos: int = 4000

    def __post_init__(self):

        if len(self.train_rule_size_distribution) > 0:
            assert len(self.train_rule_size_distribution) == len(self.rules)

        if len(self.test_rule_size_distribution) > 0:
            assert len(self.test_rule_size_distribution) == len(self.rules)


DEFAULT_CONTROL_PARAMS = ContrastiveControlParameters(
    rules=[
        ContrastingActionsRule(),
        ContrastingObjectRule(),
        PouringIntensityRule(),
    ],
    num_train_samples=10000,
    num_test_samples=1000
)


class HDF5ContrastiveSetCreator:
    """ 
    HDF5ContrastiveSetCreator : Creates the dataset holding the contrasting indices. \n
    CORE Rules to Contrast two examples i,j :
      1. `i_demo_type` != `j_demo_type`
        1. Variation in the type of task creates contrastive samples
      2. `i_demo_type` == `j_demo_type` && `i_target_id` != `j_target_id`
        1. Variation in the target object apon which task is done creates contrastive examples
      3. `i_demo_type` == `j_demo_type` && `demo_type` == `pouring` && `i_target_id` == `j_target_id` && `i_amount` != `j_amount`
        1. Variation in the amounts volume in pour creates contrastive samples
    """

    INIT_META_COLUMNS = [
        'name',
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
        'name': 'chunk_name',
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

    TRAIN_PREFIX ='train_'
    TEST_PREFIX ='test_'

    HD5DATASET_NAME = 'contrastive_ids'

    def __init__(self,
                 metafile_path: str,
                 core_demostrations_hdf5pth: str,
                 control_params: ContrastiveControlParameters = DEFAULT_CONTROL_PARAMS) -> None:

        assert is_present(metafile_path)
        assert is_present(core_demostrations_hdf5pth)
        metadf = pandas.read_csv(metafile_path)
        assert set(self.INIT_META_COLUMNS).issubset(metadf.columns)
        self.metadf:pandas.DataFrame = metadf.rename(columns=self.MAPPING_COLUMNS) 
        self.demo_dataset = DemonstrationsDataset(core_demostrations_hdf5pth)
        self.id_list = self.demo_dataset.id_list
        self.control_params = control_params

    def _partition_train_test(self):
        # $ create train test partition by using completely unseen demonstration in the test set. 
        train_filter_rows = random.sample(list(self.metadf.index), self.control_params.total_train_demos)
        train_rows = self.metadf.iloc[train_filter_rows]
        left_rows = self.metadf.drop(index=train_filter_rows)
        test_rows = left_rows.sample(self.control_params.total_test_demos)
        return train_rows,test_rows

    def _create_distributon(self,dataframe:pandas.DataFrame,train=False):
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
            per_rule_size = int(self.control_params.num_train_samples/num_rules)
          else:
            per_rule_size = int(self.control_params.num_test_samples/num_rules)
          size_dis = [per_rule_size for _ in range(num_rules)]
          
        all_indices = []
        # $ Use rule to create contrastive indices
        for size,rule in zip(size_dis,self.control_params.rules):
          created_indices = rule(dataframe,num_samples_per_rule=size)
          all_indices.extend(created_indices)

        return all_indices

    def _collate_indices(self,dataframe:pandas.DataFrame,indices:List[Tuple[int,int]]):
        """_collate_indices 
        Correlate the indices and store return the list of tuples with identifiers to the objects
        """
        collated_id_data = []
        for idx_tuple in indices:
          pos_idx,neg_idx = idx_tuple
          pos_obj,neg_obj = dataframe.iloc[dataframe.index.get_loc(pos_idx)],\
                                dataframe.iloc[dataframe.index.get_loc(neg_idx)]
          
          posid,negid = pos_obj[self.MAIN_IDENTIFIER_NAME],\
                            neg_obj[self.MAIN_IDENTIFIER_NAME]
          collated_id_data.append(
            (posid,negid)
          )
        
        return collated_id_data

    def _map_to_demo_indexes(self,collated_ids:List[Tuple[str,str]],index_map:Dict[str,int]):
        """_map_to_demo_indexes 
        collated_ids : List of tuples with pos/neg ids in them 
        index_map : dictionary to map strings in `self.id_list` to index so that it can be used to help collate indexes for indexdata
        """
        mapped_arr = []
        for indexes in collated_ids:
          pid,nid = indexes
          mapped_arr.append([index_map[pid],index_map[nid]])
        return mapped_arr
    
    def _save_contrastive_set(self,save_path:str,sample_indices:List):
        with h5py.File(save_path,'w') as f:
          f.create_dataset(self.HD5DATASET_NAME,data=np.array(sample_indices),dtype='i')


    def _save_contrastive_samples(self,\
                                  save_path:str,\
                                  dataframe:pandas.DataFrame,\
                                  sample_indices:List,\
                                  train=False):
        safe_mkdir(save_path)
        meta_path = None 
        datasetpth = None
        if train:
          datasetpth = self.TRAIN_PREFIX + self.CONTRASITIVE_SET_FILENAME
          meta_path = self.TRAIN_PREFIX + self.CONTRASITIVE_META_FILENAME
        else:
          datasetpth = self.TEST_PREFIX + self.CONTRASITIVE_SET_FILENAME
          meta_path = self.TEST_PREFIX + self.CONTRASITIVE_META_FILENAME
        
        dataframe.to_csv(os.path.join(save_path,meta_path))
        self._save_contrastive_set(os.path.join(save_path,datasetpth),sample_indices)
    

    def make_dataset(self,save_path):
        assert not is_present(save_path)
        # $ First find partitions for the train and the test set samples from the Meta-dataframe.
        train_df, test_df = self._partition_train_test()
        # $ Apply rules to create the contrastive indices of individual train/test meta dataframes
        train_indices = self._create_distributon(train_df,train=True)
        test_indices = self._create_distributon(test_df,train=False)
        # $ Using indices created by the rules, Find the Ids' they Belong to in their dataframes .
        train_collated_ids = self._collate_indices(train_df,train_indices)
        test_collated_ids = self._collate_indices(test_df,test_indices)
        # $ Find the mapping index of the ids in the `self.id_list`. It will provide indexes in the main demonstration dataset
        # $ Make a dictionary which hold the index positions to use it .
        index_map = { k.decode('utf-8'):idx for idx,k in enumerate(self.demo_dataset.id_list) }
        train_sample_indexes = self._map_to_demo_indexes(train_collated_ids,index_map)
        test_sample_indexes = self._map_to_demo_indexes(test_collated_ids,index_map)
        # $ Save the contrastive indices of the into last hdf5 file and 
        # $ also save the Metadata about the extracted Meta data of the test/trainset.
        self._save_contrastive_samples(save_path,train_df,train_sample_indexes,train=True)
        self._save_contrastive_samples(save_path,test_df,test_sample_indexes,train=True)
        # return train_df,test_df,train_indices,test_indices
