import pandas
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import torch
import os
import h5py
from typing import List
import matplotlib as plt


from ..channel import ChannelData
from .utils import \
    load_json_from_file,\
    save_json_to_file

from .dataset import \
    GROUPNAMES,\
    ContrastiveCollateFn,\
    is_present,\
    collate_indices,\
    map_to_contrastive_indexes_to_ids,\
    USE_CHANNELS,\
    CONTRASTIVE_HDF5_DATASET_NAME_CACHE_INDICES

from .datamaker import \
    HDF5ContrastiveSetCreator,\
    ContrastiveControlParameters,\
    SampleContrastingRule


class SentenceContrastiveDataset(Dataset):
    """SentenceContrastiveDataset 
    Derives From the data created by the `HDF5ContrastiveSetCreator`

    - This class will help make a configurable dataset using different rules about the 
    core dataset made by `HDF5ContrastiveSetCreator`
    
    - Key Data Attributes : 
        - contrastive_data:
            - contrastive_indices : Hold the list of indicies to master_data:DEMOS
            - indices_rules : Holds the string in rules
        - master_data
            - id_list : hold the actual list of Ids 
            - sequences : HOLD information channels. 
            - masks : holds the masks. 
    - Instantiating Params:
        - `contrastive_set_generated_folder`:str
            - Folder which hold the data created by `HDF5ContrastiveSetCreator`
        - `use_channels`=USE_CHANNELS
            - Information chaannels to consider for the experiment. 
        - `train`=True
            - Load training set / test set
        - `use_original_contrastive_indices`:bool=False
            - use the `CONTRASTIVE_HDF5_DATASET_NAME_CACHE_INDICES` from the dataset
        - `size`:int=200
            - if `use_original_contrastive_indices == False` then the size of the 
            contrastive samples to filter from the dataset. 
    
    - Important methods 
        - `remake_indices` : Provide new custom rules to remake the indicies. 

    """
    def __init__(self,\
                contrastive_set_generated_folder:str,\
                use_channels=USE_CHANNELS,\
                train=True,\
                normalize_images=False,\
                use_original_contrastive_indices:bool=True,\
                size:int=200) -> None:
        super().__init__()
        from torchvision import transforms
        self.normalize_images= normalize_images
        self.norm_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])
        self._open_dataset(contrastive_set_generated_folder,\
                            size,\
                            use_channels=use_channels,\
                            train=train,\
                            use_original_contrastive_indices=use_original_contrastive_indices)

    def _open_dataset(self,\
                    folder_pth:str,\
                    size:int,\
                    use_original_contrastive_indices=True,\
                    use_channels=USE_CHANNELS,\
                    train=True):
        prefix = HDF5ContrastiveSetCreator.TRAIN_PREFIX if train else HDF5ContrastiveSetCreator.TEST_PREFIX
        metapth = os.path.join(\
            folder_pth,\
            prefix + HDF5ContrastiveSetCreator.CONTRASITIVE_META_FILENAME
        )
        hdf5pth = os.path.join(\
            folder_pth,\
            prefix + HDF5ContrastiveSetCreator.CONTRASITIVE_SET_FILENAME
        )
        control_parameter_pth = os.path.join(
            folder_pth,\
            HDF5ContrastiveSetCreator.CREATION_PARAMS_FILENAME
        )
        assert is_present(hdf5pth), f"Contrastive Set {hdf5pth} should exist!"
        assert is_present(metapth), f"Contrastive Set Metadata {metapth} should exist and is Required by test Set"
        assert is_present(control_parameter_pth), f"Contrastive Set Creation Control {control_parameter_pth} should exist and is Required by test Set"
        # $ HDF5 Loading
        self.h5 = h5py.File(hdf5pth,'r')
        self.id_list = list(self.h5.get(GROUPNAMES.id_list))
        self.sequences = self.load_sequences(self.h5.get(GROUPNAMES.sequences),use_channels)
        self.masks = self.load_sequences(self.h5.get(GROUPNAMES.masks),use_channels,mask=True)
        # $ Metadata Loading
        self.dataset_meta = pandas.read_csv(metapth)
        self.control_parameters = ContrastiveControlParameters.from_json(
            load_json_from_file(control_parameter_pth)
        )
        # $ us original indicies 
        if use_original_contrastive_indices:
            self.contrastive_indices = list(self.h5.get(CONTRASTIVE_HDF5_DATASET_NAME_CACHE_INDICES))
        else:
            # $ DONT load the initial indices from the h5 dataset if `use_original_contrastive_indices` set as false
            self.remake_indices(size)
        print(
            f'''
            Dataset Loaded \n
            Data Set Contains information from the Following Rules : \n
            {
                ', '.join([r.rule_name for r in self.control_parameters.rules])
            }
            Used Originally Stored Indices : {use_original_contrastive_indices}
            '''
        )
        
    
    def remake_indices(self,size:int,rules=[]):
        """remake_indices 
        This method will rerun the rules on the self.dataset_meta
        """
        # $ Make a map of the indices of the ids in the dataset. 
        index_map = {k.decode('utf-8'): idx for idx,
                     k in enumerate(self.id_list)}
        # $ extract the indicies and  
        input_rules = self.control_parameters.rules
        if len(rules) > 0 : 
            input_rules = rules
        contrastive_df_indices , rule_distribution = self._make_indices(self.dataset_meta,size,input_rules)
        collated_id_indices = collate_indices(
            self.dataset_meta,
            contrastive_df_indices,
            identifier_name= HDF5ContrastiveSetCreator.MAIN_IDENTIFIER_NAME
        )
        dataset_indices = map_to_contrastive_indexes_to_ids(
            collated_id_indices,index_map
        )
        self.contrastive_indices = dataset_indices
        self.indices_rules = rule_distribution
        
    @staticmethod
    def _make_indices(dataframe:pandas.DataFrame,size:int,rules:List[SampleContrastingRule]):
        """_make_indices [summary]
        Takes a dataframe , runs the list of `SampleContrastingRule`'s to create contrastive indices of a total certain `size`
        returns : Tuple(all_indices,rule_distribution)
            - `all_indices` : List[List[int,int]]
            - `rule_distribution` : List[str] : list where each index corresponds to what rule created that contrastive sample. . 
        """
        size_dis = [ 
            int(size/len(rules))\
                for _ in range(len(rules))
        ]
        all_indices = []
        rule_distribution = []
        # $ Use rule to create contrastive indices
        for size_value, rule in zip(size_dis, rules):
            created_indices = rule(dataframe, num_samples_per_rule=size_value)
            rule_distribution.extend([rule.rule_name for _ in range(len(created_indices))])
            all_indices.extend(created_indices)

        return all_indices , rule_distribution
    
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
            seq = torch.from_numpy(self.sequences[k][index])
            if k == 'image_sequence' and self.normalize_images: # Monkey Patch
                seq_frames = []
                for frame in seq:
                    seq_frames.append(self.norm_transform(frame))
                seq = torch.stack(seq_frames)
                
            channel_dict[k] = ChannelData(
                mask=mask,
                sequence=seq,
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

