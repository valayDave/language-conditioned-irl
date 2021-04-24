import pandas
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import torch
import os
import h5py
from typing import List

from ..models.robotics.reward_model import LGRRoboRewardLearner
from ..dataloaders.channel import ChannelData,ChannelHolder,ContrastiveGenerator
from ..dataloaders.robotics.dataset import \
    GROUPNAMES,\
    ContrastiveCollateFn,\
    is_present,\
    load_json_from_file,\
    collate_indices,\
    map_to_contrastive_indexes_to_ids

from ..dataloaders.robotics.datamaker import \
    HDF5ContrastiveSetCreator,\
    ContrastiveControlParameters,\
    SampleContrastingRule
    
class RoboticsTestingDataset:
    pass

class TestSetCollateFn(ContrastiveCollateFn):
    """TestSetCollateFn 
    This collate function brings the metadata and along with the ids of the contrastive samples that are being compared. 
    BATCH : 
        (
            ContrastiveGenerator,
            rules : List[str],
            id_list : List[str],
        )
    """
    def __call__(self, batch):
        core_d1_channels = ChannelHolder()
        core_d2_channels = ChannelHolder()
        id_list = []
        rules = []
        for cont_dicts_tup in batch:
            d1_dict, d2_dict,batch_id_i,batch_id_j,btch_rule = cont_dicts_tup
            core_d1_channels = self.populate_channel(d1_dict, core_d1_channels)
            core_d2_channels = self.populate_channel(d2_dict, core_d2_channels)
            rules.append(btch_rule)
            id_list.append(
                (batch_id_i,batch_id_j)
            )

        core_d1_channels = self.stack_channels(core_d1_channels)
        core_d2_channels = self.stack_channels(core_d2_channels)
        return ContrastiveGenerator(core_d1_channels, core_d2_channels),rules,id_list


class RoboticsTestingDataset(Dataset):
    """RoboticsTestingDataset 
    Derives From HDF5ContrastiveSetCreator

    - This class will help make the test set configurable For differnt types of rule sets. 
    
    Key Data Attributes : 
        contrastive_data:
            contrastive_indices : Hold the list of indicies to master_data:DEMOS
            indices_rules : Holds the string in rules
        master_data
            id_list : hold the actual list of Ids 
            sequences : HOLD information channels. 
            masks : holds the masks. 

    """
    def __init__(self,\
                contrastive_set_generated_folder:str,\
                size:int=200) -> None:
        super().__init__()
        self._open_dataset(contrastive_set_generated_folder,size)

    def _open_dataset(self,folder_pth:str,size:int):
        test_metapth = os.path.join(\
            folder_pth,\
            HDF5ContrastiveSetCreator.TEST_PREFIX + HDF5ContrastiveSetCreator.CONTRASITIVE_META_FILENAME
        )
        test_hdf5pth = os.path.join(\
            folder_pth,\
            HDF5ContrastiveSetCreator.TEST_PREFIX + HDF5ContrastiveSetCreator.CONTRASITIVE_SET_FILENAME
        )
        control_parameter_pth = os.path.join(
            folder_pth,\
            HDF5ContrastiveSetCreator.TEST_PREFIX + HDF5ContrastiveSetCreator.CREATION_PARAMS_FILENAME
        )
        assert is_present(test_hdf5pth), f"Contrastive Set {test_hdf5pth} should exist!"
        assert is_present(test_hdf5pth), f"Contrastive Set Metadata {test_metapth} should exist and is Required by test Set"
        assert is_present(control_parameter_pth), f"Contrastive Set Creation Control {control_parameter_pth} should exist and is Required by test Set"
        # $ HDF5 Loading
        self.h5 = h5py.File(test_hdf5pth,'r')
        self.id_list = list(self.h5.get(GROUPNAMES.id_list))
        self.sequences = self.load_sequences(self.h5.get(GROUPNAMES.sequences))
        self.masks = self.load_sequences(self.h5.get(GROUPNAMES.masks))
        # self.contrastive_indices = list(self.h5.get(CONTRASTIVE_HDF5_DATASET_NAME_CACHE_INDICES))
        # $ Metadata Loading
        self.dataset_meta = pandas.DataFrame(test_metapth)
        self.control_parameters = ContrastiveControlParameters.from_json(
            load_json_from_file(control_parameter_pth)
        )
        print("Dataset Loaded")
        print(
            f'''
            Data Set Contains information from the Following Rules : 
            {
                ','.join([r.rule_name for r in self.control_parameters.rules])
            }
            '''
        )
        # $ DONT load the initial indices from the h5 dataset. 
        # $ Instead _remake the indices. 
        self.remake_indices(size)
    
    def remake_indices(self,size:int):
        """remake_indices 
        This method will rerun the rules on the self.dataset_meta
        """
        # $ Make a map of the indices of the ids in the dataset. 
        index_map = {k.decode('utf-8'): idx for idx,
                     k in enumerate(self.id_list)}
        # $ extract the indicies and  
        contrastive_df_indices , rule_distribution = self._make_indices(self.dataset_meta,size,self.control_parameters.rules)
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
        for size, rule in zip(size_dis, rules):
            created_indices = rule(dataframe, num_samples_per_rule=size)
            rule_distribution.extend([rule.rule_name for _ in range(len(created_indices))])
            all_indices.extend(created_indices)

        return all_indices , rule_distribution
    
    @staticmethod
    def load_sequences(seq):
        dd = {}
        for k in seq:
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
        btch_rule = self.indices_rules[idx] # str
        batch_id_i = self.id_list[idx_i] # str
        batch_id_j = self.id_list[idx_j] # str
        samp_i = self.get_channel_data(idx_i) # Dictionary
        samp_j = self.get_channel_data(idx_j) # Dictionary
        return samp_i,samp_j,batch_id_i,batch_id_j,btch_rule
    
    @staticmethod
    def collate_fn():
        return TestSetCollateFn()



def run_test_pipeline(model:LGRRoboRewardLearner,\
                    contrastive_set_generated_folder:str,\
                    batch_size = 20,\
                    size:int=200):
    dataset = RoboticsTestingDataset(
        contrastive_set_generated_folder,
        size=size
    )
    testing_loader = DataLoader(dataset,batch_size=batch_size,collate_fn=dataset.collate_fn())
    final_dataset_collection = []
    with torch.no_grad():
        for batch in testing_loader:
            contrastive_sample_generator,\
            rules_of_samples,\
            contrastive_id_list = batch

            contrastive_sample_generator.to_device(model.device)
            pp_channels,\
            pn_channels,\
            nn_channels,\
            np_channels = contrastive_sample_generator.create_contrastive_inputs('text')
            posp_reward = model.reward_predictor(
                model(pp_channels)
            )
            posn_reward = model.reward_predictor(
                model(nn_channels)
            )
            negp_reward = model.reward_predictor(
                model(pn_channels)
            )
            negn_reward = model.reward_predictor(
                model(np_channels)
            )
            datazipper_headers = [
                'pp_reward',
                'nn_reward',
                'pn_reward',
                'np_reward',
                'rule',
                'id_pair'
            ]
            datazipper_params = (
                posp_reward.cpu().numpy(),
                posn_reward.cpu().numpy(),
                negp_reward.cpu().numpy(),
                negn_reward.cpu().numpy(),
                rules_of_samples,
                contrastive_id_list
            )
            
            for datatuple in zip(datazipper_params):
                final_dataset_collection.append(
                    {h:t for h,t in zip(datazipper_headers,datatuple)}
                )
                
    return final_dataset_collection


