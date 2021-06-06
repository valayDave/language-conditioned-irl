import pandas
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import torch
import os
import h5py
from typing import List
import matplotlib as plt
import json


from ..channel import ChannelData
from .utils import \
    load_json_from_file,\
    save_json_to_file

from .dataset import \
    GROUPNAMES,\
    DYNAMIC_CHANNEL_MAP,\
    ContrastiveCollateFn,\
    MultiTaskContrastiveCollateFn,\
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
        self.use_channels = use_channels
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
        
    
    def remake_indices(self,size:int,rules=[],rule_distribution=[]):
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
        if len(rule_distribution) > 0:
            assert len(rule_distribution) == len(input_rules)
            rule_distribution = [i/sum(rule_distribution) for i in rule_distribution]

        contrastive_df_indices , rule_distribution = self._make_indices(self.dataset_meta,\
                                                                        size,\
                                                                        input_rules,\
                                                                        size_distribution=rule_distribution)
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
    def _make_indices(dataframe:pandas.DataFrame,size:int,rules:List[SampleContrastingRule],size_distribution=[]):
        """_make_indices [summary]
        Takes a dataframe , runs the list of `SampleContrastingRule`'s to create contrastive indices of a total certain `size` or based on `size_distribution`
        returns : Tuple(all_indices,rule_distribution)
            - `all_indices` : List[List[int,int]]
            - `rule_distribution` : List[str] : list where each index corresponds to what rule created that contrastive sample. . 
        """
        size_dis = []
        if len(size_distribution) > 0:
            size_dis = [int(i*size) for i in size_distribution]
        else:
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
    def _make_list_chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    
    @staticmethod
    def load_sequences(seq,channels,mask=False):
        dd = {}
        for k in channels:
            if not mask:
                assert k in seq.keys() or k in DYNAMIC_CHANNEL_MAP, f"Channel {k} not found in the HDF5 Dataset Sequence Or Not Found in {DYNAMIC_CHANNEL_MAP.keys()}"
            if k in DYNAMIC_CHANNEL_MAP: # Monkey Patch
                for key in DYNAMIC_CHANNEL_MAP[k]:
                    dd[key] = np.array(seq[key])
                    
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
        for k in self.use_channels:
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


class JointsChannelsConcatDataset(SentenceContrastiveDataset):
    """JointsChannelsConcatDataset 
    Experiment to concatenate everything to one dimension. 
    """
    def __init__(self, 
                contrastive_set_generated_folder:str,\
                use_channels=USE_CHANNELS,\
                train=True,\
                normalize_images=False,\
                use_original_contrastive_indices:bool=True,\
                size:int=200) -> None:
        super().__init__(contrastive_set_generated_folder, use_channels=use_channels, train=train, normalize_images=normalize_images, use_original_contrastive_indices=use_original_contrastive_indices, size=size)
        self.joint_channel_name = 'joint_combined_vector'
        self.target_pos_vec = 'final_target_coordinates'
        self.max_position_set = None
        self._create_concact_joint_channels()
        self._create_final_target_coordinates()

    def _create_final_target_coordinates(self):
        if not self.target_pos_vec in self.use_channels:
            return
        assert 'tcp_position' in self.sequences
        dfx = self.dataset_meta['object_postions_and_pose'].apply(lambda x:len(np.array(json.loads(x)))/3)
        max_obj_vals = int(dfx.max())
        id_fixed_positions = []        
        final_masks = []
        for ixx in self.id_list:
            ixp = ixx.decode('utf-8')
            # Extract values from the dataset
            r = self.dataset_meta[self.dataset_meta['demo_name'] == ixp].iloc[0]
            positions = json.loads(r['object_postions_and_pose'])
            objects = json.loads(r['object_meta'])[2:]
            main_id = r['target_id']
            # make chunks and correlate and make new array with first element as the target object
            chunked_positions = list(self._make_list_chunks(positions,3))
            new_adjusted_pos_arr = []
            for obid,pos_chunk in zip(objects,chunked_positions):
                x,y,_ = pos_chunk
                if obid == main_id:
                    tt = [x,y]
                    tt.extend(new_adjusted_pos_arr)
                    new_adjusted_pos_arr = tt
                else:
                    new_adjusted_pos_arr.extend([x,y])
            
            ob_mask_arr = [1 for _ in range(len(new_adjusted_pos_arr))]
            padding_amt = max_obj_vals*2 - len(new_adjusted_pos_arr)
            padding = [0 for _ in range(padding_amt)]
            new_adjusted_pos_arr.extend(padding)
            ob_mask_arr.extend(padding)
            
            final_masks.append(np.array(ob_mask_arr,dtype=np.int32))
            id_fixed_positions.append(np.array(new_adjusted_pos_arr,dtype=np.float32))

        correlated_final_position_list = np.stack(id_fixed_positions)
        position_masks = np.stack(final_masks)
        self.max_position_set =  position_masks.shape[1]
        self.sequences[self.target_pos_vec] = np.expand_dims(correlated_final_position_list,2)
        self.masks[self.target_pos_vec] = position_masks
        


    def _create_concact_joint_channels(self):
        if not self.joint_channel_name in self.use_channels:
            return
        assert 'joint_gripper' in self.sequences and 'joint_robot_position' in self.sequences
        save_vectors =[]
        for i in range(len(self.sequences['joint_gripper'])):
            gp = self.sequences['joint_gripper'][i]
            jp = self.sequences['joint_robot_position'][i]
            gripposv = np.expand_dims(gp,1)
            conc_vec = np.array(np.concatenate((gripposv,jp),axis=1),dtype=np.float32)
            
            save_vectors.append(conc_vec)
        
        self.sequences[self.joint_channel_name] = save_vectors
        self.masks[self.joint_channel_name] = self.masks['joint_gripper']
        

class TaskBasedSentenceContrastiveDataset(Dataset):
    """TaskBasedSentenceContrastiveDataset 
    Creates a dataset which returns rule specific contrastive loss tuples. 
    
    Parameters
    ----------
    - `rules` : 
        - Array of rules to apply. 
    """
    def __init__(self,\
                contrastive_set_generated_folder:str,\
                use_channels=USE_CHANNELS,\
                train=True,\
                rules = [],\
                normalize_images=False,\
                size:int=200) -> None:
        assert len(rules) > 0, "Need Specific Rules To Instantiate Dataset"
        self.datasets = [
            JointsChannelsConcatDataset(
                contrastive_set_generated_folder,
                use_channels=use_channels,
                use_original_contrastive_indices=True,
                normalize_images=normalize_images,
                train=train
            ) for _ in range(len(rules))
        ]
        
        for r,d in zip(rules,self.datasets):
            rule_size = int(size/len(rules))
            d.remake_indices(rule_size,rules=[r])
        
    def __getitem__(self, index):
        return tuple(
            d[index] for d in self.datasets
        )
        
    def __len__(self):
        return min([len(x) for x in self.datasets])


    @staticmethod
    def collate_fn():
        return MultiTaskContrastiveCollateFn()
        
