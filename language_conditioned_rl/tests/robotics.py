import pandas
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import torch
import os
import h5py
from typing import List
import matplotlib.pyplot as plt

from ..models.robotics.reward_model import LGRRoboRewardLearner
from ..dataloaders.channel import ChannelData,ChannelHolder,ContrastiveGenerator
from ..dataloaders.robotics.utils import \
    load_json_from_file,\
    save_json_to_file

from ..dataloaders.robotics.dataset import \
    GROUPNAMES,\
    ContrastiveCollateFn,\
    is_present,\
    collate_indices,\
    map_to_contrastive_indexes_to_ids,\
    USE_CHANNELS

from ..dataloaders.robotics.datamaker import \
    HDF5ContrastiveSetCreator,\
    ContrastiveControlParameters,\
    SampleContrastingRule

from ..dataloaders.robotics.contrastive_dataset import SentenceContrastiveDataset    

class RoboticsTestingDataset:
    pass

RULE_MAP = {
    # This helps with plotting charts by mapping pandas columns to lengend names and titles. 
    "ContrastingActionsRule":{
      "pos_traj_rw":"The sentence is describing task X and the trajectory followed task X",
      "neg_traj_rw":"The sentence is describing task Y and the trajectory followed task X",
      "plot_title":"Reward Distribution when the tasks are different "
    },
    "ContrastingObjectRule":{
      "pos_traj_rw":"The sentence is describing object X in environment and the trajectory interacted with object X",
      "neg_traj_rw":"The sentence is describing object Y in environment and the trajectory interacted with object X",
      "plot_title":"Reward Distribution when the task is the same but the object is changed in the contrastive sentence"
    },
    "PouringIntensityRule":{
      "pos_traj_rw":"The sentence is telling the robot to pour little and the robot did the same. Same for pouring a lot.",
      "neg_traj_rw":"The sentence is telling the robot to pour little but the robot pours a lot. Visa-Versa for little",
      "plot_title":"Reward Distribution in the pour Little vs lot case"
    },
}
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


class RoboTestDataset(SentenceContrastiveDataset):
    def __init__(self, contrastive_set_generated_folder: str,normalize_images:bool=False,use_channels=USE_CHANNELS, size: int=200) -> None:
        super().__init__(contrastive_set_generated_folder, \
                        normalize_images=normalize_images,\
                        use_channels=use_channels,\
                        train=False,\
                        use_original_contrastive_indices=False,\
                        size=size,)

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
                    batch_size = 20,\
                    dataset = None):
    assert dataset is not None
    assert type(dataset) == RoboTestDataset
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
            
            posp_reward_features,_ = model(pp_channels)
            posp_reward = model.reward_predictor(
                posp_reward_features
            )
            posn_reward_features,_ = model(nn_channels)
            posn_reward = model.reward_predictor(
                posn_reward_features
            )
            negp_reward_features,_ = model(pn_channels)
            negp_reward = model.reward_predictor(
                negp_reward_features
            )
            negn_reward_features,_ = model(np_channels)
            negn_reward = model.reward_predictor(
                negn_reward_features
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
                posp_reward.cpu().squeeze(0).numpy(),
                posn_reward.cpu().squeeze(0).numpy(),
                negp_reward.cpu().squeeze(0).numpy(),
                negn_reward.cpu().squeeze(0).numpy(),
                rules_of_samples,
                contrastive_id_list
            )
            
            for datatuple in zip(*datazipper_params):
              data_dict = {}
              for h,t in zip(datazipper_headers,datatuple):
                if type(t) == np.ndarray:
                  data_dict[h] = float(t[0])
                else:
                  data_dict[h]=t
              final_dataset_collection.append(data_dict)

    return final_dataset_collection



def plot_test_case_results(test_pipeline_resp,plt_name='robo-reward-dist.pdf',show=False,rule_map=RULE_MAP):
    model_responses = pandas.DataFrame(test_pipeline_resp)
    dfx = model_responses[['pp_reward','pn_reward','rule','id_pair']]
    dfx = dfx.rename(columns={'pp_reward':'pos_traj_rw','pn_reward':'neg_traj_rw'})
    dfy =model_responses[['nn_reward', 'np_reward','rule','id_pair']]
    dfy = dfx.rename(columns={'nn_reward':'pos_traj_rw','np_reward':'neg_traj_rw'})
    final_df = pandas.concat((dfx,dfy))
    fig, axes = plt.subplots(nrows=len(final_df.groupby('rule')), ncols=1, figsize=(30,30))
    for axis,g in zip(axes,final_df.groupby('rule')):
        grp_v,grp = g
        min_t = grp['neg_traj_rw'].min()
        max_t = grp['pos_traj_rw'].max()
        bins = np.linspace(min_t,max_t, 10)
        axis.hist(grp['pos_traj_rw'], bins, alpha=0.5, label=rule_map[grp_v]['pos_traj_rw'])
        axis.hist(grp['neg_traj_rw'], bins, alpha=0.5, label=rule_map[grp_v]['neg_traj_rw'])
        axis.legend(loc='upper left',prop={'size': 18})
        axis.set_xlabel('Rewards')
        axis.set_ylabel('Num Sent/Traj')
        axis.set_title(rule_map[grp_v]['plot_title'])
        for item in ([axis.title, axis.xaxis.label, axis.yaxis.label] +
                    axis.get_xticklabels() + axis.get_yticklabels()):
            item.set_fontsize(20)
    if show:
        fig.show()
    fig.savefig(plt_name)


def save_test_data(
        model:LGRRoboRewardLearner,\
        logger,
        batch_size = 20,\
        dataset = None,\
        plt_name='robo-reward-dist.pdf',\
        show_plot=False,
        use_channels=USE_CHANNELS,
    ):
    return_object = run_test_pipeline(
        model,
        dataset = dataset,
        batch_size=batch_size,
        use_channels=use_channels
    )
    save_tests = f'{logger.experiment_id}.json'
    save_json_to_file(return_object,save_tests)
    logger.experiment.log_artifact(save_tests)
    plot_test_case_results(return_object,show=show_plot,plt_name=plt_name)
    logger.experiment.log_artifact(plt_name)
    # return save_tests