import abc
from typing import List, Tuple
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataset import TensorDataset
import pandas
import json
import matplotlib.pyplot as plt
import numpy
from ..dataloaders.mountaincar.dataset import BertEmbedContrastiveTokenizedDatasetWithSentences, contloader_collate_fn_with_mask_and_cats,CATEGORY_AUGMENT_SENTENCE_MAP,MAX_TEXT_LENGTH


TEST_EPISODE_SAMPLES = 10
TEST_SENTENCE_PER_EPISODE = 2


def create_text_category_reward_hist(category_group_data, text_cat, axis):
    msk = category_group_data['data_type'] == 'Sentence==Trajectory'
    x = category_group_data[msk]['model_score']
    y = category_group_data[~msk]['model_score']

    bins = numpy.linspace(min([x.min(), y.min()]),
                          max([x.max(), y.max()]), 100)

    axis.hist(x, bins, alpha=0.5, label='Trajectory is the same as the Sentence')
    axis.hist(y, bins, alpha=0.5,
              label='Trajectory is not the same as the Sentence')
    axis.legend(loc='upper right')
    axis.set_xlabel('Rewards')
    axis.set_ylabel('Num Sent/Traj')
    axis.set_title(
        f"Text Given to the model : {CATEGORY_AUGMENT_SENTENCE_MAP[text_cat]}")
    return axis


def create_traj_category_reward_hist(category_group_data, traj_cat, axis):
    msk = category_group_data['data_type'] == 'Sentence==Trajectory'
    x = category_group_data[msk]['model_score']
    y = category_group_data[~msk]['model_score']

    bins = numpy.linspace(min([x.min(), y.min()]),
                          max([x.max(), y.max()]), 100)

    axis.hist(x, bins, alpha=0.5, label='Sentence is Describing Trajectory')
    axis.hist(y, bins, alpha=0.5, label='Sentence is Not Describing Trajectory')
    axis.legend(loc='upper right')
    axis.set_xlabel('Rewards')
    axis.set_ylabel('Num Sent/Traj')
    axis.set_title(
        f"Trajectory was Depicting Behaviour : {traj_cat} : {CATEGORY_AUGMENT_SENTENCE_MAP[traj_cat]}")
    return axis


def create_grouped_traj_category_reward_hist(txt_grp_df, axis, text_grp):
    subgrp = txt_grp_df.groupby(['trajectory_category'])
    axis.legend(loc='upper right')
    axis.set_xlabel('Rewards')
    axis.set_ylabel('Num Sent/Traj')
    axis.set_title(
        f"Text Given to the model : {CATEGORY_AUGMENT_SENTENCE_MAP[text_grp]}")
    mx_val = txt_grp_df['model_score'].max()
    mn_val = txt_grp_df['model_score'].min()
    bins = numpy.linspace(mn_val, mx_val, 100)
    legends = []
    for grp_idx in subgrp.groups.keys():
        sub_cats = subgrp.get_group(grp_idx)
        traj_behavior = sub_cats['trajectory_behaviour'].iloc[0]
        axis.hist(sub_cats['model_score'], bins, alpha=0.5)
        legends.append(f"Trajectory's Behaviour : {traj_behavior}")
    axis.legend(legends)
    return axis


def create_grouped_text_category_reward_hist(traj_grp_df, axis, traj_grp):
    subgrp = traj_grp_df.groupby(['text_category'])
    axis.legend(loc='upper right')
    axis.set_xlabel('Rewards')
    axis.set_ylabel('Num Sent/Traj')
    axis.set_title(
        f"Trajectory Behavior Given to Model: {CATEGORY_AUGMENT_SENTENCE_MAP[traj_grp]}")
    mx_val = traj_grp_df['model_score'].max()
    mn_val = traj_grp_df['model_score'].min()
    bins = numpy.linspace(mn_val, mx_val, 100)
    legends = []
    for grp_idx in subgrp.groups.keys():
        sub_cats = subgrp.get_group(grp_idx)
        given_text = CATEGORY_AUGMENT_SENTENCE_MAP[grp_idx]
        axis.hist(sub_cats['model_score'], bins, alpha=0.5)
        legends.append(f"Text Is Describing : {given_text}")
    axis.legend(legends)
    return axis


class IndividualObjTestcase(metaclass=abc.ABCMeta):
    """IndividualObjTestcase : ONLY FOR MOUNTAIN CAR!
    - For each test case define a `configuration` and a `_make_data`. Do what ever you want in that. 
    - Use the configuration during the `_make_data` step
        - This should return tuple (`T`) of a `TensorDataset` and a `pandas.DataFrame` representing metadata of the `TensorDataset`. 
    
    - `run_test` will run the test case. Add what ever sugar needed for return data type. 
        - Invoke `_run_test` with the data to get the metadata df according to rewards. 
    """
    def __init__(self,tokenizer=None,batch_size=10,sample_size=2,max_traj_length = 200,action_space=3) -> None:
        assert tokenizer is not None
        self.tokenizer = tokenizer
        self.configuration = None
        self.batch_size=batch_size
        self.max_traj_length = max_traj_length
        self.action_space = action_space
        self.sample_size=sample_size

    def to_json(self):
        return dict(
            configuration = self.configuration,
            sample_size=self.sample_size
        )
    
    @property
    def name(self):
        return self.__class__.__name__
    
    def make_data(self,data_df):
        assert 'category' in data_df.columns and 'trajectory_stats' in data_df.columns and 'sentences' in data_df.columns
        return self._make_data(data_df)

    def _make_data(self,dataframe:pandas.DataFrame) -> Tuple[TensorDataset,pandas.DataFrame]:
        raise NotImplementedError()

    def _run_test(self,rw_model,test_df:pandas.DataFrame):
        tensor_dataset, metadata_df = self.make_data(test_df)
        loader = DataLoader(tensor_dataset,batch_size=self.batch_size,shuffle=False)
        rewards = []
        with torch.no_grad():
            for datatup in loader:
                sent, sentence_mask,st,st_mask,at,at_mask  = datatup
                reward_vals = rw_model.reward_fn(
                    st, at, sent, text_mask=sentence_mask, act_mask=at_mask, st_mask=st_mask)
                rewards.append(reward_vals)
            stacked_rewards = torch.cat(rewards,dim=0).squeeze(1).cpu().numpy()
        metadata_df['rewards'] = stacked_rewards
        return metadata_df
    
    def run_test(self,rw_model,test_df:pandas.DataFrame):
        raise NotImplementedError()

    
    @staticmethod
    def make_mask_from_len(len_tensor, max_size):
        '''
        len_tensor: 
        '''
        return (torch.arange(max_size)[None, :] < len_tensor[:, None]).float()

    def encode_sentence(self, sentence):
        data_dict = self.tokenizer.batch_encode_plus(
            [sentence],                      # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            # Pad & truncate all sentences.
            max_length=MAX_TEXT_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',     # Return pytorch tensors.
        )
        return (data_dict['input_ids'].squeeze(0), data_dict['attention_mask'].squeeze(0))

    def extract_trajectory(self, traj_obj):
        act_list = list(
            map(lambda x: x['action'], traj_obj))
        state_list = list(
            map(lambda x: x['observation'], traj_obj))

        act_list_len = len(act_list)
        state_list_len = len(state_list)

        if self.max_traj_length > act_list_len:
            # Adding A NULL Action token which is a part of the transformer to ensure null action
            act_list.extend([self.action_space for _ in range(
                self.max_traj_length-len(act_list))])

        if self.max_traj_length > state_list_len:
            # Repeat last state in the state list
            state_list.extend([state_list[-1]
                               for _ in range(self.max_traj_length-len(state_list))])
        action_tensor = torch.Tensor(act_list).type(torch.LongTensor)
        state_tensor = torch.Tensor(state_list)
        return (state_tensor,\
                self.make_mask_from_len(torch.Tensor(\
                    [state_list_len]), self.max_traj_length).squeeze(0),\
                action_tensor,\
                self.make_mask_from_len(torch.tensor(\
                    [act_list_len]), self.max_traj_length).squeeze(0)\
                )
    
    def _create_tensor_dataset(self,data_dict:List[dict]):
        return_data_dict = {
            "sent" : [],
            "sent_mask" : [],
            "st" : [],
            "st_mask" : [],
            "at" : [],
            "at_mask" : [],
        }
        for data in data_dict:
            assert 'sentence' in data
            sent,sent_mask = self.encode_sentence(data['sentence'])
            trajectory =data['trajectory_stats'] #json.loads()
            st,st_mask,at,at_mask = self.extract_trajectory(trajectory)
            return_data_dict["sent"].append(sent)
            return_data_dict["sent_mask"].append(sent_mask)
            return_data_dict["st"].append(st)
            return_data_dict["st_mask"].append(st_mask)
            return_data_dict["at"].append(at)
            return_data_dict["at_mask"].append(at_mask)
        
        return TensorDataset(
            torch.stack(return_data_dict["sent"]),
            torch.stack(return_data_dict["sent_mask"]),
            torch.stack(return_data_dict["st"]),
            torch.stack(return_data_dict["st_mask"]),
            torch.stack(return_data_dict["at"]),
            torch.stack(return_data_dict["at_mask"]),
        )


class TrajectoryFixedDifferentSentenceTestCase(IndividualObjTestcase):
    """TrajectoryFixedSentenceTestCase [summary]
    Test case 1 : Rewards after fixing single one sample per traj type and varying sentences.
    """
    def __init__(self,**kwags) -> None:
        super().__init__(**kwags) # Sentence variations
        self.configuration = [
            [
                "the trolley keeps moving around at the foot of the mountain and is unsuccessful in reaching the top of the mountain",
                "the truck is successful in staying at the foot of the mountain because it keeps moving around in the valley",
                "the car is unsuccessful in reaching the top of the mountain because it keeps oscillating in the valley",
                "the car moves around at the bottom of the mountain",
            ],
            
            [
                "the car swings beyond the valley upto the middle of the mountain and is unsuccessful in reaching the top of the mountain",
                "the trolley climbs up and down the hill",
                "car swings beyond the valley upto the middle of the hill",
                "truck is unsuccessful in reaching the top of the hill because it climbs up and down the hill",
            ],
            [
                "trolley successfully reaches the top of the hill",
                "the truck swings really fast in the valley and reaches the top of the mountain",
                "car swings really fast in the valley and reaches the top of the hill",
                "car successfully reaches the top of the hill,",
            ],
        ]
        
    def _make_data(self, dataframe:pandas.DataFrame):
        # Rewards after fixing single one sample per traj type
        collected_trajectories = []
        for cat,cat_grp in dataframe.groupby('category'):
            dx = cat_grp.sample(self.sample_size)
            traj_objs = dx.to_dict(orient='records')
            collected_trajectories.append(traj_objs)

        meta_data_objects = []
        for idx,category_traj_list in enumerate(collected_trajectories):
            traj_cat = idx+1
            for sidx,cat_sentences in enumerate(self.configuration):
                sent_cat = sidx+1
                for cat_traj_obj in category_traj_list:
                    for sent in cat_sentences:
                        meta_data_objects.append(
                            dict(
                                trajectory_category = traj_cat,
                                text_category = sent_cat, 
                                episode_id = cat_traj_obj['episode_id'],
                                sentence = sent,
                                test_case_name = self.__class__.__name__,
                                trajectory_stats =  cat_traj_obj['trajectory_stats'],
                            )
                        )
        metadata_df = pandas.DataFrame(meta_data_objects)
        tensor_dataset = self._create_tensor_dataset(meta_data_objects)
        return tensor_dataset,metadata_df

    def run_test(self, rw_model, test_df: pandas.DataFrame):
        reward_dataframe = self._run_test(rw_model, test_df)
        return reward_dataframe
        
class TrajectoryFixedSynonymsTestCase(IndividualObjTestcase):
    """TrajectoryFixedSentenceTestCase [summary]
    Test case 1.1: Rewards of fixing single one sample per traj type with synonym replacement.
    - For each sentence block explicity only use same category trajectory
    """
    def __init__(self,**kwags) -> None:
        super().__init__(**kwags) # Sentence variations
        self.configuration = [
            [
                "the trolley keeps moving around at the foot of the mountain and is unsuccessful in reaching the top of the mountain",
                "the truck keeps moving around at the foot of the hill and is unsuccessful in reaching the top of the hill",
                "the trolley moves around at the bottom of the mountain",
                "the car moves around at the bottom of the mountain"
            ],
            
            [
                "the car swings beyond the valley upto the middle of the mountain and is unsuccessful in reaching the top of the mountain",
                "the truck swings beyond the valley upto the middle of the hill and is unsuccessful in reaching the top of the hill",
            ],
            [
                "the trolley reaches the top of the hill",
                "the truck reaches the top of the hill",
                "the car reaches the top of the hill",
            ],
        ]
        
    def _make_data(self, dataframe:pandas.DataFrame):
        # Rewards after fixing single one sample per traj type
        collected_trajectories = []
        for cat,cat_grp in dataframe.groupby('category'):
            dx = cat_grp.sample(self.sample_size)
            traj_objs = dx.to_dict(orient='records')
            collected_trajectories.append(traj_objs)

        meta_data_objects = []
        # Rewards of fixing single one sample per traj type with synonym replacements in sentences.
        for cat_idx,data_tup in enumerate(zip(collected_trajectories,self.configuration)):
            # For each sentence block explicity only use same category trajectory
            category_traj_list, cat_sentences = data_tup
            for cat_traj_obj in category_traj_list:
                for sent in cat_sentences:
                    meta_data_objects.append(
                        dict(
                            trajectory_category = cat_idx+1,
                            text_category = cat_idx+1, 
                            episode_id = cat_traj_obj['episode_id'],
                            sentence = sent,
                            test_case_name = self.__class__.__name__,
                            trajectory_stats =  cat_traj_obj['trajectory_stats'],
                        )
                    )
        metadata_df = pandas.DataFrame(meta_data_objects)
        tensor_dataset = self._create_tensor_dataset(meta_data_objects)
        return tensor_dataset,metadata_df

    def run_test(self, rw_model, test_df: pandas.DataFrame):
        reward_dataframe = self._run_test(rw_model, test_df)
        return reward_dataframe



class TrajectoryFixedSemanticSimilarTestCase(IndividualObjTestcase):
    """TrajectoryFixedSemanticSimilarTestCase [summary]
    Test case 1.2: Single trajectory and multiple sentences that are semantically the same,
        - For each sentence block explicity only use same category trajectory
    """
    def __init__(self,**kwags) -> None:
        super().__init__(**kwags) # Sentence variations
        self.configuration = [
            [
                "the truck keeps moving around at the foot of the hill and is unsuccessful in reaching the top of the hill",
                "the trolley moves around at the bottom of the mountain",
                
            ],
            [
                "car swings beyond the valley upto the middle of the hill",
                "the truck swings beyond the valley upto the middle of the hill and is unsuccessful in reaching the top of the hill",
            ],
            [
                "the car reaches the top of the hill",
                "the truck swings really fast in the valley and reaches the top of the mountain",
            ],
        ]
        
    def _make_data(self, dataframe:pandas.DataFrame):
        # Rewards after fixing single one sample per traj type
        collected_trajectories = []
        for cat,cat_grp in dataframe.groupby('category'):
            dx = cat_grp.sample(self.sample_size)
            traj_objs = dx.to_dict(orient='records')
            collected_trajectories.append(traj_objs)

        meta_data_objects = []
        # Rewards of fixing single one sample per traj type with synonym replacements in sentences.
        for cat_idx,data_tup in enumerate(zip(collected_trajectories,self.configuration)):
            # For each sentence block explicity only use same category trajectory
            category_traj_list, cat_sentences = data_tup
            for cat_traj_obj in category_traj_list:
                for sent in cat_sentences:
                    meta_data_objects.append(
                        dict(
                            trajectory_category = cat_idx+1,
                            text_category = cat_idx+1, 
                            episode_id = cat_traj_obj['episode_id'],
                            sentence = sent,
                            test_case_name = self.__class__.__name__,
                            trajectory_stats =  cat_traj_obj['trajectory_stats'],
                        )
                    )
        metadata_df = pandas.DataFrame(meta_data_objects)
        tensor_dataset = self._create_tensor_dataset(meta_data_objects)
        return tensor_dataset,metadata_df

    def run_test(self, rw_model, test_df: pandas.DataFrame):
        reward_dataframe = self._run_test(rw_model, test_df)
        return reward_dataframe

class TextRewardTestCase(IndividualObjTestcase):
    """TextRewardTestCase 
    Test case 2 : Rewards of fixing single one sample sentence but try multiple trajectory of each type.
    """
    def __init__(self,**kwags) -> None:
        super().__init__(**kwags) # Sentence variations
        self.configuration = [
            [
                "the trolley moves around at the bottom of the mountain",
            ],
            [
                "the truck swings beyond the valley upto the middle of the hill",
            ],
            [
                "the car reaches the top of the hill",
            ],
        ]
        
    def _make_data(self, dataframe:pandas.DataFrame):
        # Rewards after fixing single one sample per traj type
        collected_trajectories = []
        for cat,cat_grp in dataframe.groupby('category'):
            dx = cat_grp.sample(self.sample_size)
            traj_objs = dx.to_dict(orient='records')
            collected_trajectories.append(traj_objs)

        meta_data_objects = []
        for idx,category_traj_list in enumerate(collected_trajectories):
            traj_cat = idx+1
            for sidx,cat_sentences in enumerate(self.configuration):
                sent_cat = sidx+1
                for cat_traj_obj in category_traj_list:
                    for sent in cat_sentences:
                        meta_data_objects.append(
                            dict(
                                trajectory_category = traj_cat,
                                text_category = sent_cat, 
                                episode_id = cat_traj_obj['episode_id'],
                                sentence = sent,
                                test_case_name = self.__class__.__name__,
                                trajectory_stats =  cat_traj_obj['trajectory_stats'],
                            )
                        )
        metadata_df = pandas.DataFrame(meta_data_objects)
        tensor_dataset = self._create_tensor_dataset(meta_data_objects)
        return tensor_dataset,metadata_df

    def run_test(self, rw_model, test_df: pandas.DataFrame):
        reward_dataframe = self._run_test(rw_model, test_df)
        return reward_dataframe


def single_object_reference_results(rw_model,test_data_frame,tokenizer=None):
    test_objects = [
        TrajectoryFixedDifferentSentenceTestCase(sample_size=2,tokenizer=tokenizer),\
        TrajectoryFixedSynonymsTestCase(sample_size=1,tokenizer=tokenizer),\
        TrajectoryFixedSemanticSimilarTestCase(sample_size=1,tokenizer=tokenizer),\
        TextRewardTestCase(sample_size=4,tokenizer=tokenizer),\
    ]
    return_data = []
    for proc_obj in test_objects:
        result_df = proc_obj.run_test(rw_model,test_data_frame)
        result_object = dict(
            test_case_name = proc_obj.name,
            meta_data = proc_obj.to_json(),
            results = result_df.to_dict(orient='records')
        )
        return_data.append(result_object)
    return return_data
        

def mountain_car_based_test_results(rw_model, test_data_frame, tokenizer=None, only_core_samples=True):
    cont_loader = DataLoader(BertEmbedContrastiveTokenizedDatasetWithSentences(test_data_frame,
                                                                               tokenizer,
                                                                               num_episode_samples=TEST_EPISODE_SAMPLES,
                                                                               num_sentences_per_episode=TEST_SENTENCE_PER_EPISODE,
                                                                               with_mask=True,
                                                                               only_core_samples=only_core_samples, with_cats=True),
                             batch_size=400, shuffle=True, collate_fn=contloader_collate_fn_with_mask_and_cats)
    # cont_op = iter().next()
    for step, cont_op in enumerate(cont_loader):
        pos_sent, pos_sentence_mask, pos_traj, pos_sentence, neg_sent, neg_sentence_mask, neg_traj, neg_sentence, pos_cat, neg_cat = cont_op
        # print(pos_cat)
        # print(neg_cat)
        with torch.no_grad():
            pos_st = pos_traj[0]
            pos_at = pos_traj[2]
            neg_st = neg_traj[0]
            neg_at = neg_traj[2]

            pp_tensor = rw_model.reward_fn(
                pos_st, pos_at, pos_sent, text_mask=pos_sentence_mask, act_mask=pos_traj[3], st_mask=pos_traj[1])
            nn_tensor = rw_model.reward_fn(
                neg_st, neg_at, neg_sent, text_mask=neg_sentence_mask, act_mask=neg_traj[3], st_mask=neg_traj[1])
            np_tensor = rw_model.reward_fn(
                neg_st, neg_at, pos_sent, text_mask=pos_sentence_mask, act_mask=neg_traj[3], st_mask=neg_traj[1])
            pn_tensor = rw_model.reward_fn(
                pos_st, pos_at, neg_sent, text_mask=neg_sentence_mask, act_mask=pos_traj[3], st_mask=pos_traj[1])
            final_list = []
            final_list += list(zip(pos_sentence, pos_sentence, pos_cat, pos_cat, pp_tensor.cpu(), [
                               'Sentence==Trajectory' for _ in range(len(pos_sentence))]))
            final_list += list(zip(neg_sentence, neg_sentence, neg_cat, neg_cat, nn_tensor.cpu(), [
                               'Sentence==Trajectory' for _ in range(len(pos_sentence))]))

            final_list += list(zip(neg_sentence, pos_sentence, neg_cat, pos_cat, np_tensor.cpu(), [
                               'Sentence!=Trajectory' for _ in range(len(pos_sentence))]))
            final_list += list(zip(pos_sentence, neg_sentence, pos_cat, neg_cat, pn_tensor.cpu(), [
                               'Sentence!=Trajectory' for _ in range(len(pos_sentence))]))
            data = []
            for s, n, traj_c, text_cat, sc, dt in final_list:
                data.append(dict(
                    trajectory_behaviour=CATEGORY_AUGMENT_SENTENCE_MAP[int(
                        traj_c.item())],
                    model_input_sentence=n,
                    model_score=sc.item(),
                    data_type=dt,
                    trajectory_category=traj_c.item(),
                    text_category=text_cat.item(),
                ))
        return pandas.DataFrame(data)


def mountain_car_base_test_case(rw_model, testing_df, only_core_samples=True, tokenizer=None):
    contrastive_scored_examples = mountain_car_based_test_results(
        rw_model, testing_df, only_core_samples=only_core_samples, tokenizer=tokenizer)
    core_sent_df = contrastive_scored_examples
    agg_df = core_sent_df[['trajectory_category', 'text_category', 'model_score']].groupby(
        ['trajectory_category', 'text_category']).agg(mean_Reward=('model_score', 'mean'), variance_Reward=('model_score', 'var'))
    count_df = core_sent_df[['trajectory_category', 'text_category', 'model_score']].groupby(
        ['trajectory_category', 'text_category']).count()
    count_df = count_df.rename(columns={'model_score': 'data_count'})
    agg_df = agg_df.join(count_df, on=['trajectory_category', 'text_category'])

    return agg_df, core_sent_df


class MountainCarTestCase:
    def __init__(self,
                 model,
                 tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def run(self, testing_dataframe, only_core_samples=False):
        return mountain_car_base_test_case(self.model,
                                           testing_dataframe,
                                           tokenizer=self.tokenizer,
                                           only_core_samples=only_core_samples)

    @staticmethod
    def test_case_2(data_df, figsize=(20, 20)):
        num_cats = len(CATEGORY_AUGMENT_SENTENCE_MAP.keys())
        agg_df = data_df.groupby(['text_category'])
        fig, axes = plt.subplots(nrows=num_cats, ncols=1, figsize=figsize)
        fig.suptitle('Reward Distribution based on Different Text Input')
        for text_cat, ax_row in enumerate(axes):
            ax = ax_row
            category_group = agg_df.get_group((text_cat+1))
            ax = create_text_category_reward_hist(
                category_group, text_cat+1, ax)
        return fig

    @staticmethod
    def test_case_3(data_df, figsize=(20, 20)):
        num_cats = len(CATEGORY_AUGMENT_SENTENCE_MAP.keys())
        agg_df = data_df.groupby(['trajectory_category'])
        fig, axes = plt.subplots(nrows=num_cats, ncols=1, figsize=(20, 20))
        fig.suptitle(
            'Reward Distribution based on Different Trajectory behaviour')
        for traj_cat, ax_row in enumerate(axes):
            ax = ax_row
            category_group = agg_df.get_group((traj_cat+1))
            ax = create_traj_category_reward_hist(
                category_group, traj_cat+1, ax)
        return fig

    @staticmethod
    def test_case_4(data_df, figsize=(20, 20)):
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 20))
        agg_df = data_df.groupby(['text_category'])
        fig.suptitle(
            'Reward Distribution Grouped based on Different Text Input')
        for txt_grp, ax in enumerate(axes):
            txt_grp_df = agg_df.get_group(txt_grp+1)
            ax = create_grouped_traj_category_reward_hist(
                txt_grp_df, ax, txt_grp+1)
        return fig

    @staticmethod
    def test_case_5(data_df, figsize=(20, 20)):
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 20))
        fig.suptitle(
            'Reward Distribution Grouped based on Different Trajectory behaviour')
        agg_df = data_df.groupby(['trajectory_category'])
        for txt_grp, ax in enumerate(axes):
            txt_grp_df = agg_df.get_group(txt_grp+1)
            ax = create_grouped_text_category_reward_hist(
                txt_grp_df, ax, txt_grp+1)
        return fig

    def print_figures(self, testdataframe, only_core_samples=False, figsize=(20, 20)):
        testing_reward_summary, core_sent_df = self.run(
            testdataframe, only_core_samples=only_core_samples)
        fig_1 = self.test_case_2(core_sent_df, figsize=figsize)
        fig_2 = self.test_case_3(core_sent_df, figsize=figsize)
        fig_3 = self.test_case_4(core_sent_df, figsize=figsize)
        fig_4 = self.test_case_5(core_sent_df, figsize=figsize)
        return (fig_1, fig_2, fig_3, fig_4)

    def save_tests(self, testdataframe, logger, only_core_samples=False, figsize=(20, 20)):
        from neptunecontrib.api import log_table
        # get core data.
        testing_reward_summary, core_sent_df = self.run(
            testdataframe, only_core_samples=only_core_samples)
        sample_add_str = ''
        if only_core_samples:
            sample_add_str = '_only_core_samples'

        fig_1 = self.test_case_2(core_sent_df, figsize=figsize)
        tc_1 = f'reward_distribution_by_text{sample_add_str}.png'
        fig_1.savefig(tc_1)

        fig_2 = self.test_case_3(core_sent_df, figsize=figsize)
        tc_2 = f'reward_distribution_by_trajectory{sample_add_str}.png'
        fig_2.savefig(tc_2)

        fig_3 = self.test_case_4(core_sent_df, figsize=figsize)
        tc_3 = f'reward_grouped_distribution_by_text{sample_add_str}.png'
        fig_3.savefig(tc_3)

        fig_4 = self.test_case_5(core_sent_df, figsize=figsize)
        tc_4 = f'reward_grouped_distribution_by_trajectory{sample_add_str}.png'
        fig_4.savefig(tc_4)

        category_reward_distribution = f'category_reward_distribution{sample_add_str}'
        logger.experiment.log_artifact(tc_1)
        logger.experiment.log_artifact(tc_2)
        logger.experiment.log_artifact(tc_3)
        logger.experiment.log_artifact(tc_4)
        log_table(category_reward_distribution,
                  testing_reward_summary, experiment=logger.experiment)


def save_tests(model, testing_df, logger, only_core_samples=False):
    from neptunecontrib.api import log_chart
    from neptunecontrib.api import log_table
    sample_add_str = ''
    if only_core_samples:
        sample_add_str = '_only_core_samples'
    testing_result_reward_summary, core_sent_df = mountain_car_based_test_results(
        model, testing_df, only_core_samples=only_core_samples)

    fig_1 = MountainCarTestCase.test_case_2(core_sent_df)
    tc_1 = f'reward_distribution_by_text{sample_add_str}.png'
    fig_1.savefig(tc_1)

    fig_2 = MountainCarTestCase.test_case_3(core_sent_df)
    tc_2 = f'reward_distribution_by_trajectory{sample_add_str}.png'
    fig_2.savefig(tc_2)

    fig_3 = MountainCarTestCase.test_case_4(core_sent_df)
    tc_3 = f'reward_grouped_distribution_by_text{sample_add_str}.png'
    fig_3.savefig(tc_3)

    fig_4 = MountainCarTestCase.test_case_5(core_sent_df)
    tc_4 = f'reward_grouped_distribution_by_trajectory{sample_add_str}.png'
    fig_4.savefig(tc_4)

    category_reward_distribution = f'category_reward_distribution{sample_add_str}'
    logger.experiment.log_artifact(tc_1)
    logger.experiment.log_artifact(tc_2)
    logger.experiment.log_artifact(tc_3)
    logger.experiment.log_artifact(tc_4)
    log_table(category_reward_distribution,
              testing_result_reward_summary, experiment=logger.experiment)
