import torch
from torch.utils.data.dataloader import DataLoader
import pandas
import matplotlib.pyplot as plt
from ..dataloaders.mountaincar.dataset import BertEmbedContrastiveTokenizedDatasetWithSentences, contloader_collate_fn_with_mask_and_cats


CATEGORY_AUGMENT_SENTENCE_MAP = {
    1: "The car is swings around at the bottom of the valley.",
    2: "The car is able swing beyond the bottom of the valley but does not reach the top of the mountain",
    3: "The car is able to reach the top of the mountain",
}

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
