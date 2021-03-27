from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import itertools
import torch
import random
import os 
import json
ACTION_SPACE = 'discrete'
SAMPLE = 1000
MAX_TEXT_LENGTH = 25
EPISODE_SAMPLES = 90
SENTENCE_PER_EPISODE = 10
SENTENCE_SAMPLE_BUFFER_SIZE = 15
CATEGORY_AUGMENT_SENTENCE_MAP = {
    1: "The car swings around at the bottom of the valley.",
    2: "The car is able swing beyond the bottom of the valley but does not reach the top of the mountain",
    3: "The car is able to reach the top of the mountain",
}


def safe_mkdir(pth):
    try:
        os.makedirs(pth)
    except:
        return

def load_json_from_file(file_path):
    with open(file_path,'r') as f:
        json_file = json.load(f)
    return json_file

def save_json_to_file(json_dict,file_path):
    with open(file_path,'w') as f:
        json.dump(json_dict,f)


def sample_category_sentences_from_dict(sent_cat_dict: dict, cat_sent_sample_size: int):
    """sample_category_sentences [summary]

    :param sent_cat_dict: { 1 : {"blaah blah blah ": 1,"blaah blah blah2 ": 1 ,"blaah blah blah3": 1}, 2:{"blaah blah blah111 ": 1}}
    :type sent_cat_dict: dict with category as key and value as object with sentences as keys. 
    :param cat_sent_sample_size: number of samples to extract
    """
    sentence_sampling_dict = {}
    for cat in sent_cat_dict:
        # If number of sentences larger than sample size than sample the sentences to create buffer
        if len(sent_cat_dict[cat].keys()) > cat_sent_sample_size:
            sentence_sampling_dict[cat] = random.sample(
                list(sent_cat_dict[cat].keys()), cat_sent_sample_size)
        else:  # else fill the buffer with all sentences.
            sentence_sampling_dict[cat] = list(sent_cat_dict[cat].keys())
    return sentence_sampling_dict


def make_cat_sent_dict(part_eps, num_cats):
    sent_cat_dict = {p: {} for p in range(1, num_cats+1)}
    for p in part_eps:
        for sents in p['sentences']:
            cat = int(p['category'].iloc[0])
            for s in sents:
                sent_cat_dict[cat][s] = 1
    return sent_cat_dict


def set_sampled_sentences_to_epsiodes(part_wise_json_arr, sent_cat_dict, num_samples,with_grounding_examples=False):
    ret_arr = []
    for peps in part_wise_json_arr:
        cat_arr = []
        for idx, row in enumerate(peps):
            if not with_grounding_examples:
                sents = random.sample(sent_cat_dict[row['category']], num_samples)
            else:
                sents = random.sample(sent_cat_dict[row['category']], num_samples-1) + [CATEGORY_AUGMENT_SENTENCE_MAP[row['category']]]
            ret_ob = {**row}
            ret_ob['sentences'] = sents
            cat_arr.append(ret_ob)
        ret_arr.append(cat_arr)
    return ret_arr


def make_benchmark_dataset(all_data_df,
                           BENCHMARK_MC_CONTENT_FRAC=0.20,
                           BENCHMARK_TRAIN_EPISODE_SAMPLES=80,  # per category,
                           BENCHMARK_TEST_EPISODE_SAMPLES=10,  # per category,
                           BENCHMARK_SENTENCE_PER_EPISODE=3,
                           BENCHMARK_SENTENCE_SAMPLE_BUFFER_SIZE=100,
                           WITH_GROUNDING_EXAMPLES=False,
                           save_dir='./datasets'):
    assert 'category' in all_data_df.columns and 'trajectory_stats' in all_data_df.columns and 'sentences' in all_data_df.columns
    test_set_df = all_data_df.sample(frac=BENCHMARK_MC_CONTENT_FRAC)
    train_set_df = all_data_df[~all_data_df.index.isin(test_set_df.index)]
    train_part_eps = [train_set_df[train_set_df['category'] == p]
                      for p in range(1, 4)]  # Category based splits
    test_part_eps = [test_set_df[test_set_df['category'] == p]
                     for p in range(1, 4)]  # Category based splits
    # Sample episodes if necessary
    # if episode_samples:
    train_part_eps = [e.sample(BENCHMARK_TRAIN_EPISODE_SAMPLES)
                      for e in train_part_eps]
    test_part_eps = [e.sample(BENCHMARK_TEST_EPISODE_SAMPLES)
                     for e in test_part_eps]

    train_cat_sent_dict = make_cat_sent_dict(train_part_eps, 3)
    test_cat_sent_dict = make_cat_sent_dict(test_part_eps, 3)
    train_sent_dict = sample_category_sentences_from_dict(
        train_cat_sent_dict, BENCHMARK_SENTENCE_SAMPLE_BUFFER_SIZE)
    test_sent_dict = sample_category_sentences_from_dict(
        test_cat_sent_dict, BENCHMARK_SENTENCE_SAMPLE_BUFFER_SIZE)

    train_part_eps = [p.to_dict(orient='records') for p in train_part_eps]
    test_part_eps = [p.to_dict(orient='records') for p in test_part_eps]

    train_part_eps = set_sampled_sentences_to_epsiodes(
        train_part_eps, train_sent_dict, BENCHMARK_SENTENCE_PER_EPISODE,with_grounding_examples=WITH_GROUNDING_EXAMPLES)
    test_part_eps = set_sampled_sentences_to_epsiodes(
        test_part_eps, test_sent_dict, BENCHMARK_SENTENCE_PER_EPISODE,with_grounding_examples=False)
    
    test_part_eps = [e for z in test_part_eps for e in z]
    train_part_eps = [e for z in train_part_eps for e in z]

    safe_mkdir(save_dir)
    train_save_path = os.path.join(save_dir,'train.json')
    test_save_path = os.path.join(save_dir,'test.json')
    

    save_json_to_file(train_part_eps,train_save_path)
    save_json_to_file(test_part_eps,test_save_path)

    # return train_part_eps, test_part_eps, train_sent_dict, test_sent_dict


def make_samples(data_combinations, sentence_sampling_dict, only_core_samples=True, sentence_sample=3, load_saved=False, no_core_samples=False):
    contrastive_tuples = []
    # Make combinations of the samples by category: Like blocks of cat1 and cat2
    for comb_tup in data_combinations:
        # For each combinations of the "block of cateogry samples" create combinations of individual trajectories.
        cat_x_docs, cat_y_docs = comb_tup
        # Get categories so that we can sample sentences in a healthy way
        cat_x, cat_y = cat_x_docs[0]['category'], cat_y_docs[0]['category']

        cat_x_sents = sentence_sampling_dict[cat_x]
        cat_y_sents = sentence_sampling_dict[cat_y]

        for d_x, d_y in itertools.product(cat_x_docs, cat_y_docs):
            d_x_sent = None
            d_y_sent = None
            # For each combinations of individual trajectories samples the sentences for each the
            # contrastive data tuples: (pos_sent,pos_traj,pos_cat,neg_sent,neg_traj,neg_cat)
            if only_core_samples:  # Create only core samples
                d_x_sent = [CATEGORY_AUGMENT_SENTENCE_MAP[cat_x]]
                d_y_sent = [CATEGORY_AUGMENT_SENTENCE_MAP[cat_y]]
            elif load_saved:
                d_x_sent = d_x['sentences']
                d_y_sent = d_y['sentences']
            elif no_core_samples:
                d_x_sent = random.sample(cat_x_sents, sentence_sample)
                d_y_sent = random.sample(cat_y_sents, sentence_sample)
            else:  # create something using both
                d_x_sent = random.sample(
                    cat_x_sents, sentence_sample-1)+[CATEGORY_AUGMENT_SENTENCE_MAP[cat_x]]
                d_y_sent = random.sample(
                    cat_y_sents, sentence_sample-1)+[CATEGORY_AUGMENT_SENTENCE_MAP[cat_y]]

            core_cont_tups = list(itertools.product(
                d_x_sent,
                [d_x['trajectory_stats']],
                [d_x['category']],
                d_y_sent,
                [d_y['trajectory_stats']],
                [d_y['category']]
            ))
            contrastive_tuples.extend(core_cont_tups)

    return contrastive_tuples


def load_mc_benchmark_df(data_df):
    """load_mc_benchmark_df [summary]
    This fn will load combinatorial examples from dataset as is Without any form of sampling 
    This function is used After Using the `make_benchmark_dataset` to create smaller sample set from the 
    language examples and mc trajectories. 
    Use the `create_contrastive_examples` function to create dataset on the fly from large dataframe with lots of samples and lots of trajectories. 
    """
    assert 'category' in data_df.columns and 'trajectory_stats' in data_df.columns and 'sentences' in data_df.columns
    contrastive_tuples = []
    # create parts for each category at an arrray of dfs where each df represents data about one category.
    part_eps = [data_df[data_df['category'] == p] for p in range(1, 4)]
    part_recs = [p.to_dict(orient='records') for p in part_eps]
    combs = list(itertools.combinations(part_recs, 2))
    sent_cat_dict = make_cat_sent_dict(part_eps, 3)
    sentence_sampling_dict = {}
    for cat in sent_cat_dict:
        sentence_sampling_dict[cat] = list(sent_cat_dict[cat].keys())
    
    contrastive_tuples = make_samples(combs,sentence_sampling_dict,only_core_samples=False,load_saved=True,no_core_samples=False)

    return contrastive_tuples, sentence_sampling_dict # [(pos_sent,pos_traj,pos_cat,neg_sent,neg_traj,neg_cat)],{cat:['']}


def create_contrastive_examples(data_df, num_cats=3, episode_samples=10, sentence_sample=10, cat_sent_sample_size=20, only_core_samples=False):
    """create_contrastive_examples 
    This is the main function which helps create the contrasttive examples for MC dataset.
    """
    assert 'category' in data_df.columns and 'trajectory_stats' in data_df.columns and 'sentences' in data_df.columns
    contrastive_tuples = []
    # create parts for each category at an arrray of dfs where each df represents data about one category.
    part_eps = [data_df[data_df['category'] == p]
                for p in range(1, num_cats+1)]
    # Sample episodes if necessary
    if episode_samples:
        part_eps = [e.sample(episode_samples) for e in part_eps]

    part_recs = [p.to_dict(orient='records') for p in part_eps]

    # Find Combinations of the databased on the category splits
    # [([cat1_docs],[cat2_docs]),([cat2_docs],[cat3_docs]),([cat3_docs],[cat1_docs])]
    combs = list(itertools.combinations(part_recs, 2))

    # Extract Unique Sentences for each category
    sent_cat_dict = make_cat_sent_dict(part_eps, num_cats)

    # From unique sentences create samples
    # create a sample set for Sampling downstream when creating constrastive tuples
    sentence_sampling_dict = sample_category_sentences_from_dict(
        sent_cat_dict, cat_sent_sample_size)

    # Make sample will run the iterative combitation of the data.
    contrastive_tuples = make_samples(combs, sentence_sampling_dict, only_core_samples=only_core_samples,
                 sentence_sample=sentence_sample, load_saved=False)
    # [(pos_sent,pos_traj,pos_cat,neg_sent,neg_traj,neg_cat)],{cat:['']}
    return contrastive_tuples, sentence_sampling_dict


class ContrastiveTrainingDataset(Dataset):
    '''
    This is the core contrastive dataloading baseclass
    `is_benchmark` flag is when loading the explicit dataset with create for benchmarking 
    various experiments for various studies of models. 
    '''
    def __init__(self,
                 df,
                 tokenizer,
                 max_traj_length=200,
                 action_space=3,
                 action_type='discrete',
                 is_test=False,
                 is_benchmark=False,
                 num_cats=3,
                 num_episode_samples=EPISODE_SAMPLES,
                 num_sentences_per_episode=SENTENCE_PER_EPISODE,
                 sentence_sample_buffer_size=SENTENCE_SAMPLE_BUFFER_SIZE,
                 only_core_samples=False,
                 with_mask=False):

        self.num_cats = num_cats
        self.action_type = action_type
        self.max_traj_length = max_traj_length
        self.action_space = action_space
        self.tokenizer = tokenizer
        self.with_mask = with_mask
        # self.train_examples = self.create_contrastive_training_tuples(df, sample,)
        if not is_benchmark:
            self.train_examples, self.sentence_sampling_dict = create_contrastive_examples(df,
                                                                                       num_cats=num_cats,
                                                                                       only_core_samples=only_core_samples,
                                                                                       cat_sent_sample_size=sentence_sample_buffer_size,
                                                                                       episode_samples=num_episode_samples,
                                                                                       sentence_sample=num_sentences_per_episode)
        else:
            self.train_examples, self.sentence_sampling_dict = load_mc_benchmark_df(df)

    
    def __len__(self):
        return len(self.train_examples)

    def __getitem__(self, idx):
        # row = self.df.iloc[0]
        pos_sent_tup = self.encode_sentence(idx, pos=True)
        pos_traj = self.extract_trajectory(idx, pos=True)
        neg_sent_tup = self.encode_sentence(idx, pos=False)
        neg_traj = self.extract_trajectory(idx, pos=False)
        return (
            pos_sent_tup, pos_traj, neg_sent_tup, neg_traj, 0, 1
        )

    @staticmethod
    def make_mask_from_len(len_tensor, max_size):
        '''
        len_tensor: 
        '''
        return (torch.arange(max_size)[None, :] < len_tensor[:, None]).float()

    def encode_sentence(self, idx, pos=False):
        tuple_idx = None
        if pos:
            tuple_idx = 0
        else:
            tuple_idx = 3
        # print(self.train_examples[idx][tuple_idx])
        sentence = self.train_examples[idx][tuple_idx]
        # print(f"Encodeing sentences {sentence}")
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

    def extract_trajectory(self, idx, pos=False):
        tuple_idx = None
        if pos:
            tuple_idx = 1
        else:
            tuple_idx = 4

        # for obj in :
        act_list = list(
            map(lambda x: x['action'], self.train_examples[idx][tuple_idx]))
        state_list = list(
            map(lambda x: x['observation'], self.train_examples[idx][tuple_idx]))

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

        if self.action_type == 'discrete':
            action_tensor = torch.Tensor(act_list).type(torch.LongTensor)
        else:
            action_tensor = torch.Tensor(act_list)
        state_tensor = torch.Tensor(state_list)

        if not self.with_mask:
            return (state_tensor, action_tensor)
        else:
            return (state_tensor,
                    self.make_mask_from_len(torch.Tensor(
                        [state_list_len]), self.max_traj_length).squeeze(0),
                    action_tensor,
                    self.make_mask_from_len(torch.tensor(
                        [act_list_len]), self.max_traj_length).squeeze(0)
                    )

    def check_incompatible_groups(self, cont_cat, cat):
        if cont_cat == cat:
            return False
        return True


class BertEmbedContrastiveTokenizedDataset(ContrastiveTrainingDataset):
    def __init__(self, *args, with_cats=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_cats = with_cats

    def __len__(self):
        return len(self.train_examples)

    def __getitem__(self, idx):
        # row = self.df.iloc[0]
        if not self.with_mask:
            pos_sent, _ = self.encode_sentence(idx, pos=True)
            pos_traj = self.extract_trajectory(idx, pos=True)
            neg_sent, _ = self.encode_sentence(idx, pos=False)
            neg_traj = self.extract_trajectory(idx, pos=False)
            if not self.with_cats:
                return (
                    pos_sent, pos_traj, neg_sent, neg_traj
                )
            else:
                return (
                    pos_sent, pos_traj, neg_sent, neg_traj, self.train_examples[
                        idx][2], self.train_examples[idx][5]
                )
        else:
            pos_sent, pos_sent_mask = self.encode_sentence(idx, pos=True)
            pos_traj = self.extract_trajectory(idx, pos=True)
            neg_sent, neg_sent_mask = self.encode_sentence(idx, pos=False)
            neg_traj = self.extract_trajectory(idx, pos=False)
            if not self.with_cats:
                return (
                    pos_sent, pos_sent_mask, pos_traj, neg_sent, neg_sent_mask, neg_traj,
                )
            else:
                return (
                    pos_sent, pos_sent_mask, pos_traj, neg_sent, neg_sent_mask, neg_traj, self.train_examples[
                        idx][2], self.train_examples[idx][5]
                )


class BertEmbedContrastiveTokenizedDatasetWithSentences(ContrastiveTrainingDataset):

    def __init__(self, *args, with_mask=False, with_cats=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_cats = with_cats
        self.with_mask = with_mask

    def __len__(self):
        return len(self.train_examples)

    def __getitem__(self, idx):
        # row = self.df.iloc[0]
        # if not self.with_mask:
        if not self.with_mask:
            pos_sent, _, pos_sentence = self.encode_sentence(idx, pos=True)
            pos_traj = self.extract_trajectory(idx, pos=True)
            neg_sent, _, neg_sentence = self.encode_sentence(idx, pos=False)
            neg_traj = self.extract_trajectory(idx, pos=False)
            if not self.with_cats:
                return (
                    pos_sent, pos_traj, pos_sentence, neg_sent, neg_traj, neg_sentence
                )
            else:
                return (
                    pos_sent, pos_traj, pos_sentence, neg_sent, neg_traj, neg_sentence, self.train_examples[
                        idx][2], self.train_examples[idx][5]
                )
        else:
            pos_sent, pos_sentence_mask, pos_sentence = self.encode_sentence(
                idx, pos=True)
            pos_traj = self.extract_trajectory(idx, pos=True)
            neg_sent, neg_sentence_mask, neg_sentence = self.encode_sentence(
                idx, pos=False)
            neg_traj = self.extract_trajectory(idx, pos=False)
            if not self.with_cats:
                return (
                    pos_sent, pos_sentence_mask, pos_traj, pos_sentence, neg_sent, neg_sentence_mask, neg_traj, neg_sentence
                )
            else:
                return (
                    pos_sent, pos_sentence_mask, pos_traj, pos_sentence, neg_sent, neg_sentence_mask, neg_traj, neg_sentence, self.train_examples[
                        idx][2], self.train_examples[idx][5]
                )

    def encode_sentence(self, idx, pos=False):
        tuple_idx = None
        if pos:
            tuple_idx = 0
        else:
            tuple_idx = 3
        # print(self.train_examples[idx][tuple_idx])
        sentence = self.train_examples[idx][tuple_idx]
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
        return (data_dict['input_ids'].squeeze(0), data_dict['attention_mask'].squeeze(0), sentence)


def contloader_collate_fn(batch):
    pst = []
    ptjs = []
    ptja = []
    nst = []
    ntjs = []
    ntja = []

    ps = []
    ns = []
    for item in batch:
        pos_sent, pos_traj, pos_sentence, neg_sent, neg_traj, neg_sentence = item
        pst.append(pos_sent)
        ptjs.append(pos_traj[0])
        ptja.append(pos_traj[1])
        nst.append(neg_sent)
        ntjs.append(neg_traj[0])
        ntja.append(neg_traj[1])
        ps.append(pos_sentence)
        ns.append(neg_sentence)

    return (
        torch.stack(pst),  # pos_sent,
        (torch.stack(ptjs), torch.stack(ptja)),  # pos_traj,
        ps,  # pos_sentences
        torch.stack(nst),  # neg_sent
        (torch.stack(ntjs), torch.stack(ntja)),  # ,neg_traj
        ns,  # ,neg_sentence
    )


def contloader_collate_fn_with_mask(batch):
    pst = []
    psta = []
    ptjs = []
    ptjsa = []
    ptja = []
    ptjaa = []

    nst = []
    nsta = []
    ntjs = []
    nsjsa = []
    ntja = []
    nsjaa = []

    ps = []
    ns = []
    for item in batch:
        pos_sent, pos_sentence_mask, pos_traj, pos_sentence, neg_sent, neg_sentence_mask, neg_traj, neg_sentence = item
        # pos_sent,pos_sent_mask,pos_traj,neg_sent,neg_sent_mask,neg_traj = item
        pst.append(pos_sent)
        psta.append(pos_sentence_mask)
        ptjs.append(pos_traj[0])
        ptjsa.append(pos_traj[1])
        ptja.append(pos_traj[2])
        ptjaa.append(pos_traj[3])

        nst.append(neg_sent)
        nsta.append(neg_sentence_mask)
        ntjs.append(neg_traj[0])
        nsjsa.append(neg_traj[1])
        ntja.append(neg_traj[2])
        nsjaa.append(neg_traj[3])
        ps.append(pos_sentence)
        ns.append(neg_sentence)

    return (
        torch.stack(pst),  # pos_sent,
        torch.stack(psta),
        (torch.stack(ptjs), torch.stack(ptjsa), torch.stack(
            ptja), torch.stack(ptjaa)),  # pos_traj,
        ps,  # pos_sentences
        torch.stack(nst),  # neg_sent
        torch.stack(nsta),
        (torch.stack(ntjs), torch.stack(nsjsa), torch.stack(
            ntja), torch.stack(nsjaa)),  # ,neg_traj
        ns,  # ,neg_sentence

    )


def contloader_collate_fn_with_mask_and_cats(batch):
    pst = []
    psta = []
    ptjs = []
    ptjsa = []
    ptja = []
    ptjaa = []

    nst = []
    nsta = []
    ntjs = []
    nsjsa = []
    ntja = []
    nsjaa = []

    ps = []
    ns = []

    pcat = []
    ncat = []
    for item in batch:
        pos_sent, pos_sentence_mask, pos_traj, pos_sentence, neg_sent, neg_sentence_mask, neg_traj, neg_sentence, p_cat, n_cat = item
        # pos_sent,pos_sent_mask,pos_traj,neg_sent,neg_sent_mask,neg_traj = item
        pst.append(pos_sent)
        psta.append(pos_sentence_mask)
        ptjs.append(pos_traj[0])
        ptjsa.append(pos_traj[1])
        ptja.append(pos_traj[2])
        ptjaa.append(pos_traj[3])

        nst.append(neg_sent)
        nsta.append(neg_sentence_mask)
        ntjs.append(neg_traj[0])
        nsjsa.append(neg_traj[1])
        ntja.append(neg_traj[2])
        nsjaa.append(neg_traj[3])
        ps.append(pos_sentence)
        ns.append(neg_sentence)
        pcat.append(p_cat)
        ncat.append(n_cat)

    return (
        torch.stack(pst),  # pos_sent,
        torch.stack(psta),
        (torch.stack(ptjs), torch.stack(ptjsa), torch.stack(
            ptja), torch.stack(ptjaa)),  # pos_traj,
        ps,  # pos_sentences
        torch.stack(nst),  # neg_sent
        torch.stack(nsta),
        (torch.stack(ntjs), torch.stack(nsjsa), torch.stack(
            ntja), torch.stack(nsjaa)),  # ,neg_traj
        ns,  # ,neg_sentence
        torch.Tensor(pcat),
        torch.Tensor(ncat),
    )
