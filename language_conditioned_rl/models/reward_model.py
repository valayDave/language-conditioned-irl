import torch.nn as nn
import torch
import pytorch_lightning as pl
from dataclasses import dataclass
from .optimizer import PCGrad
from .trasformer import PRETRAINED_MODEL, CrossModalBertEmbedTranformer

DEFAULT_CHECKPOINT_PROJECT_NAME = 'valay/Language-Grounded-Rewards'
DEFAULT_CHECKPOINT_EXPERIMENT_NAME = 'LAN-21'
DEFAULT_CHECKPOINT_PATH = 'checkpoints/epoch=12-val_loss=0.98.ckpt'


@dataclass
class DataAndOptimizerConf:
    NUM_TRAIN_SAMPLES:int = 10000
    BATCH_SIZE:int = 15
    WARMUP:int = 2000
    MAX_CYCLES:int = 10
    MAX_EPOCHS:int = 25
    LEARNING_RATE:float = 0.001
    

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class BehaviouralCategorical(torch.nn.Module):
    def __init__(self, backbone_dims, num_outputs=4):
        super().__init__()
        self.ff_layer = nn.Linear(backbone_dims, num_outputs)
        self.sfmx = torch.nn.Softmax(dim=1)

    def forward(self, backbone_features):
        return self.sfmx(self.ff_layer(backbone_features))


class RewardHead(torch.nn.Module):
    def __init__(self, backbone_dims, hidden_dims, with_categorical=True, num_outputs=4):
        super().__init__()
        self.sfmx = torch.nn.Softmax(dim=1)
        self.backbone_conv = nn.Conv1d(
            backbone_dims, hidden_dims, kernel_size=1, padding=0, bias=False)
        self.categorical_conv = nn.Conv1d(
            num_outputs, hidden_dims, kernel_size=1, padding=0, bias=False)
        self.reward_converter = nn.Linear(hidden_dims*2, 1)

    def get_joint_features(self, backbone_features, categorical):
        backbone_converted_features = backbone_features.unsqueeze(
            1).transpose(2, 1)
        backbone_features = self.backbone_conv(
            backbone_converted_features).permute(0, 2, 1).squeeze(1)
        categorical_converted_features = categorical.unsqueeze(
            1).transpose(2, 1)
        categorical_features = self.categorical_conv(
            categorical_converted_features).permute(0, 2, 1).squeeze(1)
        return torch.cat((backbone_features, categorical_features), dim=1)

    def forward(self, backbone_features, categorical):
        join_features = self.get_joint_features(backbone_features, categorical)
        return self.reward_converter(join_features)


class LGRBehaviouralDiffLearnerPCGrad(pl.LightningModule):
    """LGRBehaviouralDiffLearnerPCGrad [summary]
    Uses a contrastive Loss and a Categorical Cross Entropy Loss to create the Model.
    Uses Loss Metric Contrastive Loss : https://arxiv.org/abs/2009.01325

    """

    def __init__(self,
                 num_layers=3,
                 dropout=0.1,
                 num_attention_heads=4,
                 scale=0.2,
                 pretrained_model=PRETRAINED_MODEL,
                 num_actions=3,
                 common_conv_dim=128,
                 embedding_size=256,
                 layer_norm_epsilon=0.00001,
                 resid_pdrop=0.1,
                 embd_pdrop=0.1,
                 attn_pdrop=0.1,
                 state_dims=2,
                 action_type='discrete',
                 temperature=0.5,
                 loss_scale=None,
                 data_params: DataAndOptimizerConf = DataAndOptimizerConf()
                 ):
        super().__init__()
        core_tran_params = dict(
            num_layers=int(num_layers),
            common_conv_dim=int(common_conv_dim),
            embedding_size=int(embedding_size),
            layer_norm_epsilon=layer_norm_epsilon,
            scale=scale,
            pretrained_model=pretrained_model,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
            num_heads=int(num_attention_heads),
            embd_pdrop=embd_pdrop,
            state_dims=int(state_dims),
            num_actions=int(num_actions),
            action_type=action_type,
        )
        common_conv_dim = core_tran_params['common_conv_dim']
        self.model = CrossModalBertEmbedTranformer(**core_tran_params)
        self.embed_loss = nn.CosineEmbeddingLoss(margin=0.1)
        self.sfmx = nn.Softmax(dim=1)
        self.log_sigmoid = nn.LogSigmoid()
        # Output Passed to Zero or one on sigmoid activation
        self.classification_layer = nn.Linear(common_conv_dim*3*2, 1)
        self.behavioural_differential = BehaviouralCategorical(
            common_conv_dim*3*2, num_outputs=4)
        self.reward_predictor = RewardHead(
            common_conv_dim*3*2, common_conv_dim,)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.data_params = data_params

    def manual_backward(self, loss, optimizer):
        optim_pc = PCGrad(optimizer)
        optim_pc.pc_backward(loss)

    # state,action should be full trajectory sequences for state and action for each element in the batch.
    def forward(self, state, action, text, text_mask=None, act_mask=None, st_mask=None):
        '''
        text : (input_ids)
            - input_ids: b k :
              - b: batchsize
              - k: sequence_length
        state: state_tensor : b t s:
              - b: batchsize
              - t: trajectory length
              - s: state dimensions
        action: action_tensor: b t | b t d: would change based on discrete and continous action spaces.
              - b: batchsize
              - t: trajectory length
        *_mask = mask: b s : binary tensor.
        '''
        return self.model(state, action, text, text_mask=text_mask, act_mask=act_mask, st_mask=st_mask)

    def custom_loss_fn(self, pos_exp, neg_exp):
        return - self.log_sigmoid(pos_exp - neg_exp).mean()

    def get_backboone_features(self, batch):
        pos_sent, pos_sent_mask, pos_traj, neg_sent, neg_sent_mask, neg_traj, pos_cat, neg_cat = batch
        # pos_cat.to(self.device)
        # neg_cat.to(self.device)
        # print('pos_cat,neg_cat',pos_cat.device,neg_cat.device)
        pos_state = pos_traj[0]
        pos_state_mask = pos_traj[1]
        pos_action = pos_traj[2]
        pos_action_mask = pos_traj[3]

        neg_state = neg_traj[0]
        neg_state_mask = neg_traj[1]
        neg_action = neg_traj[2]
        neg_action_mask = neg_traj[3]

        pp_tensor = self(pos_state, pos_action, pos_sent, text_mask=pos_sent_mask,
                         act_mask=pos_action_mask, st_mask=pos_state_mask)  # P P
        np_tensor = self(neg_state, neg_action, pos_sent, text_mask=pos_sent_mask,
                         act_mask=neg_action_mask, st_mask=neg_state_mask)  # N P
        nn_tensor = self(neg_state, neg_action, neg_sent, text_mask=neg_sent_mask,
                         act_mask=neg_action_mask, st_mask=neg_state_mask)  # N N
        pn_tensor = self(pos_state, pos_action, neg_sent, text_mask=neg_sent_mask,
                         act_mask=pos_action_mask, st_mask=pos_state_mask)  # P N
        return ((
            pp_tensor,
            np_tensor,
            nn_tensor,
            pn_tensor
        ), (pos_cat, neg_cat))

    def get_reward_and_category(self, feature_tensor):
        categorical = self.behavioural_differential(feature_tensor)
        return (categorical, self.reward_predictor(feature_tensor, categorical))

    def get_reward_and_category_with_loss(self, feature_tensor, ground_truth_category):
        categorical, reward = self.get_reward_and_category(feature_tensor)
        ground_truth_category = ground_truth_category - 1
        categoriacal_loss = self.cross_entropy(
            categorical, ground_truth_category)
        return categorical, reward, categoriacal_loss

    def reward_fn(self, state, action, text, text_mask=None, act_mask=None, st_mask=None):
        with torch.no_grad():
            features = self.model(
                state, action, text, text_mask=text_mask, act_mask=act_mask, st_mask=st_mask)
            category, reward = self.get_reward_and_category(features)
        return reward

    def training_step(self, batch, batch_nb):
        intermediate_feature_tuple, category_tuple = self.get_backboone_features(
            batch)
        pp_tensor, np_tensor, nn_tensor, pn_tensor = intermediate_feature_tuple
        pos_cat, neg_cat = category_tuple

        pp_categorical, pp_reward, pp_categoriacal_loss = self.get_reward_and_category_with_loss(
            pp_tensor, pos_cat)
        pn_categorical, pn_reward, pn_categoriacal_loss = self.get_reward_and_category_with_loss(
            pn_tensor, pos_cat)
        nn_categorical, nn_reward, nn_categoriacal_loss = self.get_reward_and_category_with_loss(
            nn_tensor, neg_cat)
        np_categorical, np_reward, np_categoriacal_loss = self.get_reward_and_category_with_loss(
            np_tensor, neg_cat)

        loss_categorical = torch.mean(torch.stack(
            [pp_categoriacal_loss, pn_categoriacal_loss, nn_categoriacal_loss, np_categoriacal_loss]))

        p_loss = self.custom_loss_fn(pp_reward, pn_reward)
        n_loss = self.custom_loss_fn(nn_reward, np_reward)

        loss_reward_diff = torch.mean(torch.stack([p_loss, n_loss]))

        loss = loss_reward_diff + loss_categorical

        pc_grad_losses = [
            loss_reward_diff,
            loss_categorical
        ]
        optimizer = self.optimizers()
        self.manual_backward(pc_grad_losses, optimizer)
        optimizer.step()

        self.logger.log_metrics({
            'train_loss_categorical': loss_categorical.detach().cpu().numpy(),
            'train_loss_reward_diff': loss_reward_diff.detach().cpu().numpy(),
            # 'train_loss_embed':(loss_embed_pc+loss_embed_nc).detach().cpu().numpy(),
            'train_loss': loss.detach().cpu().numpy(),
            'epoch': self.current_epoch,
        })
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        intermediate_feature_tuple, category_tuple = self.get_backboone_features(
            batch)
        pp_tensor, np_tensor, nn_tensor, pn_tensor = intermediate_feature_tuple
        pos_cat, neg_cat = category_tuple
        pp_categorical, pp_reward, pp_categoriacal_loss = self.get_reward_and_category_with_loss(
            pp_tensor, pos_cat)
        pn_categorical, pn_reward, pn_categoriacal_loss = self.get_reward_and_category_with_loss(
            pn_tensor, pos_cat)
        nn_categorical, nn_reward, nn_categoriacal_loss = self.get_reward_and_category_with_loss(
            nn_tensor, neg_cat)
        np_categorical, np_reward, np_categoriacal_loss = self.get_reward_and_category_with_loss(
            np_tensor, neg_cat)

        loss_categorical = torch.mean(torch.stack(
            [pp_categoriacal_loss, pn_categoriacal_loss, nn_categoriacal_loss, np_categoriacal_loss]))

        p_loss = self.custom_loss_fn(pp_reward, pn_reward)
        n_loss = self.custom_loss_fn(nn_reward, np_reward)

        loss_reward_diff = torch.mean(torch.stack([p_loss, n_loss]))

        loss = loss_reward_diff + loss_categorical

        self.logger.log_metrics({
            'val_loss_categorical': loss_categorical.detach().cpu().numpy(),
            'val_loss_reward_diff': loss_reward_diff.detach().cpu().numpy(),
            # 'valin_loss_embed':(loss_embed_pc+loss_embed_nc).detach().cpu().numpy(),
            'val_loss': loss.detach().cpu().numpy(),
            'epoch': self.current_epoch,
        })
        return {'loss': loss, 'val_loss': loss.detach().cpu()}

    def test_step(self, batch, batch_nb):
        intermediate_feature_tuple, category_tuple = self.get_backboone_features(
            batch)
        pp_tensor, np_tensor, nn_tensor, pn_tensor = intermediate_feature_tuple
        pos_cat, neg_cat = category_tuple
        pp_categorical, pp_reward, pp_categoriacal_loss = self.get_reward_and_category_with_loss(
            pp_tensor, pos_cat)
        pn_categorical, pn_reward, pn_categoriacal_loss = self.get_reward_and_category_with_loss(
            pn_tensor, pos_cat)
        nn_categorical, nn_reward, nn_categoriacal_loss = self.get_reward_and_category_with_loss(
            nn_tensor, neg_cat)
        np_categorical, np_reward, np_categoriacal_loss = self.get_reward_and_category_with_loss(
            np_tensor, neg_cat)

        loss_categorical = torch.mean(torch.stack(
            [pp_categoriacal_loss, pn_categoriacal_loss, nn_categoriacal_loss, np_categoriacal_loss]))

        p_loss = self.custom_loss_fn(pp_reward, pn_reward)
        n_loss = self.custom_loss_fn(nn_reward, np_reward)

        loss_reward_diff = torch.mean(torch.stack([p_loss, n_loss]))

        loss = loss_reward_diff + loss_categorical

        self.logger.log_metrics({
            'test_loss_categorical': loss_categorical.detach().cpu().numpy(),
            'test_loss_reward_diff': loss_reward_diff.detach().cpu().numpy(),
            # 'testin_loss_embed':(loss_embed_pc+loss_embed_nc).detach().cpu().numpy(),
            'test_loss': loss.detach().cpu().numpy(),
            'epoch': self.current_epoch,
        })
        return {'loss': loss}

    def configure_optimizers(self):
        from transformers import AdamW, get_cosine_with_hard_restarts_schedule_with_warmup

        optimizer = AdamW(self.parameters(), lr=self.data_params.LEARNING_RATE,
                          eps=1e-12, betas=(0.9, 0.999))
        num_minibatch_steps = (
            self.data_params.NUM_TRAIN_SAMPLES)/(self.data_params.BATCH_SIZE)
        max_epochs = self.data_params.MAX_EPOCHS
        warmup = self.data_params.WARMUP
        t_total = max_epochs * num_minibatch_steps
        num_cycles = self.data_params.MAX_CYCLES
        lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer, warmup, t_total, num_cycles=num_cycles)
        return [optimizer], [lr_scheduler]


class LGRInferenceMixin(object):
    def __init__(self,
                 max_traj_length=200,
                 action_space=3,
                 max_text_len=25,
                 action_type='discrete',
                 experiment_name=None,
                 loaded_checkpoint=None,
                 pretrained_model=PRETRAINED_MODEL):

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.max_traj_length = max_traj_length
        self.action_space = action_space
        self.max_text_len = max_text_len
        self.action_type = action_type
        self.experiment_name = experiment_name
        self.loaded_checkpoint = loaded_checkpoint

    def encode_sent(self, sents, max_text_len=25):
        data_dict = self.tokenizer.batch_encode_plus(
            sents,                      # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_text_len,           # Pad & truncate all sentences.
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',     # Return pytorch tensors.
        )
        return (data_dict['input_ids'], data_dict['attention_mask'])

    def encode_trajectory(self, act_list, state_list):
        act_list_len = len(act_list)
        state_list_len = len(state_list)

        if self.max_traj_length > len(act_list):
            # Adding A NULL Action token which is a part of the transformer to ensure null action
            act_list.extend([self.action_space for _ in range(
                self.max_traj_length-len(act_list))])
        if self.max_traj_length > len(state_list):
            # Repeat last state in the state list
            state_list.extend([state_list[-1]
                               for _ in range(self.max_traj_length-len(state_list))])

        if self.action_type == 'discrete':
            action_tensor = torch.Tensor(act_list).type(torch.LongTensor)
        else:
            action_tensor = torch.Tensor(act_list)
        state_tensor = torch.Tensor(state_list)

        return (
            state_tensor.unsqueeze(0),
            self.make_mask_from_len(torch.Tensor(
                [state_list_len]), self.max_traj_length),
            action_tensor.unsqueeze(0),
            self.make_mask_from_len(torch.tensor(
                [act_list_len]), self.max_traj_length)
        )

    @staticmethod
    def make_mask_from_len(len_tensor, max_size):
        '''
        len_tensor: 
        '''
        return (torch.arange(max_size)[None, :] < len_tensor[:, None]).float()

    @classmethod
    def from_neptune(cls,
                     project_name,
                     experiment_name,
                     checkpoint_path, base_path='model_checkpoints/',
                     api_token=None,):
        import neptune
        import os
        if api_token is None:
            raise Exception("API Token Missing")

        project = neptune.init(project_name,
                               api_token=api_token
                               )
        my_exp = project.get_experiments(id=experiment_name)[0]
        my_exp.download_artifact(checkpoint_path, destination_dir=base_path)
        ckck_name = checkpoint_path.split('/')[1]
        checkpoint = torch.load(os.path.join(
            base_path, ckck_name), map_location=torch.device('cpu'))

        config = my_exp.get_parameters()
        if 'note' in config:
            del config['note']
        if 'loss_scale' in config:
            del config['loss_scale']
        if 'data_params' in config:
            del config['data_params']
        # print(config)
        if 'transformer_params' in config: # The was after Bringing new ddataset to log everything properly
            config = config['transformer_params']
        
        trans = cls(**config, experiment_name=experiment_name,
                    loaded_checkpoint=checkpoint_path)

        missing_keys, unexpected_keys = trans.load_state_dict(
            checkpoint['state_dict'])
        # print(f'missing_keys ,unexpected_keys, {missing_keys ,unexpected_keys}')
        return trans, config

    def get_rewards(self, state, action, text: str):
        '''
        gives scalar reward given 1 trajectory and 1 text 
        state: []
        action : []
        text: str : "the car moved up the hill"
        '''
        state, state_mask, action, action_mask = self.encode_trajectory(
            action, state)
        text_tensor, text_mask = self.encode_sent([text])
        reward_op = self.reward_fn(
            state, action, text_tensor, text_mask=text_mask, act_mask=action_mask, st_mask=state_mask
        )
        # print(reward_op)
        return reward_op[0].item()


class RewardHeadWithOnlyBackbone(torch.nn.Module):
    def __init__(self, backbone_dims, hidden):
        super().__init__()
        self.reward_converter = nn.Sequential(
            nn.Linear(backbone_dims, hidden),
            nn.Linear(hidden, 1),
        )

    def forward(self, backbone_features):
        return self.reward_converter(backbone_features)


class LGRRewardOnlyHeadLearner(pl.LightningModule):
  def __init__(self,
               num_layers=3,
               dropout=0.1,
               num_attention_heads=4,
               scale=0.2,
               pretrained_model=PRETRAINED_MODEL,
               num_actions=3,
               common_conv_dim=128,
               embedding_size=256,
               layer_norm_epsilon=0.00001,
               resid_pdrop=0.1,
               embd_pdrop=0.1,
               attn_pdrop=0.1,
               state_dims=2,
               action_type='discrete',
               data_params: DataAndOptimizerConf = DataAndOptimizerConf()
               ):
    super().__init__()
    core_tran_params = dict(
        num_layers=int(num_layers),
        common_conv_dim=int(common_conv_dim),
        embedding_size=int(embedding_size),
        layer_norm_epsilon=layer_norm_epsilon,
        scale=scale,
        pretrained_model=pretrained_model,
        resid_pdrop=resid_pdrop,
        attn_pdrop=attn_pdrop,
        num_heads=int(num_attention_heads),
        embd_pdrop=embd_pdrop,
        state_dims=int(state_dims),
        num_actions=int(num_actions),
        action_type=action_type,
    )
    common_conv_dim = core_tran_params['common_conv_dim']
    self.model = CrossModalBertEmbedTranformer(**core_tran_params)
    self.embed_loss = nn.CosineEmbeddingLoss(margin=0.1)
    self.sfmx = nn.Softmax(dim=1)
    self.log_sigmoid = nn.LogSigmoid()
    # Output Passed to Zero or one on sigmoid activation
    self.classification_layer = nn.Linear(common_conv_dim*3*2, 1)
    self.behavioural_differential = BehaviouralCategorical(
        common_conv_dim*3*2, num_outputs=4)
    self.reward_predictor = RewardHeadWithOnlyBackbone(
        common_conv_dim*3*2, common_conv_dim,)
    self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
    self.data_params = data_params

  # state,action should be full trajectory sequences for state and action for each element in the batch.
  def forward(self, state, action, text, text_mask=None, act_mask=None, st_mask=None):
    '''
    text : (input_ids)
        - input_ids: b k :
          - b: batchsize
          - k: sequence_length
    state: state_tensor : b t s:
          - b: batchsize
          - t: trajectory length
          - s: state dimensions
    action: action_tensor: b t | b t d: would change based on discrete and continous action spaces.
          - b: batchsize
          - t: trajectory length
    *_mask = mask: b s : binary tensor. 
    '''
    return self.model(state, action, text, text_mask=text_mask, act_mask=act_mask, st_mask=st_mask)

  def custom_loss_fn(self, pos_exp, neg_exp):
    return - self.log_sigmoid(pos_exp - neg_exp).mean()

  def get_backboone_features(self, batch):
    pos_sent, pos_sent_mask, pos_traj, neg_sent, neg_sent_mask, neg_traj, pos_cat, neg_cat = batch
    # pos_cat.to(self.device)
    # neg_cat.to(self.device)
    # print('pos_cat,neg_cat',pos_cat.device,neg_cat.device)
    pos_state = pos_traj[0]
    pos_state_mask = pos_traj[1]
    pos_action = pos_traj[2]
    pos_action_mask = pos_traj[3]

    neg_state = neg_traj[0]
    neg_state_mask = neg_traj[1]
    neg_action = neg_traj[2]
    neg_action_mask = neg_traj[3]
    # This will all break with return_attentions is True at training time.
    pp_tensor = self(pos_state, pos_action, pos_sent, text_mask=pos_sent_mask,
                     act_mask=pos_action_mask, st_mask=pos_state_mask)  # P P
    np_tensor = self(neg_state, neg_action, pos_sent, text_mask=pos_sent_mask,
                     act_mask=neg_action_mask, st_mask=neg_state_mask)  # N P
    nn_tensor = self(neg_state, neg_action, neg_sent, text_mask=neg_sent_mask,
                     act_mask=neg_action_mask, st_mask=neg_state_mask)  # N N
    pn_tensor = self(pos_state, pos_action, neg_sent, text_mask=neg_sent_mask,
                     act_mask=pos_action_mask, st_mask=pos_state_mask)  # P N
    return ((
        pp_tensor,
        np_tensor,
        nn_tensor,
        pn_tensor
    ), (pos_cat, neg_cat))

  def get_reward_from_features(self, feature_tensor):
    return self.reward_predictor(feature_tensor)

  def reward_fn(self, state, action, text, text_mask=None, act_mask=None, st_mask=None):
    with torch.no_grad():
      features = self.model(state.to(self.device), action.to(self.device), text.to(self.device), text_mask=text_mask.to(
          self.device), act_mask=act_mask.to(self.device), st_mask=st_mask.to(self.device))
      reward = self.get_reward_from_features(features)
    return reward

  def training_step(self, batch, batch_nb):
    intermediate_feature_tuple, category_tuple = self.get_backboone_features(
        batch)
    pp_tensor, np_tensor, nn_tensor, pn_tensor = intermediate_feature_tuple
    pos_cat, neg_cat = category_tuple

    pp_reward = self.get_reward_from_features(pp_tensor)
    pn_reward = self.get_reward_from_features(pn_tensor)
    nn_reward = self.get_reward_from_features(nn_tensor)
    np_reward = self.get_reward_from_features(np_tensor)

    p_loss = self.custom_loss_fn(pp_reward, np_reward)
    n_loss = self.custom_loss_fn(nn_reward, pn_reward)

    loss_reward_diff = torch.mean(torch.stack([p_loss, n_loss]))

    loss = loss_reward_diff

    self.logger.log_metrics({
        'train_loss': loss.detach().cpu().numpy(),
        'epoch': self.current_epoch,
    })
    return {'loss': loss}

  def validation_step(self, batch, batch_nb):
    intermediate_feature_tuple, category_tuple = self.get_backboone_features(
        batch)
    pp_tensor, np_tensor, nn_tensor, pn_tensor = intermediate_feature_tuple
    pos_cat, neg_cat = category_tuple

    pp_reward = self.get_reward_from_features(pp_tensor)
    pn_reward = self.get_reward_from_features(pn_tensor)
    nn_reward = self.get_reward_from_features(nn_tensor)
    np_reward = self.get_reward_from_features(np_tensor)

    p_loss = self.custom_loss_fn(pp_reward, np_reward)
    n_loss = self.custom_loss_fn(nn_reward, pn_reward)

    loss_reward_diff = torch.mean(torch.stack([p_loss, n_loss]))

    loss = loss_reward_diff

    self.logger.log_metrics({
        'val_loss': loss.detach().cpu().numpy(),
        'epoch': self.current_epoch,
    })
    return {'loss': loss, 'val_loss': loss.detach().cpu()}

  def test_step(self, batch, batch_nb):
    intermediate_feature_tuple, category_tuple = self.get_backboone_features(
        batch)
    pp_tensor, np_tensor, nn_tensor, pn_tensor = intermediate_feature_tuple
    pos_cat, neg_cat = category_tuple

    pp_reward = self.get_reward_from_features(pp_tensor)
    pn_reward = self.get_reward_from_features(pn_tensor)
    nn_reward = self.get_reward_from_features(nn_tensor)
    np_reward = self.get_reward_from_features(np_tensor)

    p_loss = self.custom_loss_fn(pp_reward, np_reward)
    n_loss = self.custom_loss_fn(nn_reward, pn_reward)

    loss_reward_diff = torch.mean(torch.stack([p_loss, n_loss]))

    loss = loss_reward_diff

    self.logger.log_metrics({
        'test_loss': loss.detach().cpu().numpy(),
        'epoch': self.current_epoch,
    })
    return {'loss': loss}

  def configure_optimizers(self):
    from transformers import AdamW, get_cosine_with_hard_restarts_schedule_with_warmup

    optimizer = AdamW(self.parameters(), lr=self.data_params.LEARNING_RATE,
                        eps=1e-12, betas=(0.9, 0.999))
    num_minibatch_steps = (
        self.data_params.NUM_TRAIN_SAMPLES)/(self.data_params.BATCH_SIZE)
    max_epochs = self.data_params.MAX_EPOCHS
    warmup = self.data_params.WARMUP
    t_total = max_epochs * num_minibatch_steps
    num_cycles = self.data_params.MAX_CYCLES
    lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer, warmup, t_total, num_cycles=num_cycles)
    return [optimizer], [lr_scheduler]



class LGRBehaviouralDiffLearnerInference(LGRInferenceMixin, LGRBehaviouralDiffLearnerPCGrad):
    def __init__(self,
                 *args,
                 max_traj_length=200,
                 action_space=3,
                 max_text_len=25,
                 action_type='discrete',
                 experiment_name=None,
                 loaded_checkpoint=None,
                 pretrained_model=PRETRAINED_MODEL, **kwargs):
        LGRBehaviouralDiffLearnerPCGrad.__init__(self, *args, **kwargs)
        LGRInferenceMixin.__init__(self,
                                   max_traj_length=max_traj_length,
                                   action_space=action_space,
                                   max_text_len=max_text_len,
                                   action_type=action_type,
                                   experiment_name=experiment_name,
                                   loaded_checkpoint=loaded_checkpoint,
                                   pretrained_model=pretrained_model)


class LGRPureContrastiveRewardLearner(LGRInferenceMixin, LGRRewardOnlyHeadLearner):
    def __init__(self,
                 *args,
                 max_traj_length=200,
                 action_space=3,
                 max_text_len=25,
                 action_type='discrete',
                 experiment_name=None,
                 loaded_checkpoint=None,
                 pretrained_model=PRETRAINED_MODEL, **kwargs):
        LGRRewardOnlyHeadLearner.__init__(self, *args, **kwargs)
        LGRInferenceMixin.__init__(self,
                                   max_traj_length=max_traj_length,
                                   action_space=action_space,
                                   max_text_len=max_text_len,
                                   action_type=action_type,
                                   experiment_name=experiment_name,
                                   loaded_checkpoint=loaded_checkpoint,
                                   pretrained_model=pretrained_model)




    
