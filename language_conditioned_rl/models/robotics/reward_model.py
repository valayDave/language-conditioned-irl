import pytorch_lightning as pl 
import torch
import torch.nn as nn
from typing import List
from ...dataloaders.channel import ContrastiveGenerator,ChannelData
from ..transformer import OmniTransformerCoreConfig,OmniChannelTransformer
from ..reward_model import DataAndOptimizerConf,RewardHeadWithOnlyBackbone

class LGRRoboRewardLearner(pl.LightningModule):
  def __init__(self,
               config:OmniTransformerCoreConfig,
               data_params: DataAndOptimizerConf = DataAndOptimizerConf()
               ):
    super().__init__()
    self.model = OmniChannelTransformer(config)
    self.sfmx = nn.Softmax(dim=1)
    self.model.final_layer_dims 
    self.log_sigmoid = nn.LogSigmoid()
    # Output Passed to Zero or one on sigmoid activation
    self.reward_predictor = RewardHeadWithOnlyBackbone(
        self.model.final_layer_dims , self.model.config.transformer_embedding_size)
    self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
    self.data_params = data_params

  # state,action should be full trajectory sequences for state and action for each element in the batch.
  def forward(self,input_channels: List[ChannelData],return_attentions=False):
    return self.model(input_channels,return_attentions=return_attentions)

  def custom_loss_fn(self, pos_exp, neg_exp):
    return - self.log_sigmoid(pos_exp - neg_exp).mean()

  def get_backboone_features(self, batch:ContrastiveGenerator):
    pp_channels,pn_channels,nn_channels,np_channels = batch.create_contrastive_inputs('text')
    pp_tensor,_ = self(pp_channels)  # P P
    np_tensor,_ = self(pn_channels)  # N P
    nn_tensor,_ = self(nn_channels)  # N N
    pn_tensor,_ = self(np_channels)  # P N
    return (
        pp_tensor,
        np_tensor,
        nn_tensor,
        pn_tensor
    )


  def get_reward_from_features(self, feature_tensor):
    return self.reward_predictor(feature_tensor)

  def training_step(self, batch, batch_nb):
    batch.to_device(self.device)
    intermediate_feature_tuple = self.get_backboone_features(
        batch)
    pp_tensor, np_tensor, nn_tensor, pn_tensor = intermediate_feature_tuple

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
    batch.to_device(self.device)
    batch.to_device(self.device)
    intermediate_feature_tuple = self.get_backboone_features(
        batch)
    pp_tensor, np_tensor, nn_tensor, pn_tensor = intermediate_feature_tuple

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
    batch.to_device(self.device)
    batch.to_device(self.device)
    intermediate_feature_tuple = self.get_backboone_features(
        batch)
    pp_tensor, np_tensor, nn_tensor, pn_tensor = intermediate_feature_tuple

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
    if self.data_params.NO_LR_SCHEDULER:
        return [optimizer]
    num_minibatch_steps = (
        self.data_params.NUM_TRAIN_SAMPLES)/(self.data_params.BATCH_SIZE)
    max_epochs = self.data_params.MAX_EPOCHS
    warmup = self.data_params.WARMUP
    t_total = self.data_params.TOTAL_STEPS
    num_cycles = self.data_params.MAX_CYCLES
    lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer, warmup, t_total, num_cycles=num_cycles)
    return [optimizer], [{'scheduler':lr_scheduler,'interval':self.data_params.LR_SCHEDULER_FREQUENCY}]

