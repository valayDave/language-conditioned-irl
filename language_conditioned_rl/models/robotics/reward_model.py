import pytorch_lightning as pl 
import torch
import torch.nn as nn
from typing import List, Tuple
from ...dataloaders.channel import ContrastiveGenerator,ChannelData
from ..transformer import \
    OmniTransformerCoreConfig,\
    OmniChannelTransformer,\
    ChannelConfiguration,\
    VideoPatchEmbedding,\
    ImagePatchEmbedding,\
    ChannelEmbeddingDiscrete,\
    TextEmbeddingsPretrain,\
    DEFAULT_OMNI_TRANSFORMER_PARAMS

from ..reward_model import DataAndOptimizerConf,RewardHeadWithOnlyBackbone
from ...dataloaders.robotics.dataset import CONTINOUS_VALUE_DIMS

DISCRETE_EMBEDDING_SIZE = 128
PATCH_SIZE=64
PATCH_EMBEDDING_DIMS = 128
IMAGE_SIZE = (256,256)


USE_CHANNELS = [
  # 'joint_gripper_velocity',
  # 'joint_robot_velocity',
  'joint_robot_position',
  'image_sequence',
  'text',
  'joint_gripper',
]
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


def make_model(
  use_channels = USE_CHANNELS,\
  CORE_TRANSFORMER_PARAMETERS = DEFAULT_OMNI_TRANSFORMER_PARAMS,\
  dataparams:DataAndOptimizerConf = DataAndOptimizerConf(),\
  channel_embed_size:int = DISCRETE_EMBEDDING_SIZE,\
  patch_embed_size:int = PATCH_EMBEDDING_DIMS,\
  patch_size:int=PATCH_SIZE,\
  image_size:Tuple[int,int] = IMAGE_SIZE,\
):
  video_channel = ChannelConfiguration(
    name='image_sequence',
    channel_type='continous',
    input_dim=None,
    embedding_layer = VideoPatchEmbedding(image_size[0],patch_size,embedding_size=patch_embed_size),
    use_position_embed = True,
    embedding_size = channel_embed_size,
    no_embedding=False,
  )
  image_channel = ChannelConfiguration(
    name='image',
    channel_type='continous',
    input_dim=None,
    embedding_layer = ImagePatchEmbedding(image_size[0],patch_size,embedding_size=patch_embed_size),
    use_position_embed = True,
    embedding_size = channel_embed_size,
    no_embedding=False,
  )
  joint_gripper_channel = ChannelConfiguration(
      name='joint_gripper',
      channel_type='discrete',
      input_dim=channel_embed_size,
      embedding_layer = ChannelEmbeddingDiscrete(4,embedding_size=channel_embed_size),
      use_position_embed = True,
      embedding_size = channel_embed_size,
  )

  joint_gripper_velocity_channel = ChannelConfiguration(
      name='joint_gripper_velocity',
      channel_type='continous',
      input_dim=CONTINOUS_VALUE_DIMS['joint_gripper_velocity'],
      embedding_layer =None,
      no_embedding=True,
      use_position_embed = True,
  )
  joint_robot_position_channel = ChannelConfiguration(
      name='joint_robot_position',
      channel_type='continous',
      input_dim=CONTINOUS_VALUE_DIMS['joint_robot_position'],
      embedding_layer =None,
      no_embedding=True,
      use_position_embed = True,
  )
  joint_robot_velocity_channel = ChannelConfiguration(
      name='joint_robot_velocity',
      channel_type='continous',
      input_dim=CONTINOUS_VALUE_DIMS['joint_robot_velocity'],
      embedding_layer =None,
      no_embedding=True,
      use_position_embed = True,
  )
  tcp_angular_veloctiy_channel = ChannelConfiguration(
      name='tcp_angular_veloctiy',
      channel_type='continous',
      input_dim=CONTINOUS_VALUE_DIMS['tcp_angular_veloctiy'],
      embedding_layer =None,
      no_embedding=True,
      use_position_embed = True,
  )
  tcp_linear_velocity_channel = ChannelConfiguration(
      name='tcp_linear_velocity',
      channel_type='continous',
      input_dim=CONTINOUS_VALUE_DIMS['tcp_linear_velocity'],
      embedding_layer =None,
      no_embedding=True,
      use_position_embed = True,
  )
  tcp_target_position_channel = ChannelConfiguration(
      name='tcp_target_position',
      channel_type='continous',
      input_dim=CONTINOUS_VALUE_DIMS['tcp_target_position'],
      embedding_layer =None,
      no_embedding=True,
      use_position_embed = True,
  )

  tcp_orientation_channel = ChannelConfiguration(
      name='tcp_orientation',
      channel_type='continous',
      input_dim=CONTINOUS_VALUE_DIMS['tcp_orientation'],
      embedding_layer =None,
      no_embedding=True,
      use_position_embed = True,
  )

  tcp_position_channel = ChannelConfiguration(
      name='tcp_position',
      channel_type='continous',
      input_dim=CONTINOUS_VALUE_DIMS['tcp_linear_velocity'],
      embedding_layer =None,
      no_embedding=True,
      use_position_embed = True,
  )

  tcp_target_orientation_channel = ChannelConfiguration(
      name='tcp_target_orientation',
      channel_type='continous',
      input_dim=CONTINOUS_VALUE_DIMS['tcp_target_orientation'],
      embedding_layer =None,
      no_embedding=True,
      use_position_embed = True,
  )

  txt_emb_layer = TextEmbeddingsPretrain(is_learnable=False)
  text_channel = ChannelConfiguration(
      name='text',
      channel_type='discrete',
      input_dim=None,
      embedding_size=txt_emb_layer.embeddings.embedding_dim,
      no_embedding=False,
      embedding_layer = txt_emb_layer,
      use_position_embed=True,
  )
  # USE_CHANNEL_CONFIG 
  FILTER_CHANNELS = [
    joint_gripper_channel,
    joint_robot_position_channel,
    joint_robot_velocity_channel,
    video_channel
  ]
  for f in FILTER_CHANNELS:
    f.route_to_everything = False
    f.restricted_channels = ['text']
 
  channel_configurations = [
      video_channel,
      text_channel,
      joint_gripper_channel,
      joint_gripper_velocity_channel,
      joint_robot_position_channel,
      joint_robot_velocity_channel,
      tcp_angular_veloctiy_channel,
      tcp_linear_velocity_channel,
      tcp_target_position_channel,
      tcp_orientation_channel,
      tcp_position_channel,
      tcp_target_orientation_channel,
      image_channel
  ]
  channel_configurations = [config for config in channel_configurations if config.name in use_channels]
  transformer_config = OmniTransformerCoreConfig(**CORE_TRANSFORMER_PARAMETERS,channel_configurations=channel_configurations)
  trans = LGRRoboRewardLearner(transformer_config,data_params=dataparams)
  return trans