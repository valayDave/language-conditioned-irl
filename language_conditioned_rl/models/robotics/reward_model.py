import pytorch_lightning as pl 
import json
import torch
import torch.nn as nn
from typing import List, Tuple
from ...dataloaders.channel import ContrastiveGenerator,ChannelData
from ..optimizer import PCGrad
from ..transformer import \
    OmniTransformerCoreConfig,\
    OmniChannelTransformer,\
    ChannelConfiguration,\
    VideoPatchEmbedding,\
    ImagePatchEmbedding,\
    ChannelEmbeddingDiscrete,\
    TextEmbeddingsPretrain,\
    DEFAULT_OMNI_TRANSFORMER_PARAMS,\
    PRETRAINED_MODEL

from ..reward_model import DataAndOptimizerConf,RewardHeadWithOnlyBackbone
from ...dataloaders.robotics.dataset import CONTINOUS_VALUE_DIMS, MAX_TEXT_LENGTH, MAX_TRAJ_LEN, MAX_VIDEO_FRAMES
from ...dataloaders.robotics.utils import RoboDataUtils

DISCRETE_EMBEDDING_SIZE = 128
PATCH_SIZE=64
PATCH_EMBEDDING_DIMS = 128
IMAGE_SIZE = (256,256)


def NEPTUNE_JSON_FIXER(json_str):
    return json.loads(json_str.replace("'",'"').replace('None','null').replace('True','true').replace('False','false'))



USE_CHANNELS = [
  # 'joint_gripper_velocity',
  # 'joint_robot_velocity',
  'joint_robot_position',
  'image_sequence',
  'text',
  'joint_gripper',
]

GLOBAL_EMBEDDINGS = None

GLOBAL_RESNET_BACKBONE = None

def set_global_embed(embed_layer):
  global GLOBAL_EMBEDDINGS
  GLOBAL_EMBEDDINGS = embed_layer

def make_embedding(tensor):
  global GLOBAL_EMBEDDINGS
  if GLOBAL_EMBEDDINGS is None:
    raise Exception("Global Embeddings Don't Exist!")

  if GLOBAL_EMBEDDINGS.weight.device != tensor.device:
      z = tensor.to(GLOBAL_EMBEDDINGS.weight.device)
      return GLOBAL_EMBEDDINGS(z).to(tensor.device)
  else:
      return GLOBAL_EMBEDDINGS(tensor)


class DetachedTextEmbeddingsPretrain(nn.Module):
  
  def __init__(self,pretrain_model=PRETRAINED_MODEL):
      super().__init__()
      from transformers import AutoModel
      bert_model = AutoModel.from_pretrained(pretrain_model)
      bert_emb = bert_model.embeddings.word_embeddings
      text_embedding_dim = bert_emb.embedding_dim
      num_emb = bert_emb.num_embeddings
      # self.is_learnable=is_learnable
      embeddings = nn.Embedding(
          num_embeddings=num_emb, embedding_dim=text_embedding_dim)
      embeddings.load_state_dict(bert_emb.state_dict())
      embeddings.weight.requires_grad = False
      set_global_embed(embeddings)
      
    
  def forward(self, channel_seq):
    return make_embedding(channel_seq)
      

class VisualBackbone(nn.Module):
    r"""
    THANK YOU : https://github.com/kdexd/virtex/blob/master/virtex/modules/visual_backbones.py
    Base class for all visual backbones. All child classes can simply inherit
    from :class:`~torch.nn.Module`, however this is kept here for uniform
    type annotations.
    """

    def __init__(self, visual_feature_size: int):
        super().__init__()
        self.visual_feature_size = visual_feature_size


class TorchvisionVisualBackbone(VisualBackbone):
    r"""
    A visual backbone from `Torchvision model zoo
    <https://pytorch.org/docs/stable/torchvision/models.html>`_. Any model can
    be specified using corresponding method name from the model zoo.
    Parameters
    ----------
    name: str, optional (default = "resnet50")
        Name of the model from Torchvision model zoo.
    visual_feature_size: int, optional (default = 2048)
        Size of the channel dimension of output visual features from forward pass.
    pretrained: bool, optional (default = False)
        Whether to load ImageNet pretrained weights from Torchvision.
    frozen: float, optional (default = False)
        Whether to keep all weights frozen during training.
    """

    def __init__(
        self,
        visual_feature_size: int = 2048,
        pretrained: bool = False,
        frozen: bool = False,
    ):
        super().__init__(visual_feature_size)
        from torchvision.models import resnet18
        self.cnn = resnet18(
            pretrained, zero_init_residual=True
        )
        # Do nothing after the final residual stage.
        self.cnn.fc = nn.Identity()

        # Freeze all weights if specified.
        if frozen:
            for param in self.cnn.parameters():
                param.requires_grad = False
            self.cnn.eval()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        r"""
        Compute visual features for a batch of input images.
        Parameters
        ----------
        image: torch.Tensor
            Batch of input images. A tensor of shape
            ``(batch_size, 3, height, width)``.
        Returns
        -------
        torch.Tensor
            A tensor of shape ``(batch_size, channels, height, width)``, for
            example it will be ``(batch_size, 2048, 7, 7)`` for ResNet-50.
        """

        for idx, (name, layer) in enumerate(self.cnn.named_children()):
            out = layer(image) if idx == 0 else layer(out)

            # These are the spatial features we need.
            if name == "layer4":
                # shape: (batch_size, channels, height, width)
                return out


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

  def on_pretrain_routine_start(self) -> None:
    # Put Main Embeddings On CPU
    if getattr(self.model.text__embedding,'embedding',None) is not None:
      self.model.text__embedding.embeddings.to(torch.device('cpu'))

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
class LGRMulitTaskRoboRewardLearner(pl.LightningModule):
  def __init__(self,
               config:OmniTransformerCoreConfig,
               pc_grad_loss:bool=False,
               data_params: DataAndOptimizerConf = DataAndOptimizerConf()
               ):
    super().__init__()
    self.model = OmniChannelTransformer(config)
    self.sfmx = nn.Softmax(dim=1)
    self.pc_grad_loss = pc_grad_loss
    if pc_grad_loss:
      self.automatic_optimization = False
    self.model.final_layer_dims 
    self.log_sigmoid = nn.LogSigmoid()
    # Output Passed to Zero or one on sigmoid activation
    self.reward_predictor = RewardHeadWithOnlyBackbone(
        self.model.final_layer_dims , self.model.config.transformer_embedding_size)
    self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
    self.data_params = data_params

  def on_pretrain_routine_start(self) -> None:
    # Put Main Embeddings On CPU
    if getattr(self.model.text__embedding,'embedding',None) is not None:
      self.model.text__embedding.embeddings.to(torch.device('cpu'))

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
  
  def manual_backward(self, losses, optimizer):
    optim_pc = PCGrad(optimizer)
    optim_pc.pc_backward(losses)
    optim_pc.step()
    return torch.mean(torch.stack([l.detach() for l in losses]))


  def get_reward_from_features(self, feature_tensor):
    return self.reward_predictor(feature_tensor)

  def per_batch_step(self,batch:ContrastiveGenerator):
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
    return loss


  def training_step(self, batch, batch_nb):
    task_losses = []
    for taskb in batch:
      task_loss = self.per_batch_step(taskb)
      task_losses.append(task_loss)
    
    if self.pc_grad_loss:
      optimizer = self.optimizers()
      loss = self.manual_backward(task_losses,optimizer)
    else:
      loss = torch.mean(torch.stack(task_losses))

    self.logger.log_metrics({
        'train_loss': loss.detach().cpu().numpy(),
        'epoch': self.current_epoch,
    })
    return {'loss': loss}

  def validation_step(self, batch, batch_nb):
    task_losses = []
    for taskb in batch:
      task_loss = self.per_batch_step(taskb)
      task_losses.append(task_loss)
    
    loss = torch.mean(torch.stack(task_losses))

    self.logger.log_metrics({
        'val_loss': loss.detach().cpu().numpy(),
        'epoch': self.current_epoch,
    })
    return {'loss': loss, 'val_loss': loss.detach().cpu()}

  def test_step(self, batch, batch_nb):
    task_losses = []
    for taskb in batch:
      task_loss = self.per_batch_step(taskb)
      task_losses.append(task_loss)
    
    loss = torch.mean(torch.stack(task_losses))

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

class RoboRewardFnMixin(object):

    def __init__(self,
                 max_video_len = MAX_VIDEO_FRAMES,
                 max_traj_length=MAX_TRAJ_LEN,
                 max_text_len=MAX_TEXT_LENGTH,
                 experiment_name=None,
                 loaded_checkpoint=None,
                 pretrained_model=PRETRAINED_MODEL):

        from transformers import AutoTokenizer
        self.data_utils = RoboDataUtils(
          tokenizer = AutoTokenizer.from_pretrained(pretrained_model),
          image_resize_dims=IMAGE_SIZE,
          max_traj_len=max_traj_length,
          max_txt_len=max_text_len
        )
        self.max_traj_length=max_traj_length
        self.max_txt_len=max_text_len
        self.experiment_name = experiment_name
        self.loaded_checkpoint = loaded_checkpoint
        self.max_video_len = max_video_len
        self.use_channels = [c.name for c in self.model.config.channel_configurations]
    
    def get_rewards(self,text,states):
      assert type(text) == str
      assert type(states) == dict
      assert 'image_sequence' in states
      assert 'trajectory' in states
      
      text_ids, txt_msk = self.data_utils.encode_sentence(text)
      channel_data_objects = [
        ChannelData(
          sequence=text_ids.unsqueeze(0), mask=txt_msk.unsqueeze(0),name='text'
        )
      ]
      if 'image_sequence' in self.use_channels:
        channel_data_objects.append(
          ChannelData(
            name='image_sequence',
            sequence= self.data_utils.make_tensor_from_video(
              states['image_sequence'],max_frames=self.max_video_len
            ).unsqueeze(0)
          )  
        )
      channel_data_objects.extend(self._make_trajectory(states))      
      return self._get_rewards(channel_data_objects)
    
    def _get_rewards(self,channel_data:List[ChannelData]):
      with torch.no_grad():
        features,  _ = self(channel_data)
        return self.get_reward_from_features(features)

    def _make_trajectory(self,states):
      trajectory_states = list(set(self.use_channels) - set(['text','image_sequence']))
      assert set(trajectory_states).issubset(set(states['trajectory'][0].keys())),\
         f"{set(trajectory_states)} Not a Subset of {set(states['trajectory'][0].keys())}"

      if len(trajectory_states)  == 0:
        return []
      
      trajectory_dict = self.data_utils.create_trajectory(
        states['trajectory'],use_channels=self.use_channels
      )
      return [ChannelData(sequence=trajectory_dict[traj_chan].unsqueeze(0),name=traj_chan) for traj_chan in trajectory_dict]

      





class RobotLearningRewardFunction(RoboRewardFnMixin, LGRRoboRewardLearner):
    def __init__(self,\
                *args,\
                max_video_len = MAX_VIDEO_FRAMES,\
                max_traj_length=MAX_TRAJ_LEN,\
                max_text_len=MAX_TEXT_LENGTH,\
                experiment_name=None,\
                loaded_checkpoint=None,\
                pretrained_model=PRETRAINED_MODEL, **kwargs):
        LGRRoboRewardLearner.__init__(self, *args, **kwargs)
        RoboRewardFnMixin.__init__(self,
                                  max_video_len = max_video_len,\
                                  max_traj_length=max_traj_length,\
                                  max_text_len=max_text_len,\
                                  experiment_name=experiment_name,\
                                  loaded_checkpoint=loaded_checkpoint,\
                                  pretrained_model=pretrained_model)



def make_model(
  use_channels = USE_CHANNELS,\
  CORE_TRANSFORMER_PARAMETERS = DEFAULT_OMNI_TRANSFORMER_PARAMS,\
  dataparams:DataAndOptimizerConf = DataAndOptimizerConf(),\
  channel_embed_size:int = DISCRETE_EMBEDDING_SIZE,\
  patch_embed_size:int = PATCH_EMBEDDING_DIMS,\
  is_multitask:bool=False,\
  is_pc_grad:bool=False,\
  patch_size:int=PATCH_SIZE,\
  image_size:Tuple[int,int] = IMAGE_SIZE,\
  max_video_len = MAX_VIDEO_FRAMES,\
  max_traj_length=MAX_TRAJ_LEN,\
  detached_text_embed:bool=False,\
  is_inference:bool=False,\
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
      input_dim=CONTINOUS_VALUE_DIMS['tcp_position'],
      embedding_layer =None,
      no_embedding=True,
      use_position_embed = True,
  )
  final_coord_channel_size = 1
  final_target_coordinates_channel = ChannelConfiguration(
      name='final_target_coordinates',
      channel_type='continous',
      input_dim= final_coord_channel_size,
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
      embedding_layer = txt_emb_layer if not detached_text_embed else DetachedTextEmbeddingsPretrain(),
      use_position_embed=True,
  )
  joint_combined_vector = ChannelConfiguration(
      name='joint_combined_vector',
      channel_type='continous',
      input_dim=CONTINOUS_VALUE_DIMS['joint_robot_position']+1,# For robot gripper state
      embedding_layer =None,
      no_embedding=True,
      use_position_embed = True,
  )
  # USE_CHANNEL_CONFIG 
  FILTER_CHANNELS = [
    joint_gripper_channel,
    joint_robot_position_channel,
    joint_robot_velocity_channel,
    video_channel,
    tcp_position_channel,
    joint_combined_vector,
    final_target_coordinates_channel
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
      image_channel,
      joint_combined_vector,
      final_target_coordinates_channel
  ]
  channel_configurations = [config for config in channel_configurations if config.name in use_channels]
  transformer_config = OmniTransformerCoreConfig(**CORE_TRANSFORMER_PARAMETERS,channel_configurations=channel_configurations)
  if is_inference:
    return RobotLearningRewardFunction(transformer_config,
                                      max_video_len = max_video_len,\
                                      max_traj_length = max_traj_length,\
                                      data_params=dataparams)
  elif not is_multitask:
    return LGRRoboRewardLearner(transformer_config,\
                                data_params=dataparams,\
                                )
  else:
    return LGRMulitTaskRoboRewardLearner(
      transformer_config,pc_grad_loss=is_pc_grad,data_params=dataparams
    )
      



def get_checkpoint(project_name='valay/Language-Grounded-Rewards-Robo',
                experiment_name='LRO-89',
                checkpoint_path='checkpoints/last.pt', \
                base_path='model_checkpoints/',
                api_token=None,):
  import neptune
  import os
  import json
  if api_token is None:
      raise Exception("API Token Missing")

  project = neptune.init(project_name,
                          api_token=api_token
                          )
  my_exp = project.get_experiments(id=experiment_name)[0]
  my_exp.download_artifact(checkpoint_path, destination_dir=base_path)
  config = my_exp.get_parameters()
  ckck_name = checkpoint_path.split('/')[1]
  checkpoint = torch.load(os.path.join(
      base_path, ckck_name), map_location=torch.device('cpu'))
  
  if 'channel_configurations' in config:
    config['channel_configurations'] = NEPTUNE_JSON_FIXER(config['channel_configurations'])
  if 'transformer_params' in config:
    config['transformer_params']  = NEPTUNE_JSON_FIXER(config['transformer_params'])
  if 'data_params' in config:
    config['data_params']  = NEPTUNE_JSON_FIXER(config['data_params'])

  return checkpoint,config


def make_model_from_checkpoint(\
                project_name='valay/Language-Grounded-Rewards-Robo',\
                experiment_name='LRO-89',\
                checkpoint_path='checkpoints/last.pt', \
                base_path='model_checkpoints/',\
                max_video_len = MAX_VIDEO_FRAMES,\
                max_traj_length=MAX_TRAJ_LEN,\
                api_token=None):
  
  checkpoint,config = get_checkpoint(
                  project_name = project_name,
                  experiment_name = experiment_name,
                  checkpoint_path = checkpoint_path,
                  base_path = base_path,
                  api_token = api_token,
                )
  use_channels = [c['name'] for c in config['channel_configurations']]
  
  trans = make_model(
    use_channels=use_channels,
    CORE_TRANSFORMER_PARAMETERS=config['transformer_params'],
    channel_embed_size=DISCRETE_EMBEDDING_SIZE,
    patch_embed_size=PATCH_EMBEDDING_DIMS,
    patch_size=PATCH_SIZE,
    is_inference=True,
    max_video_len =max_video_len,
    max_traj_length =max_traj_length,
  )
  missing_keys, unexpected_keys = trans.load_state_dict(checkpoint['state_dict'])
  trans.experiment_name = experiment_name
  trans.loaded_checkpoint = checkpoint_path
  return trans