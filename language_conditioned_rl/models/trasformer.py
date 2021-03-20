
from dataclasses import dataclass, field
import torch.nn as nn
from typing import List, Tuple
import math
import torch
import einops
import abc
import itertools
import torch.nn.functional as F
from .cross_modal_attention import MultiModalAttentionBlock
from .attention import Block
from .embeddings import ActionEmbedding, SinusoidalPositionalEmbedding
from einops.layers.torch import Rearrange


PRETRAINED_MODEL = 'bert-base-uncased'


class Transformer(nn.Module):
    def __init__(self,
                 num_layers=4,
                 embedding_size=256,
                 layer_norm_epsilon=0.00001,
                 scale=0.2,
                 resid_pdrop=0.1,
                 attn_pdrop=0.1,
                 num_attention_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(Block(
                embedding_size=embedding_size,
                layer_norm_epsilon=layer_norm_epsilon,
                scale=scale,
                resid_pdrop=resid_pdrop,
                attn_pdrop=attn_pdrop,
                num_attention_heads=num_attention_heads))

    def forward(self, x, mask=None):
        hidden = x
        for attention_block in self.layers:
            hidden = attention_block(hidden)
        return hidden


class VanillaTransformer(nn.Module):
    '''
    Contains Sinusoidal Embedding for sequences as part of the Framework. 
    '''

    def __init__(self,
                 num_layers=4,
                 embed_dropout=0.1,
                 embedding_size=256,
                 layer_norm_epsilon=0.00001,
                 scale=0.2,
                 use_pos_embed=True,
                 resid_pdrop=0.1,
                 attn_pdrop=0.1,
                 num_attention_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.embed_dropout = embed_dropout
        self.embed_scale = math.sqrt(embedding_size)
        if use_pos_embed:
            self.embed_positions = SinusoidalPositionalEmbedding(
                embedding_size)

        for _ in range(num_layers):
            self.layers.append(MultiModalAttentionBlock(
                embedding_size=embedding_size,
                layer_norm_epsilon=layer_norm_epsilon,
                scale=scale,
                resid_pdrop=resid_pdrop,
                attn_pdrop=attn_pdrop,
                num_attention_heads=num_attention_heads))

    def forward(self, x, mask=None,return_attentions=False):
        # Add positional embedding
        x = self.embed_scale * x  # (b,len,d)
        if self.embed_positions is not None:
            x += self.embed_positions(x[:, :, 0])
        x = F.dropout(x, p=self.embed_dropout, training=self.training)

        hidden = x
        if not return_attentions:
            for attention_block in self.layers:
                hidden = attention_block(hidden,hidden,mask=mask)
            return hidden
        else:
            attn_vals= []
            for attention_block in self.layers:
                hidden,attn = attention_block(hidden,hidden,mask=mask)
                attn_vals.append(attn)
            return hidden, torch.stack(attn_vals)


class MultiModalTransformer(nn.Module):
    def __init__(self,
                 num_layers=4,
                 embed_dropout=0.1,
                 embedding_size=256,
                 layer_norm_epsilon=0.00001,
                 scale=0.2,
                 use_pos_embed=True,
                 resid_pdrop=0.1,
                 attn_pdrop=0.1,
                 num_attention_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.embed_dropout = embed_dropout
        self.embed_scale = math.sqrt(embedding_size)
        if use_pos_embed:
            self.embed_positions = SinusoidalPositionalEmbedding(
                embedding_size)

        for _ in range(num_layers):

            self.layers.append(MultiModalAttentionBlock(
                embedding_size=embedding_size,
                layer_norm_epsilon=layer_norm_epsilon,
                scale=scale,
                resid_pdrop=resid_pdrop,
                attn_pdrop=attn_pdrop,
                num_attention_heads=num_attention_heads))

    def forward(self, seq_x, seq_y, mask=None,mask_y=None, return_attentions=False):
        # Add positional embedding
        seq_x = self.embed_scale * seq_x
        if self.embed_positions is not None:
            seq_x += self.embed_positions(seq_x[:, :, 0])
        seq_x = F.dropout(seq_x, p=self.embed_dropout, training=self.training)

        seq_y = self.embed_scale * seq_y
        if self.embed_positions is not None:
            seq_y += self.embed_positions(seq_y[:, :, 0])
        seq_y = F.dropout(seq_y, p=self.embed_dropout, training=self.training)

        if not return_attentions:
            hidden = seq_x
            for attention_block in self.layers:
                hidden = attention_block(hidden, seq_y, mask=mask,mask_y=mask_y)
            return hidden
        else:
            hidden = seq_x
            attn_vals = []
            for attention_block in self.layers:
                hidden, attn = attention_block(
                    hidden, seq_y, mask=mask,mask_y=mask_y, return_attentions=return_attentions)
                attn_vals.append(attn)
            return hidden, torch.stack(attn_vals)


class CrossModalBertEmbedTranformer(nn.Module):
    '''
    Creates Multi-Transformers under ["state","action","text"] modalities. 

    Use Glove Embeddings with Explicit Glove Embedding Layer. 


    '''
    join_str = "__"

    modalities = ["state", "action", "text"]

    cross_mod_trans_common_name = '_transformer'

    mod_trans_common_name = '_collate_transformer'

    def __init__(self,
                 num_layers=3,
                 dropout=0.1,
                 num_heads=4,
                 scale=0.2,
                 pretrained_model=PRETRAINED_MODEL,
                 num_actions=3,
                 embd_pdrop=0.1,
                 common_conv_dim=128,
                 embedding_size=256,
                 layer_norm_epsilon=0.00001,
                 resid_pdrop=0.1,
                 attn_pdrop=0.1,
                 state_dims=2,
                 return_hidden_tensors=False,
                 action_type='discrete'):

        super().__init__()

        # No Text Transoformer Backend Using Glove Based Embeddings instead.

        # Create Different Transformers for the Modalities of Information
        self.action_embedding = ActionEmbedding(
            num_actions=num_actions+1, action_type=action_type, hidden_dims=embedding_size)

        # Create Text Embedding
        from transformers import AutoModel
        bert_model = AutoModel.from_pretrained(PRETRAINED_MODEL)
        bert_emb = bert_model.embeddings.word_embeddings
        text_embedding_dim = bert_emb.embedding_dim
        num_emb = bert_emb.num_embeddings
        self.text_embeddings = nn.Embedding(
            num_embeddings=num_emb, embedding_dim=text_embedding_dim)
        # bert_model.embeddings.word_embeddings
        self.text_embeddings.load_state_dict(bert_emb.state_dict())
        # self.text_embeddings,_ , text_embedding_dim = self.create_emb_layer(glove_weights)
        self.text_embeddings.weight.requires_grad = False

        # 1 D Convs for Modalities
        self.action_conv = nn.Conv1d(
            embedding_size, common_conv_dim, kernel_size=1, padding=0, bias=False)
        self.state_conv = nn.Conv1d(
            state_dims, common_conv_dim, kernel_size=1, padding=0, bias=False)
        self.text_conv = nn.Conv1d(
            text_embedding_dim, common_conv_dim, kernel_size=1, padding=0, bias=False)

        self.action_cls_token = nn.Parameter(
            torch.randn(1, 1, common_conv_dim))
        self.state_cls_token = nn.Parameter(torch.randn(1, 1, common_conv_dim))
        self.text_cls_token = nn.Parameter(torch.randn(1, 1, common_conv_dim))
        self.to_cls = nn.Identity()
        self.final_layer = nn.Sequential(
            nn.Linear(common_conv_dim*3*2, common_conv_dim*3*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(common_conv_dim*3*2, common_conv_dim*3*2)
        )

        # self.multimodal_trans_map = self.get_joint_modality_names()
        self.create_multi_modal_transformers(common_conv_dim,
                                             dropout=dropout,
                                             num_heads=num_heads,
                                             scale=scale,
                                             embd_pdrop=embd_pdrop,
                                             per_module_depth=num_layers,
                                             layer_norm_epsilon=layer_norm_epsilon,
                                             resid_pdrop=resid_pdrop,
                                             attn_pdrop=attn_pdrop)

    @staticmethod
    def create_emb_layer(weights_matrix, non_trainable=True):
        num_embeddings, embedding_dim = weights_matrix.shape
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.weight.data = torch.Tensor(weights_matrix)
        if non_trainable:
            emb_layer.weight.requires_grad = False
        return emb_layer, num_embeddings, embedding_dim

    # should be called only once!
    def create_multi_modal_transformers(self, hidden_size,
                                        per_module_depth=3,
                                        layer_norm_epsilon=0.00001,
                                        resid_pdrop=0.1,
                                        attn_pdrop=0.1,
                                        dropout=0.1,
                                        num_heads=4,
                                        embd_pdrop=0.1,
                                        scale=0.2):
        self.multimodal_trans_map = [
            m+self.cross_mod_trans_common_name for m in self.get_joint_modality_names()]
        for name in self.multimodal_trans_map:
            # Create cross model transformer of names
            cross_modal_transformer = MultiModalTransformer(num_layers=per_module_depth,
                                                            embedding_size=hidden_size,
                                                            layer_norm_epsilon=layer_norm_epsilon,
                                                            scale=scale,
                                                            embed_dropout=embd_pdrop,
                                                            resid_pdrop=resid_pdrop,
                                                            attn_pdrop=attn_pdrop,
                                                            num_attention_heads=num_heads)
            setattr(self, name, cross_modal_transformer)

        self.modality_map = [
            m + self.mod_trans_common_name for m in self.modalities]
        for mod in self.modality_map:
            # Create tranformer that runs on concatenation of cross modalities of its own and joints them.
            modal_trans = VanillaTransformer(num_layers=per_module_depth,
                                             embedding_size=2*hidden_size,
                                             layer_norm_epsilon=layer_norm_epsilon,
                                             scale=scale,
                                             embed_dropout=embd_pdrop,
                                             resid_pdrop=resid_pdrop,
                                             attn_pdrop=attn_pdrop,
                                             num_attention_heads=num_heads)
            setattr(self, mod, modal_trans)
        print(
            f"Created The Multi-Modal Transformers For {','.join(self.modality_map)} {','.join(self.multimodal_trans_map)}")

    def get_joint_modality_names(self):
        strs = []
        JOIN_STR = self.join_str
        for x in itertools.product(self.modalities, self.modalities):
            if x[0] == x[1]:
                continue
            strs.append(x[0]+JOIN_STR+x[1])
        return strs

    def apply_dim_reduction_convolutions(self, st, at, tt):
        '''
        expects: (b,dim,seq_len)
        return (b,seq_len,out_dim)
        '''
        return (
            self.state_conv(st).permute(0, 2, 1),
            self.action_conv(at).permute(0, 2, 1),
            self.text_conv(tt).permute(0, 2, 1)
        )

    def extract_cross_modal_tensors(self, modality, modality_tensor, cross_modal_tensors, modality_mask=None, return_attentions=False):
        '''
        modality : str :["state","action","text"]
        modality_tensor : torch.Tensor (B,x,d)
        cross_modal_tensors : 
        (
            (modality,torch.Tensor (B,p,d)),(modality,torch.Tensor (B,v,d))
        ) : tuple of tuples. Each contains modalitystring and tensor

        Runs: 
        '''
        cross_mod_tensors = []
        attn_map = {}
        for cross_mod, cm_tensor in cross_modal_tensors:
            cross_mod_trans_name = modality + self.join_str + \
                cross_mod + self.cross_mod_trans_common_name
            transformer = getattr(self, cross_mod_trans_name)
            if not return_attentions:
                cross_mod_tensors.append(transformer(
                    modality_tensor, cm_tensor, mask=modality_mask))
            else:
                opx, attns = transformer(
                    modality_tensor, cm_tensor, mask=modality_mask, return_attentions=return_attentions)
                cross_mod_tensors.append(opx)
                attn_map[cross_mod_trans_name] = attns

        cross_modal_tensors = torch.cat(cross_mod_tensors, dim=2)
        modality_trans_name = modality + self.mod_trans_common_name
        mod_transformer = getattr(self, modality_trans_name)
        if not return_attentions:
            return mod_transformer(cross_modal_tensors)
        else:
            return mod_transformer(cross_modal_tensors), attn_map

    def forward(self, state_tensor, action_tensor, text_tensor, text_mask=None, act_mask=None, st_mask=None, return_attentions=False):
        '''
        state_tensor : (B,L,ds)
        action_tensor : (B,L,da)
        text_tensor: (B,T,dt)
        L: trajectory_length
        S: Length of the Text. 

        Returns Projection : (b,common_dim*6)
        '''
        # Convert Action/State to Embedding
        action_tensor = self.action_embedding(action_tensor)
        text_tensor = self.text_embeddings(text_tensor)

        # Transpose lenght,dim to dim,length for convolutions
        state_tensor = state_tensor.transpose(1, 2)
        action_tensor = action_tensor.transpose(1, 2)
        text_tensor = text_tensor.transpose(1, 2)

        # print(f"Shape PRE-CONV state_tensor {state_tensor.shape} action_tensor {action_tensor.shape} text_tensor {text_tensor.shape}")
        # Apply Dimensionality Reduction to the information. :
        state_tensor, action_tensor, text_tensor = self.apply_dim_reduction_convolutions(
            state_tensor, action_tensor, text_tensor)

        # Prepend CLS tokens and Finally extract thoose instead of the last token.
        b, n, _ = state_tensor.size()
        action_cls_token = einops.repeat(
            self.action_cls_token, '() n d -> b n d', b=b)
        state_cls_token = einops.repeat(
            self.state_cls_token, '() n d -> b n d', b=b)
        text_cls_token = einops.repeat(
            self.text_cls_token, '() n d -> b n d', b=b)

        state_tensor = torch.cat((state_cls_token, state_tensor), dim=1)
        action_tensor = torch.cat((action_cls_token, action_tensor), dim=1)
        text_tensor = torch.cat((text_cls_token, text_tensor), dim=1)
        # adding Extra one for cls tokens that get prepended the tensors
        if text_mask != None:
            text_mask = torch.cat((torch.ones(b).unsqueeze(
                1).to(state_tensor.device), text_mask), dim=1)
        if act_mask != None:
            act_mask = torch.cat((torch.ones(b).unsqueeze(
                1).to(state_tensor.device), act_mask), dim=1)
        if st_mask != None:
            st_mask = torch.cat((torch.ones(b).unsqueeze(
                1).to(state_tensor.device), st_mask), dim=1)

        # Apply cross modality Transformer.

        # Create Cross modality tuples for input to cross_mod tensors.
        st_at_tup = (('state', state_tensor), ('action', action_tensor))
        st_tt_tup = (('state', state_tensor), ('text', text_tensor))
        tt_at_tup = (('text', text_tensor), ('action', action_tensor))

        # print(f"Shape PRE-MODEL state_tensor {state_tensor.shape} action_tensor {action_tensor.shape} text_tensor {text_tensor.shape}")
        # input to the cross modal transformers is the modality input and the cross modal inputs.
        if not return_attentions:
            text_j_tensor = self.extract_cross_modal_tensors(
                'text', text_tensor, st_at_tup, modality_mask=text_mask)
            state_j_tensor = self.extract_cross_modal_tensors(
                'state', state_tensor, tt_at_tup, modality_mask=st_mask)
            action_j_tensor = self.extract_cross_modal_tensors(
                'action', action_tensor, st_tt_tup, modality_mask=act_mask)
        else:
            text_j_tensor, text_attn_map = self.extract_cross_modal_tensors(
                'text', text_tensor, st_at_tup, modality_mask=text_mask, return_attentions=return_attentions)
            state_j_tensor, state_attn_map = self.extract_cross_modal_tensors(
                'state', state_tensor, tt_at_tup, modality_mask=st_mask, return_attentions=return_attentions)
            action_j_tensor, action_attn_map = self.extract_cross_modal_tensors(
                'action', action_tensor, st_tt_tup, modality_mask=act_mask, return_attentions=return_attentions)

        # print(f"Shape POST-MODEL state_tensor {state_j_tensor.shape} action_tensor {action_j_tensor.shape} text_tensor {text_j_tensor.shape}")
        # concat Last values of SEQUENCES from CROSS MODAL OUTPUT : .select(1,-1) does select from dim 1 index -1 where dim =1 is the sequence length
        # concat_tensor = torch.cat([text_j_tensor.select(1,-1), state_j_tensor.select(1,-1), action_j_tensor.select(1,-1)], dim=1)
        l_txt = self.to_cls(text_j_tensor[:, 0])
        l_st = self.to_cls(state_j_tensor[:, 0])
        l_at = self.to_cls(action_j_tensor[:, 0])
        concat_tensor = torch.cat((l_txt, l_st, l_at), dim=1)
        # print(f"Shape of concat_tensor {concat_tensor.shape}")

        # A residual block
        concat_tensor_proj = self.final_layer(concat_tensor)
        concat_tensor_proj += concat_tensor
        if not return_attentions:  # return the CLS Token extracted Tensor.
            return concat_tensor_proj
        # (CLS Token extracted Tensor,tuple Attention outputs of each of the seqeuences in the final layers.)
        return (concat_tensor_proj, (text_attn_map, state_attn_map, action_attn_map))


class ChannelEmbeddingDiscrete(nn.Module):
    def __init__(self, num_embeddings, embedding_size=128, is_learnable=False):
        super().__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_size)
        if not is_learnable:
            self.embeddings.weight.requires_grad = False

    def forward(self, channel_seq):
        return self.embeddings(channel_seq)


class ChannelEmbeddingContinous(nn.Module):
    def __init__(self, input_dims, embedding_size=128, is_learnable=True):
        super().__init__()
        self.embeddings = nn.Linear(input_dims, embedding_size)
        if not is_learnable:
            self.embeddings.weight.requires_grad = False

    def forward(self, channel_seq):
        return self.embeddings(channel_seq)


class ImagePatchEmbedding(nn.Module):
    """ImagePatchEmbedding [summary]
    THANK YOU https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_pytorch.py
    """

    def __init__(self, image_size, patch_size, embedding_size=128, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, embedding_size),
        )

    def forward(self, image):
        return self.to_patch_embedding(image)


@dataclass
class ChannelConfiguration:
    """ []
    The is configuration class 
    Configuration Given For each channel based on which 
    this omni channel transformer will create embedding layer. 

    name:
        Name of channel Configuration

    channel_type :  Values can be 'continous' | 'discrete'
        The type of variable. Categorical vs continous. 
        If 'discrete' then embedding dim will be created. 
        if 'continous' then linear layer attached to it. 

    input_dim 
        if channel_type == 'continous' 
            dim of the Individual Item in the sequence of the channel
        if channel_type == 'discrete'
            number of categorical variables.  

    embedding_size
        This is super useful when coming to figure 1d convolutions

    no_embedding: 
        Will not Create/Use Embedding Layer for this channel. 

    embedding_layer:
        Instantiated nn.Module. 

    use_position_embed : 
        will inform weather Position embeddings will be used in 
        any of the transformer layers. 

    """
    name: str = ''
    channel_type: str = 'discrete'
    input_dim: int = None
    embedding_size: int = None
    no_embedding: bool = False
    embedding_layer: nn.Module = None
    use_position_embed: bool = True

    def __post_init__(self):
        if not self.no_embedding and self.embedding_layer is None:
            raise Exception(
                "If `no_embedding` is False, then embedding_layer needs to be provided to map the inputs")

        if self.no_embedding and self.input_dim == None:
            raise Exception(
                "If No Embedding are given then Dimsion of an individual item in input sequence is required")


@dataclass
class OmniTransformerCoreConfig:
    num_layers: int = 3
    dropout: float = 0.1
    num_heads: int = 4
    scale: float = 0.2
    embd_pdrop: float = 0.1
    layer_norm_epsilon: float = 0.00001
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1

    #  if pooling_strategy == 'mean' then mean of all
    pooling_strategy: str = 'cls'  # Can be 'cls' or 'mean'
    # This is the size of the embedding
    # that goes into the transformer
    transformer_embedding_size: int = 256

    # Per Channel Config With Embedding layer Comes Here.
    channel_configurations: List[ChannelConfiguration] = field(
        default_factory=[])

    def to_json(self):
        pass  # todo


@dataclass
class ChannelData:
    mask: torch.Tensor = None
    sequence: torch.Tensor = None
    name: str = None


class OmniChannelTransformer(nn.Module):
    """OmniChannelTransformer 
    If Attention is All you need. 
    Then attention should be applied to data coming from any modality. 
    
    Why Not? 
    This is a transformer which is elastic for N-Sequence input channels
    and computes attention between each and every one of the sequences.

    """

    join_str = "__"

    modalities = []

    cross_modal_transformer_names=[]

    embeddings = {}  # Holds all Embedding layers In this transformer.

    cross_mod_trans_common_name = '_transformer'

    mod_trans_common_name = '_collate_transformer'

    embedding_layer_common_name = '__embedding'

    interm_convs_layer_common_name = '__1dcov'

    cls_tokens_common_name = '__cls_token'

    def __init__(self, config: OmniTransformerCoreConfig):
        super().__init__()
        self.modalities = [c.name for c in config.channel_configurations]
        # $ create embedding layer here
        self.create_embeddings(config.channel_configurations)
        # $ Create 1 D Conv Here.
        self.create_interim_convs(config)
        # $ create class Tokens here.
        self.create_class_tokens(config)
        # $ Pos embedding come via sinusiods and they will inform the configuration
        # $ of the cross modal transformer
        # $ Create Cross channel transformers here.
        self.create_cross_modal_transformer(config)

        self.to_cls = nn.Identity()
        self.pooling_strategy = config.pooling_strategy

        self.config = config

        final_dims = config.transformer_embedding_size * 2 * len(self.modalities)

        self.final_layer = nn.Sequential(
            nn.Linear(final_dims,final_dims),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(final_dims, final_dims)
        )
        self.final_layer_dims = final_dims

    # NAMING FNS COME HERE>
    def get_vanilla_trasformer_layer_name(self, mod: str):
        return mod + self.mod_trans_common_name

    def get_cross_mod_layer_name(self, m1: str, m2: str):
        return m1+self.join_str+m2+self.cross_mod_trans_common_name

    def get_conv_layer_name(self, mod: str):
        return mod + self.interm_convs_layer_common_name

    def get_cls_token_name(self, mod: str):
        return mod + self.cls_tokens_common_name

    def get_embedding_layer_name(self, mod: str):
        return mod+self.embedding_layer_common_name

    def create_class_tokens(self, config: OmniTransformerCoreConfig):
        for c in config.channel_configurations:
            cls_token = nn.Parameter(torch.randn(
                1, 1, config.transformer_embedding_size))
            layer_name = self.get_cls_token_name(c.name)
            setattr(self, layer_name, cls_token)

    def create_cross_modal_transformer(self, config: OmniTransformerCoreConfig):
        confs = config.channel_configurations
        config_combs = list(itertools.product(confs,confs))
        for ch1, ch2 in config_combs:
            if ch1.name == ch2.name:
                continue
            trans_name = self.get_cross_mod_layer_name(ch1.name, ch2.name)
            # This will always add position Embed
            cm_transformer = MultiModalTransformer(
                num_layers=config.num_layers,
                num_attention_heads=config.num_heads,
                scale=config.scale,
                embedding_size=config.transformer_embedding_size,
                embed_dropout=config.embd_pdrop,
                layer_norm_epsilon=config.layer_norm_epsilon,
                resid_pdrop=config.resid_pdrop,
                attn_pdrop=config.attn_pdrop,
                use_pos_embed=ch2.use_position_embed and ch1.use_position_embed
            )
            setattr(self,trans_name, cm_transformer)
            self.cross_modal_transformer_names.append(trans_name)

        for c in config.channel_configurations:
            trans_name = self.get_vanilla_trasformer_layer_name(c.name)
            vnt = VanillaTransformer(
                num_layers=config.num_layers,
                num_attention_heads=config.num_heads,
                embedding_size=2*config.transformer_embedding_size,
                scale=config.scale,
                embed_dropout=config.embd_pdrop,
                layer_norm_epsilon=config.layer_norm_epsilon,
                resid_pdrop=config.resid_pdrop,
                attn_pdrop=config.attn_pdrop,
                use_pos_embed=ch2.use_position_embed and ch1.use_position_embed
            )
            setattr(self,trans_name, vnt)

    def create_interim_convs(self, config: OmniTransformerCoreConfig):
        #  if there is no Embedding in this channel
        #  Use input dims to figure 1dconv for the input sequence.
        #  else use the embedding size of channel and translate it to transformer's embedding size.
        for c in config.channel_configurations:
            if c.no_embedding:
                conv_layer = nn.Conv1d(c.input_dim,
                                       config.transformer_embedding_size,
                                       kernel_size=1,
                                       padding=0, bias=False)
            else:
                conv_layer = nn.Conv1d(c.embedding_size,
                                       config.transformer_embedding_size,
                                       kernel_size=1,
                                       padding=0, bias=False)

            layer_name = c.name+self.interm_convs_layer_common_name

            setattr(self, layer_name, conv_layer)
            # self.interm_convs.append(c.name)

    def create_embeddings(self, channel_configurations: List[ChannelConfiguration]):
        for c in channel_configurations:
            present = False
            if not c.no_embedding:
                present = True
                emb_layer_name = self.get_embedding_layer_name(c.name)
                setattr(self, emb_layer_name, c.embedding_layer)
            self.embeddings[c.name] = present

    def get_channel_embeddings(self, input_channels: List[ChannelData]) -> List[ChannelData]:
        return_channel_data = []
        for c in input_channels:
            if self.embeddings[c.name] == True:
                emb_layer_name = self.get_embedding_layer_name(c.name)
                emblayer = getattr(self, emb_layer_name)
                return_channel_data.append(
                    ChannelData(
                        mask=c.mask,
                        sequence=emblayer(c.sequence),
                        name=c.name
                    )
                )
        return return_channel_data

    def perform_dim_reduction_conv(self,input_channels: List[ChannelData]) -> List[ChannelData]:
        """perform_dim_reduction_conv 
        Perform 1d convolution on the input channels to 
        make dimensions even out for all sequences before we feed it to the transformer.
        """
        return_channel_data = []
        for c in input_channels:
            cnv_layer_name = self.get_conv_layer_name(c.name)
            cnv_layer = getattr(self,cnv_layer_name)
            input_tensor = c.sequence.transpose(1, 2)
            cnv_tensor = cnv_layer(input_tensor)
            return_channel_data.append(
                ChannelData(
                    mask=c.mask,
                    sequence=cnv_tensor.permute(0, 2, 1),
                    name=c.name
                )
            )
        
        return return_channel_data

    def add_cls_tokens(self, input_channels: List[ChannelData]) -> List[ChannelData]:
        b, n, _ = input_channels[0].sequence.size()
        for c in input_channels:
            cls_name = self.get_cls_token_name(c.name)
            cls_token = getattr(self,cls_name)
            prepend_cls_token = einops.repeat(cls_token,'() n d -> b n d',b=b)
            c.sequence = torch.cat((c.sequence,prepend_cls_token),dim=1)
            if c.mask is not None:
                c.mask = torch.cat((torch.ones(b).unsqueeze(1).to(c.sequence.device), c.mask), dim=1)

        return input_channels


    def get_cross_modal_features(self, input_channels: List[ChannelData], return_attentions=False)-> Tuple[List[ChannelData],List[dict]]:
        # For each `input_channel` we need combinations of all other elements 
        # Each elemen 
        all_channel_names = set([c.name for c in input_channels])
        channel_lookup_dict = {
            c.name:c for c in input_channels
        }
        transformed_channel_data = []
        attn_maps = []
        # for each channel find the other channels that it can cross-attend to. 
        for channel_tensors_tuple in itertools.combinations(input_channels,len(input_channels)-1):
            cross_channel_names = set(list(map(lambda x:x.name,channel_tensors_tuple)))
            current_channel_name = all_channel_names - cross_channel_names
            current_channel_name = list(current_channel_name)[0]
            current_channel_object = channel_lookup_dict[current_channel_name]
            if return_attentions:
                # Use the current channel and other channels to find cross channel features. 
                current_channel_features, current_channel_attn_map = self.extract_transformer_features(current_channel_object,channel_tensors_tuple,return_attentions=True)
                attn_maps.append(current_channel_attn_map)
            else:
                current_channel_features = self.extract_transformer_features(current_channel_object,channel_tensors_tuple,return_attentions=False)
            
            transformed_channel_data.append(
                ChannelData(
                    mask = current_channel_object.mask,
                    name = current_channel_object.name,
                    sequence = current_channel_features
                )
            )
            # current_channel_object.sequence = current_channel_features
        return transformed_channel_data,attn_maps
            

    def extract_transformer_features(self,index_modality:ChannelData,cross_modalities:Tuple[ChannelData],return_attentions=False):
        """extract_transformer_features [summary]

        :param index_modality: The Modality for which apply Cross-Channel Attention and run the model
        :type index_modality: ChannelData
        :param cross_modalities: [description]
        :type cross_modalities: Tuple[ChannelData]
        :param return_attentions: [description], defaults to False
        :type return_attentions: bool, optional
        :return: [description]
        :rtype: [type]
        """
        cross_mod_tensors=[]
        attn_map = {}
        for cross_mod_object in cross_modalities:
            cc_trans_name = self.get_cross_mod_layer_name(index_modality.name,cross_mod_object.name)
            cm_transformer = getattr(self,cc_trans_name)
            if not return_attentions:
                cross_mod_tensors.append(cm_transformer(
                    index_modality.sequence,
                    cross_mod_object.sequence,
                    mask=index_modality.mask,
                    mask_y=cross_mod_object.mask,
                    return_attentions=return_attentions
                ))
            else:
                cross_atten_opx, attns_weights = cm_transformer(
                    index_modality.sequence,
                    cross_mod_object.sequence,
                    mask=index_modality.mask,
                    mask_y=cross_mod_object.mask,
                    return_attentions=return_attentions
                )
                cross_mod_tensors.append(cross_atten_opx)
                attn_map[cc_trans_name] = attns_weights
        
        cross_modal_tensors = torch.cat(cross_mod_tensors, dim=2)
        vanilla_transformer_name = self.get_vanilla_trasformer_layer_name(index_modality.name)
        vanilla_transformer = getattr(self, vanilla_transformer_name)
        final_transformer_op = vanilla_transformer(cross_modal_tensors,mask=index_modality.mask)
        if not return_attentions:
            return final_transformer_op
        else:
            return final_transformer_op,attn_map

    def pool_sequences(self,input_channels: List[ChannelData]):
        """pool_sequences 
        Pools either the cls token uses mean pooling strategy.
        """
        pooled_features = []
        for c in input_channels:
            pooled_features.append(ChannelData(
                mask = c.mask,
                name = c.name,
                sequence = self.to_cls(c.sequence[:, 0]) if self.pooling_strategy == 'cls' else c.sequence.mean(dim=1)
            ))
        return pooled_features
    
    def transform_pooled_sequence_features(self,input_channels:List[ChannelData]):
        """transform_pooled_sequence_features 
        Run final feedforward and resudial on the concatenated cross channel predictions. 
        """
        concat_tensor = torch.cat(tuple(c.sequence for c in input_channels),dim=1)
        concat_tensor_proj = self.final_layer(concat_tensor)
        concat_tensor_proj += concat_tensor
        return concat_tensor_proj


    def forward(self, input_channels: List[ChannelData], return_attentions=False):
        """forward [summary]
        Use input_channels objects and 
        Mutate Input from those objects to create new objects

        :param input_channels: [description]
        :type input_channels: List[ChannelInput]
        """
        # $ Create Embeddings For Each Channel
        embedding_modded_data = self.get_channel_embeddings(input_channels)
        # $ run 1d conv layers dim reduction to bring dimensions of all seqs to be the same. 
        convd_channels = self.perform_dim_reduction_conv(embedding_modded_data)
        # $ Add class tokens to Seqs and Masks
        cls_token_added_output = self.add_cls_tokens(convd_channels)
        # $ run cross channel jazz
        transformered_features,attn_maps = self.get_cross_modal_features(cls_token_added_output,return_attentions=return_attentions)
        # $ use pooling strateggy.
        pooled_seqs = self.pool_sequences(transformered_features)
        # $ run final_layer against pooled Tenor
        projection_tensor = self.transform_pooled_sequence_features(pooled_seqs)
        return projection_tensor,attn_maps
