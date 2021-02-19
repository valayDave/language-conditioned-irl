import torch.nn as nn
from typing import Set,Tuple,List
import torch
import pytorch_lightning as pl
import json
import einops
import pandas
import torch.nn.functional as F

import math
class Conv1D(nn.Module):
    """
    THANK YOU Hugginface : 
    https://github.com/huggingface/transformers/blob/4b919657313103f1ee903e32a9213b48e6433afe/src/transformers/modeling_utils.py#L1193
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

class MLP(nn.Module):
    '''
    THANK YOU Hugging Face : 
    https://github.com/huggingface/transformers/blob/1321356bdf54a6c11048851bff98c2a0181f2084/src/transformers/models/gpt2/modeling_gpt2.py#L250
    '''
    def __init__(self, n_state,embedding_size=256,resid_pdrop=0.1):  # in MLP: n_state=(n * embedding_size)
        super().__init__()
        nx = embedding_size # n_state = outputfeatures
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state) # nx = inputfeatures
        self.act = nn.GELU()
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)

class SimpleSelfAttention(nn.Module):
  '''
  Vanilla Self attention on sequence. No Masking etc.
  '''
  def __init__(self,hidden_size,dropout=0.1,num_heads=4,scale=0.2,mlp_dim=3072):
    super().__init__()
    self.kqv_layer = nn.Linear(hidden_size,3*hidden_size)
    self.num_heads = num_heads
    self.ff_layer = nn.Sequential(
          nn.Linear(hidden_size, hidden_size),
          nn.Dropout(dropout)
    )
    
    self.scale = hidden_size ** -scale

  def forward(self,sequence_embedding):
    # print(f"Shape of Sequence Embedding {sequence_embedding.shape}")
    kqv = self.kqv_layer(sequence_embedding).chunk(3, dim = -1)
    # print(f"Shape of kqv {kqv[0].shape,kqv[1].shape}")
    k,q,v = map(lambda x:einops.rearrange(x,'b s (h d) -> b h s d',h=self.num_heads),kqv)
    scaled_dot_product = torch.einsum('bhsd,bhnd->bhsn',k,q) * self.scale
    weighted_sum = F.softmax(scaled_dot_product,dim=-1)
    value_weighted_sum = torch.einsum('bhsn,bhsd->bhnd',weighted_sum,v)
    reweighted_sequence_embedding = einops.rearrange(value_weighted_sum,'b h s d -> b s (h d)',h=self.num_heads)
    return self.ff_layer(reweighted_sequence_embedding)


class Residual(nn.Module):
    """Residual 
        Thank you : https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_pytorch.py
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Block(nn.Module):
    """Block
        Code Refactored From https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_gpt2.py 
        Thank you Hugging Face. 
    """
    def __init__(self, 
                 embedding_size=256,\
                 layer_norm_epsilon=0.00001,\
                 scale=0.2,\
                 resid_pdrop=0.1,\
                 attn_pdrop=0.1,\
                 num_attention_heads = 8):
        super().__init__()
        hidden_size = embedding_size
        inner_dim = 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.attn = Residual(SimpleSelfAttention(hidden_size,num_heads = num_attention_heads,scale=scale,dropout=attn_pdrop,mlp_dim=inner_dim))
        self.ln_2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.mlp = Residual(MLP(inner_dim,embedding_size=embedding_size,resid_pdrop=resid_pdrop))
    def forward(
        self,
        hidden_states,
    ):
        attn_outputs = self.attn(
            self.ln_1(hidden_states),
        )
        feed_forward_hidden_states = self.mlp(self.ln_2(attn_outputs))
        return feed_forward_hidden_states
