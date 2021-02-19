# Inspiration From https://arxiv.org/abs/1906.00295
import torch.nn as nn
import torch
import einops
import torch.nn.functional as F
from .attention import MLP,Residual


class CrossModalAttentionWithMask(nn.Module):
  def __init__(self,hidden_size,dropout=0.1,num_heads=4,scale=0.2):
    super().__init__()
    self.kv_layer = nn.Linear(hidden_size,2*hidden_size)
    self.q_layer = nn.Linear(hidden_size,hidden_size)
    self.num_heads = num_heads
    self.ff_layer = nn.Sequential(
          nn.Linear(hidden_size, hidden_size),
          nn.Dropout(dropout)
    )
    self.scale = hidden_size**-scale

  def forward(self,seq_x,seq_y,mask=None):
    '''
    r(seq_x|seq_y)
    r(action|text)
    r(text|action)
      action: b t d 
          - b: batchsize
          - t: trajectory length 
          - d : hidden dims
      text : b s d
          - b: batchsize
          - s : length of text
          - d: hidden dims
      *d will be same in text and traj
    '''
    kv = self.kv_layer(seq_y).chunk(2, dim = -1)
    q = einops.rearrange(self.q_layer(seq_x),'b s (h d) -> b h s d',h=self.num_heads)
    k,v = map(lambda x:einops.rearrange(x,'b s (h d) -> b h s d',h=self.num_heads),kv)
    scaled_dot_product = torch.einsum('bhsd,bhnd->bhsn',k,q) * self.scale
    # add mask here for better performance. gi
    if mask is not None:
      # print(f"Shape Of Mask {mask.shape}")
      mask_vals = self.get_extended_attention_mask(mask,seq_x.size(),device=seq_x.device)
      scaled_dot_product+=mask_vals
    # return scaled_dot_product
    weighted_sum = F.softmax(scaled_dot_product,dim=-1)
    value_weighted_sum = torch.einsum('bhsn,bhsd->bhnd',weighted_sum,v)
    reweighted_sequence_embedding = einops.rearrange(value_weighted_sum,'b h s d -> b s (h d)',h=self.num_heads)
    return self.ff_layer(reweighted_sequence_embedding)

  @staticmethod
  def get_extended_attention_mask(attention_mask, input_shape, device=torch.device('cpu'),is_decoder=False):
      """ Thank You Hugging Face : 
      https://github.com/huggingface/transformers/blob/443f67e887a030d8254eba126e5f2cdb8b70eb63/src/transformers/modeling_utils.py
      Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
      Arguments:
          attention_mask (:obj:`torch.Tensor`):
              Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
          input_shape (:obj:`Tuple[int]`):
              The shape of the input to the model.
          device: (:obj:`torch.device`):
              The device of the input to the model.
      Returns:
          :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
      """
      # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
      # ourselves in which case we just need to make it broadcastable to all heads.
      if attention_mask.dim() == 3:
          extended_attention_mask = attention_mask[:, None, :, :]
      elif attention_mask.dim() == 2:
          # Provided a padding mask of dimensions [batch_size, seq_length, d]
          # - if the model is a decoder, apply a causal mask in addition to the padding mask
          # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
          if is_decoder:
              batch_size, seq_length, d = input_shape
              seq_ids = torch.arange(seq_length, device=device)
              causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
              # in case past_key_values are used we need to add a prefix ones mask to the causal mask
              # causal and attention masks must have same type with pytorch version < 1.3
              causal_mask = causal_mask.to(attention_mask.dtype)

              if causal_mask.shape[1] < attention_mask.shape[1]:
                  prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                  causal_mask = torch.cat(
                      [
                          torch.ones(
                              (batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype
                          ),
                          causal_mask,
                      ],
                      axis=-1,
                  )

              extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
          else:
              extended_attention_mask = attention_mask[:, None, None, :]
      else:
          raise ValueError(
              "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                  input_shape, attention_mask.shape
              )
          )

      # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and -10000.0 for masked positions.
      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      # extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
      extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
      return extended_attention_mask

class MultiModalAttentionBlock(nn.Module):
    def __init__(self, 
                 embedding_size=256,\
                 layer_norm_epsilon=0.00001,\
                 scale=False,\
                 resid_pdrop=0.1,\
                 attn_pdrop=0.1,\
                 num_attention_heads = 8):
        super().__init__()
        hidden_size = embedding_size
        inner_dim = 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.attn = CrossModalAttentionWithMask(hidden_size,num_heads = num_attention_heads,scale=scale,dropout=attn_pdrop)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.mlp = Residual(MLP(inner_dim,embedding_size=embedding_size,resid_pdrop=resid_pdrop))
    def forward(
        self,
        seq_x,seq_y,mask=None
    ):
        attn_outputs = self.attn(
            self.ln_1(seq_x),self.ln_1(seq_y),mask=mask
        )
        # Residual connection
        attn_outputs = seq_x + attn_outputs
        feed_forward_hidden_states = self.mlp(self.ln_2(attn_outputs))
        return feed_forward_hidden_states


