
import torch.nn as nn
import math
import torch
import einops
import itertools
import torch.nn.functional as F
from .cross_modal_attention import MultiModalAttentionBlock
from .attention import Block
from .embeddings import ActionEmbedding, SinusoidalPositionalEmbedding

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
                 resid_pdrop=0.1,
                 attn_pdrop=0.1,
                 num_attention_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.embed_dropout = embed_dropout
        self.embed_scale = math.sqrt(embedding_size)
        self.embed_positions = SinusoidalPositionalEmbedding(embedding_size)

        for _ in range(num_layers):
            self.layers.append(Block(
                embedding_size=embedding_size,
                layer_norm_epsilon=layer_norm_epsilon,
                scale=scale,
                resid_pdrop=resid_pdrop,
                attn_pdrop=attn_pdrop,
                num_attention_heads=num_attention_heads))

    def forward(self, x, mask=None):
        # Add positional embedding
        x = self.embed_scale * x  # (b,len,d)
        if self.embed_positions is not None:
            x += self.embed_positions(x[:, :, 0])
        x = F.dropout(x, p=self.embed_dropout, training=self.training)

        hidden = x
        for attention_block in self.layers:
            hidden = attention_block(hidden)
        return hidden


class MultiModalTransformer(nn.Module):
    def __init__(self,
                 num_layers=4,
                 embed_dropout=0.1,
                 embedding_size=256,
                 layer_norm_epsilon=0.00001,
                 scale=0.2,
                 resid_pdrop=0.1,
                 attn_pdrop=0.1,
                 num_attention_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.embed_dropout = embed_dropout
        self.embed_scale = math.sqrt(embedding_size)
        self.embed_positions = SinusoidalPositionalEmbedding(embedding_size)

        for _ in range(num_layers):

            self.layers.append(MultiModalAttentionBlock(
                embedding_size=embedding_size,
                layer_norm_epsilon=layer_norm_epsilon,
                scale=scale,
                resid_pdrop=resid_pdrop,
                attn_pdrop=attn_pdrop,
                num_attention_heads=num_attention_heads))

    def forward(self, seq_x, seq_y, mask=None):
        # Add positional embedding
        seq_x = self.embed_scale * seq_x
        if self.embed_positions is not None:
            seq_x += self.embed_positions(seq_x[:, :, 0])
        seq_x = F.dropout(seq_x, p=self.embed_dropout, training=self.training)

        seq_y = self.embed_scale * seq_y
        if self.embed_positions is not None:
            seq_y += self.embed_positions(seq_y[:, :, 0])
        seq_y = F.dropout(seq_y, p=self.embed_dropout, training=self.training)

        hidden = seq_x
        for attention_block in self.layers:
            hidden = attention_block(hidden, seq_y, mask=mask)
        return hidden



class CrossModalBertEmbedTranformer(nn.Module):
    '''
        Creates Multi-Transformers under ["state","action","text"] modalities. 

        Use BERT Token Embedding for the Text Embedding Layer. 

        Inspiration From https://arxiv.org/abs/1906.00295
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
        from transformers import AutoModel

        # $ Create Embeddings for Actionos
        self.action_embedding = ActionEmbedding(
            num_actions=num_actions+1, action_type=action_type, hidden_dims=embedding_size)

        # $ Create Text Embedding
        bert_model = AutoModel.from_pretrained(PRETRAINED_MODEL)
        bert_emb = bert_model.embeddings.word_embeddings
        text_embedding_dim = bert_emb.embedding_dim
        num_emb = bert_emb.num_embeddings
        self.text_embeddings = nn.Embedding(
            num_embeddings=num_emb, embedding_dim=text_embedding_dim)
        self.text_embeddings.load_state_dict(bert_emb.state_dict())
        self.text_embeddings.weight.requires_grad = False

        # $ 1 D Convs Making same dimensions for Modalities
        self.action_conv = nn.Conv1d(
            embedding_size, common_conv_dim, kernel_size=1, padding=0, bias=False)
        self.state_conv = nn.Conv1d(
            state_dims, common_conv_dim, kernel_size=1, padding=0, bias=False)
        self.text_conv = nn.Conv1d(
            text_embedding_dim, common_conv_dim, kernel_size=1, padding=0, bias=False)

        # $ CLS token instanciations
        self.action_cls_token = nn.Parameter(
            torch.randn(1, 1, common_conv_dim))
        self.state_cls_token = nn.Parameter(torch.randn(1, 1, common_conv_dim))
        self.text_cls_token = nn.Parameter(torch.randn(1, 1, common_conv_dim))

        self.to_cls = nn.Identity()
        # $ Option to return hidden reps from all layer or just return 1 concat rep 
        self.return_hidden_tensors = return_hidden_tensors
        # $ Final layers
        self.final_layer = nn.Sequential(
            nn.Linear(common_conv_dim*3*2, common_conv_dim*3*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(common_conv_dim*3*2, common_conv_dim*3*2)
        )

        # $ Creates Transformers For multi channel Modalities. 
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

    def extract_cross_modal_tensors(self, modality, modality_tensor, cross_modal_tensors, modality_mask=None):
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
        for cross_mod, cm_tensor in cross_modal_tensors:
            cross_mod_trans_name = modality + self.join_str + \
                cross_mod + self.cross_mod_trans_common_name
            transformer = getattr(self, cross_mod_trans_name)
            cross_mod_tensors.append(transformer(
                modality_tensor, cm_tensor, mask=modality_mask))

        cross_modal_tensors = torch.cat(cross_mod_tensors, dim=2)
        modality_trans_name = modality + self.mod_trans_common_name
        mod_transformer = getattr(self, modality_trans_name)
        return mod_transformer(cross_modal_tensors)

    def forward(self, state_tensor, action_tensor, text_tensor, text_mask=None, act_mask=None, st_mask=None):
        '''
        state_tensor : (B,L,ds)
        action_tensor : (B,L,da)
        text_tensor: (B,T,dt)
        L: trajectory_length
        S: Length of the Text. 

        Returns Projection : (b,common_dim*6)
        '''
        # $ Convert Action/State to Embedding
        action_tensor = self.action_embedding(action_tensor)
        text_tensor = self.text_embeddings(text_tensor)

        # $ Transpose lenght,dim to dim,length for convolutions
        state_tensor = state_tensor.transpose(1, 2)
        action_tensor = action_tensor.transpose(1, 2)
        text_tensor = text_tensor.transpose(1, 2)

        # print(f"Shape PRE-CONV state_tensor {state_tensor.shape} action_tensor {action_tensor.shape} text_tensor {text_tensor.shape}")

        # $ Apply Dimensionality Reduction to the information. :
        state_tensor, action_tensor, text_tensor = self.apply_dim_reduction_convolutions(
            state_tensor, action_tensor, text_tensor)

        # $ Prepend CLS tokens and Finally extract thoose instead of the last token.
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

        # $ adding Extra one for cls tokens that get prepended the tensors
        if text_mask != None:
            text_mask = torch.cat((torch.ones(b).unsqueeze(
                1).to(state_tensor.device), text_mask), dim=1)
        if act_mask != None:
            act_mask = torch.cat((torch.ones(b).unsqueeze(
                1).to(state_tensor.device), act_mask), dim=1)
        if st_mask != None:
            st_mask = torch.cat((torch.ones(b).unsqueeze(
                1).to(state_tensor.device), st_mask), dim=1)

        # $ Apply cross modality Transformer.

        # $ Create Cross modality tuples for input to cross_mod tensors.
        st_at_tup = (('state', state_tensor), ('action', action_tensor))
        st_tt_tup = (('state', state_tensor), ('text', text_tensor))
        tt_at_tup = (('text', text_tensor), ('action', action_tensor))

        # print(f"Shape PRE-MODEL state_tensor {state_tensor.shape} action_tensor {action_tensor.shape} text_tensor {text_tensor.shape}")

        # $ input to the cross modal transformers is the modality input and the cross modal inputs.
        text_j_tensor = self.extract_cross_modal_tensors(
            'text', text_tensor, st_at_tup, modality_mask=text_mask)
        state_j_tensor = self.extract_cross_modal_tensors(
            'state', state_tensor, tt_at_tup, modality_mask=st_mask)
        action_j_tensor = self.extract_cross_modal_tensors(
            'action', action_tensor, st_tt_tup, modality_mask=act_mask)

        # $ concat CLS tokens of SEQUENCES from CROSS MODAL OUTPUT : tensor[:,0] does the same
        l_txt = self.to_cls(text_j_tensor[:, 0])
        l_st = self.to_cls(state_j_tensor[:, 0])
        l_at = self.to_cls(action_j_tensor[:, 0])
        concat_tensor = torch.cat((l_txt, l_st, l_at), dim=1)
        # print(f"Shape of concat_tensor {concat_tensor.shape}")

        # $ A residual block
        concat_tensor_proj = self.final_layer(concat_tensor)
        concat_tensor_proj += concat_tensor
        if not self.return_hidden_tensors:
            # $ return the CLS Token extracted Tensor.
            return concat_tensor_proj

        # $ (CLS Token extracted Tensor,tuple CLS tokens from each cross channel transformer)
        return (concat_tensor_proj, (l_txt, l_st, l_at))
