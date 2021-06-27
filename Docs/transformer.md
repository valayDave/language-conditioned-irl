# Model Docs 
There are two main models : `UniChannelTransformer` and the `OmniChannelTransformer`. To see the detailed models check [this file](../language_conditioned_rl/models/reward_model.py).

The goal of these models is to support arbitrary number of input channels and use a configuration for creating a transformer which can support those channels. To instantiate the models you need `ChannelConfiguration`s and an `OmniTransformerCoreConfig`. This helps create the transformer models. 

## Omni Channel Transformer Model 

The configuration to the transformer support Routing configuration amongst the information channels. Simplification of what this model tries to achieve:

```python
# Give Arbitrary number of input channels. 
# Get a condensed feature vector from a attention operation across channels. 
def f(x1,x2,....xN):
    # xi is an input cchannel with dimension (b,xis,d) where xis is length of the xi input channel. 
    # cross channel transformer jazz amongst input channels
    return torch.Tensor(B,D)
```

## Uni Channel Transformer Model 

Simplification of what this model tries to achieve:
```python
# Give Arbitrary number of input channels. 
# Get a condensed feature vector from a attention operation on same sequence. 
def f(x1,x2,....xN):
    # xi is an input cchannel with dimension (b,xis,d) where xis is length of the xi input channel. 
    # self attention transformer jazz within input channels
    return torch.Tensor(B,D)
```

## Channel Configuration
Each input channel of the model needs an explicit configuration instantiated like the one given below : 
```python
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
        if True : Will not Create/Use Embedding Layer for this channel. 

    embedding_layer:
        Instantiated nn.Module. 

    use_position_embed : 
        will inform weather Position embeddings will be used in 
        any of the transformer layers. 

    route_to_everything: 
        this is a boolean that will enforce that this channel will
        route to everyother channel. 
    
    restricted_channels:
        if `route_to_everything` is True then this will specify 
        the specific channels that the current channel's cross-channel-routing will be restricted for.  


    """
    name: str = ''
    channel_type: str = 'discrete'
    input_dim: int = None
    embedding_size: int = None
    no_embedding: bool = False
    embedding_layer: nn.Module = None
    use_position_embed: bool = True
    route_to_everything:bool=True
    restricted_channels:List[str] = field(default_factory=lambda:[])

    def to_json(self):
        return dict(
            name= self.name,
            channel_type= self.channel_type,
            input_dim= self.input_dim,
            embedding_size= self.embedding_size,
            no_embedding= self.no_embedding,
            embedding_layer= None,
            use_position_embed= self.use_position_embed,
            route_to_everything=self.route_to_everything,
            restricted_channels=self.restricted_channels,
        )

    def __post_init__(self):
        if not self.no_embedding and self.embedding_layer is None:
            raise Exception(
                "If `no_embedding` is False, then embedding_layer needs to be provided to map the inputs")

        if self.no_embedding and self.input_dim == None:
            raise Exception(
                "If No Embedding are given then Dimsion of an individual item in input sequence is required")
        
        if not self.route_to_everything and len(self.restricted_channels) == 0:
            raise Exception(
                "If ChannelConfiguration.route_to_everything=False then atleast one channel is required in ChannelConfiguration.restricted_channels")
```
- `ChannelConfiguration` explanations: 
    - `name:str` : Name identifier of the input channel. **`name` is a unique identifier and should be maintained even in the data loading.** 
    - `channel_type:str` : type of input variable for the channel. Are input values of the channel discrete values like tokens or are they continuous *d* dimensional vectors. `channel_type` == `discrete` or `channel_type` == `continous`. 
    - `no_embedding:bool` : flag decides whether the input channel will undergo an embedding layer to for transformation on the base sequence. If `True` then no embeddings will be applied. If `False` then embeddings will be applied. The below two arguments for the  `ChannelConfiguration` are dependent on `no_embedding`.
        - `input_dim:int`: If `no_embedding` is `True` then the dimensions of the input channel's individual item is required for 1dconv operation. The 1dconv operation brings the dimensions of all sequences to the same dimensionality. **Its mandatory If `no_embedding` is `True`**
        - `embedding_layer:nn.Module`: If `no_embedding` is `False` then this layer converts the input channel to an embedding sequence. **Its mandatory If `no_embedding` is `False`**
    
    - `use_position_embed:bool` : Informs if positional embeddings are summed with each input sequence before passing to transformer layer. 

    - `route_to_everything:bool` : This flag informs if this channel will perform the cross-channel attention operation for all the other `ChannelConfiguration` provided to the `OmniTransformerCoreConfig`. Based on the value of this boolean the following argument is dependent:
        -  `restricted_channels:List[str]` : if `route_to_everything`==`False` then at-least one channel is required for restricted cross attention.


## Omni Channel Transformer Config
To instantiate an omnichannel transformer you need to instantiate a `OmniTransformerCoreConfig` which consists of the transformer params and the configurations for the individual input channels (`ChannelConfiguration`). 
```python
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

    debug:bool=False # useless flag for now. 

    def to_json(self):
        pass  # todo
```

## Channel Maker

This class helps implement the channel as subclasses, so they can make the channels at time of instantiation of the entire mode. This abstraction helps for performance because it explicitly help create the embedding layer.  
```python
class ChannelMaker(metaclass=abc.ABCMeta):
    
    def __init__(self,
                name: str = '',
                channel_type: str = 'discrete',
                input_dim: int = None,
                embedding_size: int = None,
                no_embedding: bool = False,
                embedding_layer: nn.Module = None,
                use_position_embed: bool = True,
                route_to_everything:bool=True,
                restricted_channels:List[str] = []) -> None:
        self.name = name
        self.channel_type = channel_type
        self.input_dim = input_dim
        self.embedding_size = embedding_size
        self.no_embedding = no_embedding
        self.embedding_layer = embedding_layer
        self.use_position_embed = use_position_embed
        self.route_to_everything = route_to_everything
        self.restricted_channels = restricted_channels
        
    def make_channel(self)->ChannelConfiguration:
        raise NotImplementedError

    def from_json(self,json_dict)->ChannelConfiguration:
        raise NotImplementedError
```

## Some Observations From Training 

1. Bigger Models are finding better decision boundaries with smaller Batchsizes
2. Smaller Models are also doing good with bigger batch sizes
3. Sentence Grounding examples based data-augmentation is extreamely benificial in boosting training results. 
    1. Sentence grounding means that we creating training tuples we create 
4. Size of transfomer's embeddings were tuned down to as small as 16 but it still finds pretty distinct boundaries. 

