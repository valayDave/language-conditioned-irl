# Dataset Loading For Training Reward Model  

## Core Ideas
The data loading takes place from the raw source of data which holds episodic trajectories and sentence based description for those episodic trajectories. For either the usecase of mountaincar or robot experiments the high level process stays the same. `Raw Data` --> `Contrasting Indices Dataset`. The conversion to `Contrasting Indices Dataset` is done based on some rule to create the contrasting indices. 

## Mountain Car. 

The dataset is present [here](https://drive.google.com/uc?id=16gBW5HXXhr0x7bri-Newj8KubmAe8aq3). `gdown` can be used to download file programmatically. 

[This file](../language_conditioned_rl/dataloaders/mountaincar/dataset.py) consists of the entire data loading process for the mountain car experiments. 

The dataset for mountain car is annotated with `category`. `create_contrastive_examples` is a function which will create the `ContrastiveTrainingDataset` from 

## Robotics Experiments

### Dataset Creation

Process has the following way of building: 
1. Create the `Raw Data` [created via collect_data.py.](../../../collect_data.py)
    - *TODO* : Make this file save the video with the dataset. 
2. From `Raw Data` create intermediate `h5py` dataset and `csv` metadata file about each episode in the dataset using the `HDF5VideoDatasetCreator`. The `use_channels` variable holds all the channels to filter from the raw data to create the `h5py` dataset. Below is a snippet showing how to create the dataset. 
3. Once the intermediate `h5py` file and the associated metadata `csv` are created using the `HDF5VideoDatasetCreator`, the `HDF5ContrastiveSetCreator` helps create the train and test sets with contrastive indices. 
```python
import os
from language_conditioned_rl.dataloaders.robotics.datamaker import \
            HDF5VideoDatasetCreator,\
            ContrastiveControlParameters,\
            HDF5ContrastiveSetCreator,\
            PickingObjectContrastingRule,\
            PouringShapeSizeContrast
from language_conditioned_rl.models.transformer import PRETRAINED_MODEL
from transformers import AutoTokenizer
import random
DATA_PTH = '<PATH_TO_RAW_DATA>' # Path to folder with raw data objects
SAVE_DATA_PTH = '<PATH_TO_SAVE_INTERIM_FILE>.hdf5'
USE_CHANNELS = [ # selects the maain dataset channnelss that will be used and created. 
    'tcp_position',
    'image_sequence',
    'text',
    'joint_gripper',
]

EPISODIC_OBJECT_PATHS = [os.path.join(DATA_PTH,x) for x in os.listdir(DATA_PTH) if '.json' in x]

databuilder = HDF5VideoDatasetCreator(SAVE_DATA_PTH,\
                                    random.sample(EPISODIC_OBJECT_PATHS,2),\
                                    use_channels=USE_CHANNELS,\
                                    tokenizer=AutoTokenizer.from_pretrained(PRETRAINED_MODEL))
# Creates a `SAVE_DATA_PTH`.hdf5 and `SAVE_DATA_PTH`.hdf5.meta.csv
databuilder.build()
contrasting_set_creator = HDF5ContrastiveSetCreator(
    SAVE_DATA_PTH,\
    f'{SAVE_DATA_PTH}.meta.csv',\
    ContrastiveControlParameters(
        # Number of contrasting pairs to create from the total `total_train_demos` or `total_test_demos`
        num_train_samples = 10000,
        num_test_samples = 10000,
        rules = [ 
            # These are of type `SampleContrastingRule` which 
            # helps create the contrasting indices. Check `SampleContrastingRule` to creaate contrasating indices
            PickingObjectContrastingRule,
            PouringShapeSizeContrast,
        ], 
        # Number of actual episodes/demos to use for train/test set
        total_train_demos = 12000,
        total_test_demos = 4000,
        cache_main = True,
        created_on = None,
    )
)
CONTRASTING_SAVE_FOLDER = '<PATH_TO_FOLDER_SAVING_CONTRASTING_SET>'
contrasting_set_creator.make_dataset(
    CONTRASTING_SAVE_FOLDER,chunk_size=128
)
```


### Dataset Loading 
1. Data loading requires the folder created by the `HDF5ContrastiveSetCreator` in the previous section on Data Creation. Below is snippet for Data loading. 

```python
from language_conditioned_rl.dataloaders.robotics.datamaker import \
    SampleContrastingRule,\
    PickingNoisyContrastRule,\
    SameObjectPouringIntensityRule,\
    PouringShapeSizeContrast,\
    PickingObjectContrastingRule,\
    ContrastingActionsRule,

from language_conditioned_rl.dataloaders.robotics.contrastive_dataset import \
    SentenceContrastiveDataset,\
    TaskBasedSentenceContrastiveDataset,\
    JointsChannelsConcatDataset

RULES = [PickingNoisyContrastRule(),PickingObjectContrastingRule()]
PATH_TO_DATASET_FOLDER = '<PATH_TO_DATASET_FOLDER>'
NUM_SAMPLES = 30000 # Controls how many indicies will be made. 
RULE_DISTRIBUTION = [80,20] # Controls distribution of data among rules. If empty then even distribution is assumed. 
BATCH_SIZE= 40
dataset = JointsChannelsConcatDataset(
    PATH_TO_DATASET_FOLDER,\
    normalize_images=True,\
    use_channels=USE_CHANNELS,
)
# Basically recreates the contrastive indices based on the `NUM_SAMPLES` and `RULES` and `RULE_DISTRIBUTION`
dataset.remake_indices(NUM_SAMPLES,\
    rules=RULES,\
    rule_distribution=RULE_DISTRIBUTION,\
)
roboloader = DataLoader(dataset,\
                        batch_size=BATCH_SIZE,\
                        collate_fn=dataset.collate_fn(),\
                        num_workers=0,\
                        shuffle=True,\
                        drop_last=True)
```

- *TODO* : Add information about `TaskBasedSentenceContrastiveDataset`