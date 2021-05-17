from torchvision import transforms
import torch
from PIL import Image
import json
import numpy as np
import pandas
import zlib
import random
from typing import List
from .dataset import \
    IMAGE_SIZE,\
    MAX_TRAJ_LEN,\
    MAX_VIDEO_FRAMES,\
    MAX_TRAJ_LEN,\
    USE_CHANNELS,\
    MAX_TEXT_LENGTH,\
    DISCRETE_CHANNELS,\
    get_padding



def load_json_from_file(file_path):
    with open(file_path, 'r') as f:
        json_file = json.load(f)
    return json_file


def save_json_to_file(json_dict, file_path):
    with open(file_path, 'w') as f:
        json.dump(json_dict, f)



def pad_1d_tensors(sequences, max_len=None, padding_val=0):
    """
    :param sequences: list of tensors
      sequences = [torch(b),torch(k),....torch(z)] 


    :return:
      tuple(
        padded_seq : torch(len(sequences) x max_len),
        mask : torch(len(sequences))
      )
    """
    num = len(sequences)
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_dims = (num, max_len)
    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_val)
    mask = sequences[0].data.new(*out_dims).fill_(0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
        mask[i, :length] = 1
    return out_tensor, mask


def pad_2d_tensors(sequences, max_len=None, padding_val=0):
    """
    :param sequences: list of tensors
      sequences = [torch(b x d),torch(k x d),....torch(z x d)]
    :return:
      tuple(
        padded_seq : torch(len(sequences) x max_len x d),
        mask : torch(len(sequences))
      )
    """
    if len(set([s.size(1) for s in sequences])) > 1:
        raise Exception(
            "When Padding 2d tensored sequenced, dim=1 needs to be same for all items in the sequence. ")

    num = len(sequences)
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    _, dims = sequences[0].size()

    out_dims = (num, max_len, dims)
    out_dims_mask = (num, max_len)
    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_val)
    mask = sequences[0].data.new(*out_dims_mask).fill_(0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
        mask[i, :length] = 1
    return out_tensor, mask


class RoboDataUtils:
    def __init__(self,
                 tokenizer=None,
                 image_resize_dims=IMAGE_SIZE,
                 max_traj_len=MAX_TRAJ_LEN,
                 use_channels=USE_CHANNELS,
                 max_txt_len=MAX_TEXT_LENGTH) -> None:
        self.tokenizer = tokenizer
        self.max_txt_len = max_txt_len
        self.max_traj_len = max_traj_len
        self.use_channels = use_channels
        self.img_size = image_resize_dims
        self.resize_compose = transforms.Compose(
            [transforms.Resize(image_resize_dims), transforms.ToTensor()])

    def encode_sentence(self, sentence):
        data_dict = self.tokenizer.batch_encode_plus(
            [sentence],                      # Sentence to encode.
            add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            # Pad & truncate all sentences.
            max_length=self.max_txt_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',     # Return pytorch tensors.
        )
        return (data_dict['input_ids'].squeeze(0), data_dict['attention_mask'].squeeze(0))

    def make_tensor_from_image(self, img_arr):
        image_arr = np.asarray(img_arr, dtype=np.uint8)[:, :, ::-1]
        img = Image.fromarray(image_arr, 'RGB')
        resized_img_tensor = self.resize_compose(img)
        return resized_img_tensor

    def make_tensor_from_video(self, video_arr,max_frames=MAX_VIDEO_FRAMES):
        sample_frames = random.sample(list(range(len(video_arr))),max_frames)
        video_frames = []
        for img_idx in sample_frames:
            image_arr = np.asarray(video_arr[img_idx], dtype=np.uint8)[:, :, ::-1]
            img = Image.fromarray(image_arr, 'RGB')
            resized_img_tensor = self.resize_compose(img)
            video_frames.append(resized_img_tensor)
        video_tensor = torch.stack(video_frames)
        return video_tensor

    @staticmethod
    def make_mask_from_len(len_tensor, max_size):
        '''
        len_tensor: 
        '''
        return (torch.arange(max_size)[None, :] < len_tensor[:, None]).float()

    @staticmethod
    def create_trajectory(state_dict_arr, use_channels=USE_CHANNELS):
        if len(state_dict_arr) == 0:
            return []
        channel_keys = list(state_dict_arr[0].keys())
        channels = {}
        state_df = pandas.DataFrame(state_dict_arr)
        for k in channel_keys:
            if k not in use_channels:
                continue
            channel_seq = torch.Tensor(state_df[k])
            if k in DISCRETE_CHANNELS:
                channel_seq = channel_seq.long().squeeze(1)
            channels[k] = channel_seq

        return channels

    @staticmethod
    def make_padded_sequence(sequences, max_len=None, padding_val=0):
        """
        INPUT
          sequences : [tensor.size(L1xd1),tensor.size(L2xd1),tensor.size(L3xd1)..] : b length
          OR 
          sequences : [tensor.size(L1),tensor.size(L2),tensor.size(L3)..]
        RETURNS:
          tuple(sequence_tensor,mask)
          if 1d Size sequences:
            tensor(b x max_len),tensor(max_len)
        """
        size_set = set([len(s.size()) for s in sequences])
        if len(size_set) > 1:
            raise Exception(
                "All tensors in the sequence have to be the same size")
        size = list(size_set)[0]
        if size == 1:
            return pad_1d_tensors(sequences, max_len=max_len, padding_val=padding_val)
        elif size == 2:
            return pad_2d_tensors(sequences, max_len=max_len, padding_val=padding_val)
        else:
            raise Exception("Cannot handle More than 2 mins of padding")

    def make_tensor_saveable_object(self, robo_data_dict: dict):
        required_keys = [
            'state/dict',
            'voice',
            'image',
            'main_name'
        ]
        assert len([k for k in required_keys if k in robo_data_dict]
                   ) == len(required_keys),f"These Keys are Required {', '.join(required_keys)} for the Object"
        traj_data_dict = self.create_trajectory(
            robo_data_dict['state/dict'], use_channels=self.use_channels)
        text_ids, txt_msk = self.encode_sentence(robo_data_dict['voice'])
        image_tensor = self.make_tensor_from_image(robo_data_dict['image'])
        video_data = {}
        if 'image_sequence' in robo_data_dict:
            video_tensor = self.make_tensor_from_video(robo_data_dict['image_sequence'])
            video_data['image_sequence'] = video_tensor

        input_sequence_dict = dict(
            image=image_tensor,
            text=text_ids,
            state=traj_data_dict,
            **video_data
        )
        mask_dict = dict(
            image=None,
            image_sequence=None,
            text=txt_msk,
            state={k: None for k in traj_data_dict}
        )
        return input_sequence_dict, mask_dict

    def make_saveable_chunk(self, robo_data_dicts: List[dict]):
        sequences = []
        masks = []
        id_list = []
        for robo_data_dict in robo_data_dicts:
            # No masks except for text are added att start.
            seq_dict, msk_dict = self.make_tensor_saveable_object(
                robo_data_dict)
            id_list.append(robo_data_dict['main_name'])
            sequences.append(seq_dict)
            masks.append(msk_dict)

        text_ten = torch.stack([seq_dict['text'] for seq_dict in sequences])
        text_msk_tensor = torch.stack([msk_dict['text'] for msk_dict in masks])
        image_tensor = torch.stack([seq_dict['image']
                                    for seq_dict in sequences])
        video_data = {}
        if 'image_sequence' in robo_data_dicts[0]:
            video_tensor = torch.stack([seq_dict['image_sequence']
                                    for seq_dict in sequences])
            video_data['image_sequence'] = video_tensor
    
        # Mask state Tensors for this.
        # all sequences will be padded which are non text and mask is created.
        state_dict = dict()
        state_mask_dict = dict()
        padding_vals = get_padding()
        for traj_chan in sequences[0]['state']:
            padded_traj_seq, trj_mask = self.make_padded_sequence(
                [seq_dict['state'][traj_chan] for seq_dict in sequences], max_len=self.max_traj_len, padding_val=padding_vals[traj_chan])
            state_dict[traj_chan] = padded_traj_seq
            state_mask_dict[traj_chan] = trj_mask

        input_sequence_dict = dict(
            image=image_tensor,
            text=text_ten,
            **video_data,
            **state_dict
        )
        mask_dict = dict(  # Normalized to ensure H5DataCreatorMainDataCreator works well
            image=None,
            image_sequence = None,
            text=text_msk_tensor,
            **state_mask_dict
        )
        return input_sequence_dict, mask_dict, id_list

