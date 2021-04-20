import torch
from dataclasses import dataclass, field


@dataclass
class ChannelData:
    mask: torch.Tensor = None
    sequence: torch.Tensor = None
    name: str = None


class ChannelHolder:
    def __setitem__(self, key, item):
        if type(item) != ChannelData:
            raise Exception('Required ChannelData as Setting Value')
        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    def __unicode__(self):
        return unicode(repr(self.__dict__))

    def to_device(self, device):
        for k in self.__dict__:
            self.__dict__[k].sequence = self.__dict__[k].sequence.to(device)
            if self.__dict__[k].mask is not None:
                self.__dict__[k].mask = self.__dict__[k].mask.to(device)

    def get_channels(self):
        return list(self.__dict__.values())


@dataclass
class ContrastiveGenerator:
    """
    This class helps create contrastive examples for all the channels of
    information we possess. 
    """
    pos_channel_holder: ChannelHolder = field(default_factory=[])
    neg_channel_holder: ChannelHolder = field(default_factory=[])

    def create_contrastive_inputs(self, core_channel):
        if core_channel not in self.pos_channel_holder or core_channel not in self.neg_channel_holder:
            raise Exception('Channel Not Present')

        core_channel_px = self.pos_channel_holder[core_channel]
        core_channel_nx = self.neg_channel_holder[core_channel]
        p_chan_vals = self.get_other_channels(core_channel, pos=True)
        n_chan_vals = self.get_other_channels(core_channel, pos=False)
        pp_channels = [*p_chan_vals, core_channel_px]
        pn_channels = [*p_chan_vals, core_channel_nx]
        nn_channels = [*n_chan_vals, core_channel_nx]
        np_channels = [*p_chan_vals, core_channel_px]
        return pp_channels, pn_channels, nn_channels, np_channels

    def __len__(self):
        return len(self.pos_channel_holder)

    def to_device(self, device):
        self.pos_channel_holder.to_device(device)
        self.neg_channel_holder.to_device(device)

    def get_other_channels(self, chan, pos=False):
        if pos:
            remaining_keys = list(
                set(list(self.pos_channel_holder.keys())) - set([chan]))
        else:
            remaining_keys = list(
                set(list(self.neg_channel_holder.keys())) - set([chan]))
        chanopx = []
        for k in remaining_keys:
            if pos:
                chanopx.append(self.pos_channel_holder[k])
            else:
                chanopx.append(self.neg_channel_holder[k])
        return chanopx
