# importing matplot lib
import matplotlib.pyplot as plt
import numpy as np
import torch

# importig movie py libraries
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage


class AttentionVisualizer:
    """ 
    How to use this : 
    ```
    visulizer = AttentionVisualizer(tensor,
                                    seq_len_x_lim=18,
                                    x_label_toks=decode_toks,
                                    title_message="Some Message")
                                    
    visulizer.save_visualisation(viz_name=viz_name)
    ```
    """
    def __init__(self,
                 layer_wise_attention_weights: torch.Tensor,
                 seq_len_x_lim=15,
                 seq_len_y_lim=None,
                 chosen_item=0,
                 x_label_toks=[],
                 y_label_toks=[],
                 fig_size=(10, 10),
                 title_message='',
                 ) -> None:
        # ()
        self.num_layers, _, _, seq_len_x, seq_len_y = layer_wise_attention_weights.size()
        # Doing this ensure that it work.
        self.seq_len_x_lim = seq_len_x_lim
        self.seq_len_y_lim = seq_len_y_lim
        self.chosen_item = chosen_item
        self.layer_wise_attention_weights = layer_wise_attention_weights
        self.fig, self.ax = plt.subplots(figsize=fig_size)
        self.x_label_toks = x_label_toks
        self.y_label_toks = y_label_toks
        self.title_message = title_message

    def __call__(self, t):
        # clear
        self.ax.clear()
        conv_arr = self.layer_wise_attention_weights[int(
            t)][self.chosen_item].sum(dim=0).cpu().numpy()
        if self.seq_len_x_lim is not None:
            conv_arr = conv_arr[:, :self.seq_len_x_lim]
        if self.seq_len_y_lim is not None:
            conv_arr = conv_arr[:self.seq_len_y_lim]

        cax = self.ax.matshow(conv_arr, origin='lower',
                              cmap='viridis', aspect='auto')

        self.y_label_toks
        if len(self.x_label_toks) > 0:
            self.ax.set_xticklabels(self.x_label_toks)
        if len(self.y_label_toks) > 0:
            self.ax.set_yticklabels(self.y_label_toks)
        print("Message for chart : ", self.title_message)
        if self.title_message is '':
            self.ax.set_title(f" Attetion At Layer : {int(t)} \n")
        else:
            self.ax.set_title(
                f"{self.title_message}\n Attention At Layer : {int(t)} \n")
            # self.ax.set_title(f" Attetion At Layer : {int(t)} \n")
        # returning mumpy image
        return mplfig_to_npimage(self.fig)

    def save_visualisation(self, viz_name='attention_viz.gif'):
        animation = VideoClip(make_frame=self, duration=self.num_layers)
        animation.write_gif(viz_name, fps=20)

    def show_visualisation(self, viz_name='attention_viz.gif'):
        # animation = VideoClip(self, duration = self.num_layers)
        animation = VideoClip(make_frame=self, duration=self.num_layers)
        animation.ipython_display(fps=20, loop=False, autoplay=False)
