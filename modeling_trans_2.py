import os
import torch
import whisper
from typing import Optional, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F

from transformers import VisionEncoderDecoderModel
donut_address=f"donut.pth"
whisper_address = f"/home/srt11/data_8849/model/transmodel/whisper_pre_2/model.acc.best.pth"
pre_add = f"/home/srt11/data_8849/model/transmodel/train_bz8_lr0.0001/model.acc.best.pth"

class transModel(nn.Module):
    def __init__(self, device):
        super(transModel, self).__init__()
        
        self.donut_model = torch.load(donut_address, map_location=device, weights_only=False).to(device)
        self.whisper_model = torch.load(whisper_address, map_location=device, weights_only=False).to(device)
        self.pre_model = torch.load(pre_add, map_location=device, weights_only=False).to(device)
        # self.whisper_model = whisper.load_model(name='base', download_root="/home/srt11/.cache/whisper/").to(device)
        # self.donut_model = self.donut_pre_model.module.donut_model
        self.encoder = self.donut_model.encoder
        self.w_encoder = self.whisper_model.w_encoder
        self.decoder = self.whisper_model.decoder
        self.image_linear = self.pre_model.image_linear
        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.mix_linear = nn.Linear(768, 768)
        self.GELU = F.gelu
        # self.image_act = F.relu
        self.device = device
        # self.donut_processor = donut_processor
        

    def change_device(self, device='cpu'):
        self.device = device

    def forward(self, data_donut = None, data_whisper = None, labels = None):

        d_encoder_output = self.encoder(pixel_values=data_donut, output_attentions = True, output_hidden_states=True, return_dict=True).last_hidden_state
        d_encoder_output = self.GELU(self.image_linear(d_encoder_output))

        w_encoder_output = self.w_encoder(data_whisper)

        attn_output, _ = self.cross_attention(query=w_encoder_output, key=d_encoder_output, value=d_encoder_output)

        encoder_output = self.GELU(self.mix_linear(attn_output))
        output = self.decoder(x=labels, xa=encoder_output)
        return output