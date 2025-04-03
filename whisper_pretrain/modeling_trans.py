import os
import torch
import whisper
from typing import Optional, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F

from transformers import VisionEncoderDecoderModel
whisper_address = f"/home/srt11/data_8849/model/transmodel/whisper_pre_2/model.acc.best.pth"
class transModel(nn.Module):
    def __init__(self, device):
        super(transModel, self).__init__()
        
        self.whisper_model = whisper.load_model(name='small.en', download_root="/home/srt11/.cache/whisper/").to(device)
        # self.whisper_model = torch.load(whisper_address, map_location=device, weights_only=False).to(device)
        self.w_encoder = self.whisper_model.encoder
        # self.w_encoder = self.whisper_model.w_encoder
        self.decoder = self.whisper_model.decoder

        # self.m_linear = nn.Linear(768, 768)
        self.GELU = F.gelu
        self.device = device
        
    def change_device(self, device='cpu'):
        self.device = device

    def forward(self, data_whisper = None, labels = None):

        encoder_output = self.w_encoder(data_whisper)

        # encoder_output = self.GELU(self.m_linear(encoder_output))
        output = self.decoder(x=labels, xa=encoder_output)
        # output = self.whisper_model(data_whisper, labels)
        return output
    
    def generate(self, data_whisper, max_length=100, start_token=50257, end_token=50256):
        # 初始化输入序列，通常以起始token开始
        batch_size = data_whisper.shape[0]
        input_seq = [[start_token] for _ in range(batch_size)]
        generated_tokens = [[] for _ in range(batch_size)]
        encoder_output = self.w_encoder(data_whisper)

        for idx in range(max_length):
            # 将输入序列转换为张量
            input_tensor = torch.tensor([input_seq])[0].to(encoder_output.device)

            # 使用解码器生成下一个token
            output = self.decoder(x=input_tensor, xa=encoder_output)
            next_token = torch.argmax(output, dim=-1)
            next_token = next_token[:,-1:]  # 获取预测的token

            # 将预测的token添加到生成序列中
            for i in range(batch_size):
                if idx != 0:
                    if generated_tokens[i][-1] == end_token:
                        generated_tokens[i].append(end_token)
                        input_seq[i].append(end_token)
                    else:
                        next_token_values = next_token[i].item()
                        generated_tokens[i].append(next_token_values)
                        input_seq[i].append(next_token_values)
                else:
                    next_token_values = next_token[i].item()
                    generated_tokens[i].append(next_token_values)
                    input_seq[i].append(next_token_values)

            # 如果预测到结束token，停止生成
            if torch.all(torch.Tensor(generated_tokens)[:,-1] == end_token):
                break

            # 更新输入序列
            
        return generated_tokens