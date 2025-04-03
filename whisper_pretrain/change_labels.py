import re
from tqdm import tqdm
import torch
import json
from torch.utils.data import DataLoader
import whisper
from transformers import VisionEncoderDecoderModel, DonutProcessor
import numpy as np
from nltk import edit_distance

add = '/hdd/srt11/Dataset/val_data/labels'
with open('/hdd/srt11/dataset/val_data/target.json', 'r') as f:
    load_json = json.load(f)
target_sequences = load_json['target_sequences']
processor = DonutProcessor.from_pretrained(f"/home/srt11/Haggingface_Donut/models--naver-clova-ix--donut-base-finetuned-cord-v2/snapshots/8003d433113256b4ce3a0f5bf604b29ff78a7451")
tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language="en", task='transcribe') 
special_token_set = set(tokenizer.special_tokens.values())
sot = tokenizer.sot_sequence_including_notimestamps
eot = tokenizer.eot
for i in range(len(target_sequences)):
    answer = target_sequences[i]
    answer = answer.replace(processor.tokenizer.eos_token, "")
    tokens = tokenizer.encode(answer)
    tokens = list(sot)+tokens+[eot]
    torch.save(tokens, add+f"/{i}.pth")