import numpy as np
import os            
import time            
import torch            
import whisper            
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from whisper.normalizers import EnglishTextNormalizer
from vistext_dataset import CustomDataset, to_device
from tqdm import tqdm
from torchinfo import summary
import json                            
import evaluate                            

def collate_wrapper(batch):
    input_features = torch.stack([i[0] for i in batch])
    labels = [i[1] for i in batch]

    return input_features, labels                            

batch_size = 32                                                                
torch.cuda.empty_cache()
tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False, language="en", task='transcribe') 
device = "cuda:0" if torch.cuda.is_available() else "cpu"
std = EnglishTextNormalizer()
pixel_values_test = '/hdd/srt11/Dataset/test_data/pixel_values'
input_features_test = '/hdd/srt11/Dataset/test_data/input_features'
labels_test = '/hdd/srt11/Dataset/test_data/labels_false'
model_add_donut = f"/home/srt11/data_8849/model/transmodel/train_bz8_lr0.0001/model.acc.best.pth"
model_add_whisper = f'/home/srt11/data_8849/model/transmodel/whisper_pre_2/model.acc.best.pth'
model_add_cat = f'/home/srt11/data_8849/model/transmodel/train_bz8_lr0.0001_3/model.acc.best.pth'
test_set = CustomDataset(input_features_path=input_features_test, labels_path=labels_test, device=device)
test_data = DataLoader(test_set, batch_size=batch_size, collate_fn=collate_wrapper, shuffle=False)
options = whisper.decoding.DecodingOptions(language="en", fp16=False, without_timestamps=True)
special_token_set = set(tokenizer.special_tokens.values())



model = torch.load(model_add_whisper, map_location=device, weights_only=False).to(device)
model = nn.DataParallel(model, device_ids=[0,1])



# validation
model.eval()                                                                                                                


o_list, o2_list, o3_list, l_list = [], [], [], []
with torch.no_grad():
    for idx, (input_features, labels) in enumerate(tqdm(test_data, unit="batch")):          

        input_features = to_device(input_features, device)
  
        # Forward pass
        logits = model.module.generate(data_whisper = input_features)


        for o, l in zip(logits, labels):

            o[o == -100] = tokenizer.eot

            o_out = std(tokenizer.decode([t for t in o if t not in special_token_set]))# o3_out = std(tokenizer.decode([t for t in o3 if t not in special_token_set]))
            l_out = std(tokenizer.decode([t for t in l if t not in special_token_set]))
            if not l_out.strip():
                l_out = '<empty>'                                                                                                                      
            if not o_out.strip():
                o_out = '<empty>'  

            o_list.append(o_out)

            l_list.append(l_out)
            

    with open(f"test_out2.txt", "w", encoding="utf-8") as file:
        for r in o_list:
            file.write(r)
            file.write("\n")

    with open(f"test_ref.txt", "w", encoding="utf-8") as file:
        for r in l_list:
            file.write(r)
            file.write("\n")
