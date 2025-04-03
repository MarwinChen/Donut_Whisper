import os
import re
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from nltk import edit_distance
from custom_datasets import CustomDataset
from transformers import VisionEncoderDecoderModel, DonutProcessor
from tqdm import tqdm
import numpy as np
from torchinfo import summary

def collate_wrapper(batch):
    pixel_values = torch.stack([i[0] for i in batch])
    labels = [i[1] for i in batch]
    return pixel_values, labels

def logging(s, logfile, logging_=True, log_=True):
    print(s)
    if log_:
        with open(logfile, 'a+') as f_log:
            f_log.write(s + '\n')

###basic configuration
os.environ["TOKENIZERS_PARALLELISM"] = "false"
num_epochs = 50 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1
lr = 0.0001

processor = DonutProcessor.from_pretrained(f"/home/srt11/Haggingface_Donut/models--naver-clova-ix--donut-base-finetuned-cord-v2/snapshots/8003d433113256b4ce3a0f5bf604b29ff78a7451")
logfile = '/home/srt11/code_8849/donut_whisper_work/checkpoint/exp2/logfile.txt'
logfile_simp = '/home/srt11/code_8849/donut_whisper_work/checkpoint/exp2/logfile_simp.txt'
expdir = '/home/srt11/code_8849/donut_whisper_work/checkpoint/exp2'
scheduler = 'warmuplr'
accumgrad = 10
decay_pct = 0.2
warmup_pct = 0.0
log_interval = 1000
max_length = 768

model = VisionEncoderDecoderModel.from_pretrained(f"/home/srt11/Haggingface_Donut/models--naver-clova-ix--donut-base-finetuned-cord-v2/snapshots/8003d433113256b4ce3a0f5bf604b29ff78a7451")
model = model.to(device)
frame_folder = "/home/srt11/data_8849/Frame_remake/S01_ALL/"

lora_mode = 0
lora_pretrain = 0
lora_alpha = 64
lora_r = 4 

##################
# Lora
##################
if lora_mode != 999:  # lora_mode=999表示不使用lora
    import loralib as lora
    import sys
    import whisper
    sys.path.append("/nfs/home/zhaoqiuming/projects/quantization/torch-quantization/code/whisper")
    from lora_replace import replace_attn_layers, replace_conv_layers, replace_attn_conv_layers, replace_attn_conv_layers_encoder, replace_linear_layers, replace_linear_conv_layers

    # replace layer
    if lora_pretrain == 0:
        if lora_mode == 0:
            replace_attn_layers(model, lora_alpha=lora_alpha, lora_r=lora_r)
        elif lora_mode == 3:
            replace_conv_layers(model, lora_alpha=lora_alpha, lora_r=lora_r)
        elif lora_mode == 4:
            replace_attn_conv_layers(model, lora_alpha=lora_alpha, lora_r=lora_r)
        elif lora_mode == 7:
            replace_attn_conv_layers_encoder(model, lora_alpha=lora_alpha, lora_r=lora_r)
        elif lora_mode == 10:
            replace_linear_layers(model, lora_alpha=lora_alpha, lora_r=lora_r)
        elif lora_mode == 14:
            replace_linear_conv_layers(model, lora_alpha=lora_alpha, lora_r=lora_r)
    
    model.to(device)

    # freeze
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
        else:
            p.requires_grad = True

summary(model)

###Loading data
pixel_values = torch.load('/hdd/srt11/dataset/train_data/pixel_values.pth')
labels = torch.load('/hdd/srt11/dataset/train_data/labels.pth')
with open('/hdd/srt11/dataset/train_data/target.json', 'r') as f:
    load_json = json.load(f)
target_sequences = load_json['target_sequences']
train_dataset = CustomDataset(pixel_values, labels, target_sequences)

pixel_values = torch.load('/hdd/srt11/dataset/val_data/pixel_values.pth')
labels = torch.load('/hdd/srt11/dataset/val_data/labels.pth')
with open('/hdd/srt11/dataset/val_data/target.json', 'r') as f:
    load_json = json.load(f)
target_sequences = load_json['target_sequences']
val_dataset = CustomDataset(pixel_values, labels, target_sequences)

model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([''])[0]

print("Pad token ID:", processor.decode([model.config.pad_token_id]))
print("Decoder start token ID:", processor.decode([model.config.decoder_start_token_id]))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

batch = next(iter(train_dataloader))
pixel_values, labels, target_sequences = batch
print(pixel_values.shape)
print(target_sequences[0])

config = {"max_epochs":30,
          "val_check_interval":0.2, # how many times we want to validate during an epoch
          "check_val_every_n_epoch":2,
          "gradient_clip_val":1.0,
          "num_training_samples_per_epoch": 800,
          "lr":3e-5,
          "train_batch_sizes": [8],
          "val_batch_sizes": [8],
          # "seed":2022,
          "num_nodes": 4,
          "warmup_steps": 300, # 800/8*30/10, 10%
          "result_path": "./result",
          "verbose": True,
          }

criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# model_module = DonutModelPLModule(config, processor, model)

# trainer
logging(f"batch_size: {batch_size}", logfile)
logging(f"batch_size: {batch_size}", logfile_simp)
# scheduler
logging(f"scheduler: {scheduler}, lr: {lr}, max_epoch: {num_epochs}", logfile)
logging(f"scheduler: {scheduler}, lr: {lr}, max_epoch: {num_epochs}", logfile_simp)

bestacc = 0
bestepoch = 0
logging("Start of training", logfile)
logging("Start of training", logfile_simp)
for epoch in range(num_epochs):
    start = time.time()
    totalloss = 0

    #training
    model.train()
    with tqdm(train_dataloader, unit="batch") as tepoch:
        for idx, batch in enumerate(tqdm(train_dataloader, unit="batch")):
            pixel_values, labels, _ = batch
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
        
            outputs = model(pixel_values, labels=labels)
            loss = outputs.loss

            # Backward and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            totalloss += loss.item()
            optimizer.step()
            tepoch.set_postfix(loss=loss.item())

            if idx != 0 and (idx + 1) % accumgrad == 0:
                # LR scheduler
                if scheduler == "warmuplr":
                    currentstep = epoch * len(train_dataloader) + idx + 1
                    totalstep = num_epochs * len(train_dataloader)
                    if currentstep > int(decay_pct * totalstep):
                        factor = (totalstep - currentstep) / (totalstep - int(decay_pct * totalstep))
                        optimizer.param_groups[0]['lr'] = lr * max(0, factor)
                        logging("model-parameter update lr: {}".format(optimizer.param_groups[0]['lr']), logfile)
                    elif currentstep < int(warmup_pct * totalstep):
                        factor = currentstep / int(warmup_pct * totalstep)
                        optimizer.param_groups[0]['lr'] = lr * factor
                        logging("model-parameter update lr: {}".format(optimizer.param_groups[0]['lr']), logfile)
                elif scheduler == "fixlr":
                    pass
                elif scheduler == "steplr":
                    optimizer.param_groups[0]['lr'] = lr * (0.9 ** epoch)
                optimizer.step()
                # logging("model-parameter update.", logfile)

            if idx != 0 and idx % log_interval == 0:
                logging("{} / {} steps finished in {} | Loss: {} | lr: {}".format(
                    idx, len(train_dataloader), time.time()-start, totalloss/log_interval, 
                    optimizer.param_groups[0]['lr']), logfile)
                logging("{} / {} steps finished in {} | Loss: {} | lr: {}".format(
                    idx, len(train_dataloader), time.time()-start, totalloss/log_interval, 
                    optimizer.param_groups[0]['lr']), logfile_simp)
                totalloss = 0

    # validation
    model.eval()
    totalvalacc = 0
    totalvalset = 0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_dataloader, unit="batch")):
            # Forward pass
            pixel_values, _, answers = batch
            pixel_values = pixel_values.to(device)
            batch_size = pixel_values.shape[0]
            # we feed the prompt to the model
            decoder_input_ids = torch.full((batch_size, 1), model.config.decoder_start_token_id, device=device)
        
            outputs = model.generate(pixel_values,
                                   decoder_input_ids=decoder_input_ids,
                                   max_length=max_length,
                                   early_stopping=True,
                                   pad_token_id=processor.tokenizer.pad_token_id,
                                   eos_token_id=processor.tokenizer.eos_token_id,
                                   use_cache=True,
                                   num_beams=1,
                                   bad_words_ids=[[processor.tokenizer.unk_token_id]],
                                   return_dict_in_generate=True,)
    
            predictions = []
            for seq in processor.tokenizer.batch_decode(outputs.sequences):
                seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
                seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
                predictions.append(seq)

            scores = []
            for pred, answer in zip(predictions, answers):
                pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
                answer = answer.replace(processor.tokenizer.eos_token, "")
                if max(len(pred), len(answer)) == 0:
                    scores.append(0)
                else:
                    scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

                if config.get("verbose", False) and len(scores) == 1:
                    print(f"Prediction: {pred}")
                    print(f"    Answer: {answer}")
                    print(f" Normed ED: {scores[0]}")

            score = np.mean(scores)
            totalvalacc += score

            if idx % 50 == 0 and idx > 0:
                logging("{} out of {} finished | time elapsed {} | ACC: {}".format(
                    idx, len(val_dataloader), time.time()-start, totalvalacc/(idx+1)), logfile)
                
        logging("[epoch {}] Total ACC: {}".format(epoch+1, totalvalacc/(idx+1)), logfile)
        logging("[epoch {}] Total ACC: {}".format(epoch+1, totalvalacc/(idx+1)), logfile_simp)

        totalacc = totalvalacc / (idx+1)
    totalacc = 1 - totalacc

    if totalacc > bestacc:
        torch.save(model, os.path.join(expdir, "model.acc.best.pth"))
        bestacc = totalacc
        bestepoch = epoch + 1
        logging("Saving best model at epoch {}".format(epoch+1), logfile)
        logging("Saving best model at epoch {}".format(epoch+1), logfile_simp)

    torch.save(model, os.path.join(expdir, "snapshot.ep.{}.pth".format(epoch+1)))

logging("Saving best model at epoch {}, Max ACC: {}".format(bestepoch, bestacc), logfile)
logging("Saving best model at epoch {}, Max ACC: {}".format(bestepoch, bestacc), logfile_simp)

print("Program Over")