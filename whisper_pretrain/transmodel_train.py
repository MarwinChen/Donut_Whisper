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
from modeling_trans import transModel
from tqdm import tqdm
import numpy as np
from torchinfo import summary
import json

def collate_wrapper(batch):
    pixel_values = torch.stack([i[0] for i in batch])
    labels = [i[1] for i in batch]
    return pixel_values, labels

def logging(s, logfile, logging_=True, log_=True):
    print(s)
    if log_:
        with open(logfile, 'a+') as f_log:
            f_log.write(s + '\n')

num_epochs = 100
device = "cuda:1" if torch.cuda.is_available() else "cpu"
batch_size = 8
lr = 0.0001

logfile = "/home/srt11/data_8849/model/transmodel/train_bz8_lr0.0001/logfile.txt"
logfile_simp = "/home/srt11/data_8849/model/transmodel/train_bz8_lr0.0001/logfile_simp.txt"
expdir = "/home/srt11/data_8849/model/transmodel/train_bz8_lr0.0001"
scheduler = 'warmuplr'
accumgrad = 10
decay_pct = 0.2
warmup_pct = 0.0
log_interval = 200

###Loading data
pixel_values_train = torch.load('/hdd/srt11/data_8849/data_VisDio/Visdio_Data/train/pixel_values.pth', map_location=torch.device('cpu'))
labels_train = torch.load("/hdd/srt11/data_8849/data_VisDio/Visdio_Data/train/labels.pth", map_location=torch.device('cpu'))

train_set = CustomDataset(pixel_values=pixel_values_train, labels=labels_train)
train_data = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_wrapper, shuffle=True)
# train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)

pixel_values_dev = torch.load("/hdd/srt11/data_8849/data_VisDio/Visdio_Data/validation/pixel_values.pth", map_location=torch.device('cpu'))
labels_dev = torch.load("/hdd/srt11/data_8849/data_VisDio/Visdio_Data/validation/labels.pth", map_location=torch.device('cpu'))

dev_set = CustomDataset(pixel_values=pixel_values_dev, labels=labels_dev)
dev_data = DataLoader(dev_set, batch_size=batch_size, collate_fn=collate_wrapper, shuffle=True)
# dev_data = DataLoader(dev_set, batch_size=batch_size, shuffle=True, num_workers=8)

options = whisper.decoding.DecodingOptions(language="en", fp16=False, without_timestamps=True)
model = transModel(device=device).to(device)
whisper_model = model.whisper_model
criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
decodetask = whisper.decoding.DecodingTask(whisper_model, options)
sot_sequence = decodetask.sot_sequence
sotlen = len(sot_sequence)

for name, param in model.named_parameters():
    if 'image_conv' not in name and 'image_linear' not in name and 'image_act' not in name:
        param.requires_grad = False

summary(model)

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
    with tqdm(train_data, unit="batch") as tepoch:
        for idx, batch in enumerate(tqdm(train_data, unit="batch")):
            pixel_values, labels= batch
            pixel_values = to_device(pixel_values, device)
            oritarget = [torch.tensor(y) for y in labels]
            target = pad_sequence(oritarget, batch_first=True, padding_value=-100).to(model.device)
            targetmask = target != -100
            tepoch.set_description(f"Epoch {epoch + 1}")
            optimizer.zero_grad()

            # Forward pass
            logits = model(data_donut=pixel_values, labels=target*targetmask)
            output = torch.log_softmax(logits[:, sotlen-1:-1], dim=-1)
            hyp = output.view(-1, output.size(-1))
            ref = target[:, sotlen:].reshape(-1)
            loss = F.nll_loss(output.view(-1, output.size(-1)), target[:, sotlen:].reshape(-1))
    
            # Backward and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            totalloss += loss.item()
            optimizer.step()
            tepoch.set_postfix(loss=loss.item())

            if idx != 0 and (idx + 1) % accumgrad == 0:
                # LR scheduler
                if scheduler == "warmuplr":
                    currentstep = epoch * len(train_data) + idx + 1
                    totalstep = num_epochs * len(train_data)
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
                    idx, len(train_data), time.time()-start, totalloss/log_interval, 
                    optimizer.param_groups[0]['lr']), logfile)
                logging("{} / {} steps finished in {} | Loss: {} | lr: {}".format(
                    idx, len(train_data), time.time()-start, totalloss/log_interval, 
                    optimizer.param_groups[0]['lr']), logfile_simp)
                totalloss = 0

    # validation
    model.eval()
    totalvalacc = 0
    totalvalset = 0
    with torch.no_grad():
        for idx, (pixel_values, labels) in enumerate(tqdm(dev_data, unit="batch")):          
            pixel_values = to_device(pixel_values, device)
            oritarget = [torch.tensor(y, dtype=torch.long) for y in labels]
            target = pad_sequence(oritarget, batch_first=True, padding_value=-100).to(model.device)
            targetmask = target != -100
                
            # Forward pass
            logits = model(data_donut=pixel_values, labels=target*targetmask)
            output = torch.log_softmax(logits[:, sotlen-1:-1], dim=-1)

            target = target[:, sotlen:]
            output = output.view(target.size(0), target.size(1), -1).max(dim=-1)[1]
            totalvalacc += ((output == target) * targetmask[:, sotlen:]).sum()
            totalvalset += targetmask[:, sotlen:].sum()

            if idx % 50 == 0 and idx > 0:
                logging("{} out of {} finished | time elapsed {} | ACC: {}".format(
                    idx, len(dev_data), time.time()-start, totalvalacc/totalvalset), logfile)
                
        logging("[epoch {}] Total ACC: {}".format(epoch+1, totalvalacc/totalvalset), logfile)
        logging("[epoch {}] Total ACC: {}".format(epoch+1, totalvalacc/totalvalset), logfile_simp)

        totalacc = totalvalacc / totalvalset

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