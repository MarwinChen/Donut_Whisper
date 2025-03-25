import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from vistext_dataset import CustomDataset, to_device
from tqdm import tqdm
import numpy as np
from torchinfo import summary

import sys
current_dir = '/home/srt11/code_8849/Donut_Finetune/donut_to_whisper/'
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from donut_whisper.normalizers import EnglishTextNormalizer # type: ignore
import whisper

def collate_wrapper(batch):
    pixel_values = torch.stack([i[0] for i in batch]) 
    input_features = torch.stack([i[1] for i in batch])
    labels = [i[2] for i in batch]
    return pixel_values, input_features, labels

torch.cuda.set_device(0)

input_features = '/hdd/srt11/Dataset/test_data/input_features'
pixel_values = '/hdd/srt11/Dataset/test_data/pixel_values'
labels = '/hdd/srt11/Dataset/test_data/labels'
# dec_input_ids = torch.load("/home/srt11/data_8849/data_VisDio/whisper_data/test/dec_input_ids.pth", map_location=torch.device("cpu"))


ref_path = "/home/srt11/data_8849/model/donut_whisper_flamingo/test_1_nosample_bz8_lr0.0005/hyp.txt"
hyp_path = "/home/srt11/data_8849/model/donut_whisper_flamingo/test_1_nosample_bz8_lr0.0005/hyp.txt"

batch_size = 8
device = "cuda:0" if torch.cuda.is_available() else "cpu"
test_set = CustomDataset(input_features_path=input_features, pixel_values_path=pixel_values, labels_path = labels)
test_data = DataLoader(dataset=test_set, batch_size=batch_size, collate_fn=collate_wrapper, shuffle=True)

std = EnglishTextNormalizer()
model = whisper.load_model(name='base', download_root="/home/srt11/.cache/whisper/").to(device)
model.eval()
tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False, task='transcribe') 
special_token_set = set(tokenizer.special_tokens.values())

index = 1
for idx, (pixel_values, input_features, labels) in enumerate(tqdm(test_data, unit="batch")):
    input_features= input_features.cuda()
    options = whisper.DecodingOptions(
        language="en",
        without_timestamps=True,
        beam_size=1,
        fp16=False,
    )

    # labels = labels.to(device)  
    oritarget = [torch.tensor(y) for y in labels]
    target = pad_sequence(oritarget, batch_first=True, padding_value=-100).to(model.device)
    input_features, pixel_values = to_device(input_features, device), to_device(pixel_values, device)
    result = whisper.decode(model=model, mel=input_features, options=options)
    hyp, ref = [], []

    for l, r in zip(labels, result):
        hyp.append(std(r.text))
        text = std(r.text)
        # with open(hyp_path, "a", encoding='utf-8') as hyp_file:
            # hyp_file.write(text + "(" + str(index) + ")" + "\n")
        print(f"result:{text}")

        l[l == -100] = tokenizer.eot
        refer = tokenizer.decode([t for t in l if t not in special_token_set])
        ref.append(std(refer))
        refer = std(refer)
        # with open(ref_path, "a", encoding='utf-8') as ref_file:
            # ref_file.write(refer + "(" + str(index) + ")" + "\n")
        print(f"refer:{refer}")

        index = index + 1

print("Program Over")