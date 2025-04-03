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

# model_2 = torch.load(model_add_whisper, weights_only=False).to(device)
# model_2 = nn.DataParallel(model_2, device_ids=[0,1])

# model_3 = torch.load(model_add_cat, weights_only=False).to(device)
# model_3 = nn.DataParallel(model_3, device_ids=[0,1])


# validation
model.eval()
# model_2.eval()
# model_3.eval()

o_list, o2_list, o3_list, l_list = [], [], [], []
with torch.no_grad():
    for idx, (input_features, labels) in enumerate(tqdm(test_data, unit="batch")):          

        input_features = to_device(input_features, device)
        # oritarget = [torch.tensor(y) for y in labels]
        # labell = pad_sequence([y[1:] for y in oritarget], batch_first=True, padding_value=-100).to(model.module.device)
        # target = pad_sequence([y[:-1] for y in oritarget], batch_first=True, padding_value=tokenizer.eot).to(model.module.device)
        
        # Forward pass
        logits = model.module.generate(data_whisper = input_features)
        # logits2 = model_2(data_whisper = input_features, labels = target)
        # logits3 = model_3(data_donut=pixel_values, data_whisper = input_features, labels=target)
        # logits[logits == -100] = tokenizer.eot
        # labell[labell == -100] = tokenizer.eot
    
        # output = torch.log_softmax(logits[:, sotlen-1:-1], dim=-1)

        for o, l in zip(logits, labels):
            # o = torch.argmax(o, dim=1)
            # o2 = torch.argmax(o2, dim=1)
            # o3 = torch.argmax(o3, dim=1)
            o[o == -100] = tokenizer.eot
            # o2[o == -100] = tokenizer.eot
            # o3[o == -100] = tokenizer.eot
            o_out = std(tokenizer.decode([t for t in o if t not in special_token_set]))
            # o2_out = std(tokenizer.decode([t for t in o2 if t not in special_token_set]))
            # o3_out = std(tokenizer.decode([t for t in o3 if t not in special_token_set]))
            l_out = std(tokenizer.decode([t for t in l if t not in special_token_set]))
            if not l_out.strip():
                l_out = '<empty>'
            if not o_out.strip():
                o_out = '<empty>'
            # if not o2_out.strip():
            #     o2_out = '<empty>'
            # if not o3_out.strip():
            #     o3_out = '<empty>'
            o_list.append(o_out)
            # o2_list.append(o2_out)
            # o3_list.append(o3_out)
            l_list.append(l_out)
            

    with open(f"test_out2.txt", "w", encoding="utf-8") as file:
        for r in o_list:
            file.write(r)
            file.write("\n")
    # with open(f"test_out2.txt", "w", encoding="utf-8") as file:
    #     for r in o2_list:
    #         file.write(r)
    #         file.write("\n")
    # with open(f"test_out3.txt", "w", encoding="utf-8") as file:
    #     for r in o3_list:
    #         file.write(r)
    #         file.write("\n")
    with open(f"test_ref.txt", "w", encoding="utf-8") as file:
        for r in l_list:
            file.write(r)
            file.write("\n")


def compute_wer(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    m = len(ref_words)
    n = len(hyp_words)
    
    # 初始化动态规划表
    dp = np.zeros((m+1, n+1))
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        if j < m+1:
            dp[j][0] = j
    
    # 填写动态规划表
    for i in range(1, m+1):
        for j in range(1, n+1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],      # 删除
                                   dp[i][j-1],      # 插入
                                   dp[i-1][j-1])    # 替换
    
    # 计算WER
    wer = dp[m][n] / m
    
    # 找出错误的位置和类型（替换、删除、插入）
    errors = []
    i, j = m, n
    while i > 0 and j > 0:
        if ref_words[i-1] == hyp_words[j-1]:
            i -= 1
            j -= 1
        else:
            if dp[i][j] == dp[i-1][j] + 1:
                errors.append(('删除', i-1))  # 删除参考文本中的词
                i -= 1
            elif dp[i][j] == dp[i][j-1] + 1:
                errors.append(('插入', j-1))  # 插入生成文本中的词
                j -= 1
            else:
                errors.append(('替换', i-1))  # 替换错误
                i -= 1
                j -= 1
    errors.reverse()
    
    return dp[m][n], m, errors

def analyze_complementarity_batch(references, texts1, texts2, texts3):
    total_t1_errors = 0
    total_t2_errors = 0
    total_t3_errors = 0
    total_t1_error_where_t2_correct = 0
    total_t1_error_where_t3_correct = 0
    total_t2_error_where_t1_correct = 0
    total_t2_error_where_t3_correct = 0
    total_t3_error_where_t1_correct = 0
    total_t3_error_where_t2_correct = 0
    total_a = 0
    total_b = 0
    total_c = 0
    total_m1 = 0
    total_m2 = 0
    total_m3 = 0
    total_all_correct = 0  # 三个模型都正确的错误位置
    
    for ref, t1, t2, t3 in zip(references, texts1, texts2, texts3):
        # 计算WER和错误位置
        a, m1, t1_errors = compute_wer(ref, t1)
        b, m2, t2_errors = compute_wer(ref, t2)
        c, m3, t3_errors = compute_wer(ref, t3)
        total_a +=a
        total_b +=b
        total_c +=c
        total_m1 += m1
        total_m2 += m2
        total_m3 += m3
        
        ref_words = ref.split()
        
        # 获取错误位置集合
        t1_error_positions = set(pos for _, pos in t1_errors)
        t2_error_positions = set(pos for _, pos in t2_errors)
        t3_error_positions = set(pos for _, pos in t3_errors)
        
        # 检查互补性
        # text1错误的位置中，text2或text3是否正确
        for pos in t1_error_positions:
            if pos < len(t2.split()) and t2.split()[pos] == ref_words[pos]:
                total_t1_error_where_t2_correct += 1
            if pos < len(t3.split()) and t3.split()[pos] == ref_words[pos]:
                total_t1_error_where_t3_correct += 1
        
        # text2错误的位置中，text1或text3是否正确
        for pos in t2_error_positions:
            if pos < len(t1.split()) and t1.split()[pos] == ref_words[pos]:
                total_t2_error_where_t1_correct += 1
            if pos < len(t3.split()) and t3.split()[pos] == ref_words[pos]:
                total_t2_error_where_t3_correct += 1
        
        # text3错误的位置中，text1或text2是否正确
        for pos in t3_error_positions:
            if pos < len(t1.split()) and t1.split()[pos] == ref_words[pos]:
                total_t3_error_where_t1_correct += 1
            if pos < len(t2.split()) and t2.split()[pos] == ref_words[pos]:
                total_t3_error_where_t2_correct += 1
        
        # 统计所有模型都正确的错误位置
        all_error_positions = t1_error_positions.union(t2_error_positions).union(t3_error_positions)
        for pos in all_error_positions:
            if pos < len(t1.split()) and t1.split()[pos] == ref_words[pos]:
                continue
            if pos < len(t2.split()) and t2.split()[pos] == ref_words[pos]:
                continue
            if pos < len(t3.split()) and t3.split()[pos] == ref_words[pos]:
                continue
            total_all_correct += 1
        
        # 更新错误总数
        total_t1_errors += len(t1_errors)
        total_t2_errors += len(t2_errors)
        total_t3_errors += len(t3_errors)
    
    # 返回整体统计结果
    return {
        't1_wer' : total_a/total_m1,
        't2_wer' : total_b/total_m2,
        't3_wer' : total_c/total_m3,
        'total_t1_errors': total_t1_errors,
        'total_t2_errors': total_t2_errors,
        'total_t3_errors': total_t3_errors,
        'total_t1_correct_where_t2_error': total_t1_error_where_t2_correct,
        'total_t1_correct_where_t3_error': total_t1_error_where_t3_correct,
        'total_t2_correct_where_t1_error': total_t2_error_where_t1_correct,
        'total_t2_correct_where_t3_error': total_t2_error_where_t3_correct,
        'total_t3_correct_where_t1_error': total_t3_error_where_t1_correct,
        'total_t3_correct_where_t2_error': total_t3_error_where_t2_correct,
        'total_all_correct': total_all_correct,
    }



# 示例用法
# references = l_list

# texts1 = o_list

# texts2 = o2_list

# texts3 = o3_list




# # 计算整体互补性
# complementarity = analyze_complementarity_batch(references, texts1, texts2, texts3)

# print("\nOverall Complementarity Analysis:")
# print(complementarity['t1_wer'])
# print(complementarity['t2_wer'])
# print(complementarity['t3_wer'])
# print(f"Total errors in text1: {complementarity['total_t1_errors']}")
# print(f"Total errors in text2: {complementarity['total_t2_errors']}")
# print(f"Total errors in text3: {complementarity['total_t3_errors']}")
# print("\nComplementarity:")
# print(f"Number of times text1 is wrong where text2 is correct: {complementarity['total_t1_correct_where_t2_error']}")
# print(f"Number of times text1 is wrong where text3 is correct: {complementarity['total_t1_correct_where_t3_error']}")
# print(f"Number of times text2 is wrong where text1 is correct: {complementarity['total_t2_correct_where_t1_error']}")
# print(f"Number of times text2 is wrong where text2 is correct: {complementarity['total_t2_correct_where_t3_error']}")
# print(f"Number of times text3 is wrong where text1 is correct: {complementarity['total_t3_correct_where_t1_error']}")
# print(f"Number of times text3 is wrong where text2 is correct: {complementarity['total_t3_correct_where_t2_error']}")