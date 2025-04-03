import torch
from functools import reduce
import loralib as lora


"""
    linear related
"""
def replace_one_linear_layer(model, name, module, lora_alpha, lora_r):
    # 获取原来的nn.Linear层的输入和输出特征数
    in_features = module.in_features
    out_features = module.out_features
    # 获取原来的nn.Linear层的权重和偏置
    weight = module.weight
    bias = module.bias
    # 创建新的lora.Linear层，其输入和输出特征数与原来的nn.Linear层相同
    new_module = lora.Linear(in_features, out_features, r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.2, merge_weights=False)
    # 将原来的nn.Linear层的权重和偏置复制给新的lora.Linear层
    new_module.weight.data = weight.data.clone()
    if bias is not None:
        new_module.bias.data = bias.data.clone()
    
    # 将新的lora.Linear层赋值给模型的相应位置
    # setattr(model, name, new_module)
    # 分割模块的名字，获取父模块和当前模块的名字
    name_parts = name.split('.')
    sub_module_name = name_parts[-1]
    parent_module_name = '.'.join(name_parts[:-1])

    # 获取父模块的实例
    parent_module = reduce(getattr, parent_module_name.split('.'), model)

    # 将新的lora.Conv1d层赋值给父模块的相应位置
    parent_module._modules[sub_module_name] = new_module


def replace_linear_layers(model, lora_alpha=1, lora_r=16):
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear):
            replace_one_linear_layer(model, name, module, lora_alpha, lora_r)


def replace_attn_layers(model, lora_alpha=1, lora_r=16):
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear) and any(s in name for s in ["query", "key", "value", "out"]):
            replace_one_linear_layer(model, name, module, lora_alpha, lora_r)


def replace_attn_layers_encoder(model, lora_alpha=1, lora_r=16):
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear) and any(s in name for s in ["query", "key", "value", "out"]) and "encoder.blocks" in name:
            replace_one_linear_layer(model, name, module, lora_alpha, lora_r)


"""
    conv related 
"""
def replace_one_conv_layer(model, name, module, lora_alpha, lora_r):
    in_channels = module.in_channels
    out_channels = module.out_channels
    kernel_size = module.kernel_size[0]
    stride = module.stride[0]
    padding = module.padding[0]
    groups = module.groups

    weight = module.weight
    bias = module.bias

    # 创建新的lora.Linear层，其输入和输出特征数与原来的nn.Linear层相同
    new_module = lora.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.2, merge_weights=False)
    # 将原来的nn.Linear层的权重和偏置复制给新的lora.Linear层
    new_module.weight.data = weight.data.clone()
    if bias is not None:
        new_module.bias.data = bias.data.clone()
    
    # 分割模块的名字，获取父模块和当前模块的名字
    name_parts = name.split('.')
    sub_module_name = name_parts[-1]
    parent_module_name = '.'.join(name_parts[:-1])

    # 获取父模块的实例
    parent_module = reduce(getattr, parent_module_name.split('.'), model)

    # 将新的lora.Conv1d层赋值给父模块的相应位置
    parent_module._modules[sub_module_name] = new_module


def replace_conv_layers(model, lora_alpha=1, lora_r=16):
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Conv1d):
            replace_one_conv_layer(model, name, module, lora_alpha, lora_r)


def replace_conv_layers_encoder(model, lora_alpha=1, lora_r=16):
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Conv1d) and "encoder.blocks" in name:
            replace_one_conv_layer(model, name, module, lora_alpha, lora_r)

"""
    linear and conv
"""
def replace_linear_conv_layers(model, lora_alpha=1, lora_r=16):
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Conv1d):
            replace_one_conv_layer(model, name, module, lora_alpha, lora_r)
        if isinstance(module, torch.nn.Linear):
            replace_one_linear_layer(model, name, module, lora_alpha, lora_r)


def replace_attn_conv_layers(model, lora_alpha=1, lora_r=16):
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Conv1d):
            replace_one_conv_layer(model, name, module, lora_alpha, lora_r)
        if isinstance(module, torch.nn.Linear) and any(s in name for s in ["query", "key", "value", "out"]):
            replace_one_linear_layer(model, name, module, lora_alpha, lora_r)


def replace_attn_conv_layers_encoder(model, lora_alpha=1, lora_r=16):
    for name, module in list(model.named_modules()):
        if "encoder.blocks" not in name:
            continue
        if isinstance(module, torch.nn.Conv1d):
            replace_one_conv_layer(model, name, module, lora_alpha, lora_r)
        if isinstance(module, torch.nn.Linear) and any(s in name for s in ["query", "key", "value", "out"]):
            replace_one_linear_layer(model, name, module, lora_alpha, lora_r)

"""
    flamingo replace
"""

def replace_multi_layers(model, lora_alpha=1, lora_r=16):
    for name, module in list(model.named_modules()):
        if name.startswith('donut_model.encoder') or name.startswith('whisper_model.decoder'):
            if isinstance(module, torch.nn.Linear) and any(s in name for s in ["query", "key", "value", "out"]):
                replace_one_linear_layer(model, name, module, lora_alpha, lora_r)