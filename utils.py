import torch
from configs import TOTAL_CUDA_MEMORY
from pynvml import *
import json
import re
import copy
from PIL import Image  
from handler.disp import disp

nvmlInit()
deviceHandle = nvmlDeviceGetHandleByIndex(0)

class RobustJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return None

def estimate_tokens(sentence : str) -> int: #only for estimate number of tokens
    chn_pattern = r'[\u4e00-\u9fa5\s.,?!+]+'
    eng_pattern = r'[^\u4e00-\u9fa5\s.,?!+]'
    remain_eng = re.sub(chn_pattern, ' ', sentence).strip()
    remain_chn = re.sub(eng_pattern, '', sentence).strip()
    remain_chn = re.sub(r'\s+', '', remain_chn)
    len_remain_eng = 0 if remain_eng.strip()=='' else len(remain_eng.strip().split(' '))
    return len_remain_eng+len(remain_chn.strip())

def access_nested_dict(nested_dict, keychain):
    if keychain is None:
        return nested_dict
    current_level = nested_dict
    for key in keychain:
        if key in current_level.keys():
            current_level = current_level[key]
        else:
            raise ValueError("access_nested_dict() received invalid key") 
    return current_level

def get_model_size(model):
    param_sizes = [param.nelement() * param.element_size() for param in model.parameters()]
    buffer_sizes = [buf.nelement() * buf.element_size() for buf in model.buffers()]
    size = sum(param_sizes + buffer_sizes) / (1024 ** 2)
    return size

def get_remain_vram():
    free_memory = TOTAL_CUDA_MEMORY - nvmlDeviceGetMemoryInfo(deviceHandle).used / 1024**2
    return free_memory

def find_best_doublet(arrs, target, sorted, valdex = -1): #Make sure to pass in a sorted list
    assert sorted == True
    if not arrs:
        return []
    if len(arrs)<2:
        return []
    left, right = 0, len(arrs) - 1
    closest_pair = None

    while left < right:
        current_sum = arrs[left][valdex] + arrs[right][valdex]
        if current_sum < target:
            left += 1  
        else:  
            if closest_pair is None or current_sum < arrs[closest_pair[0]][valdex] + arrs[closest_pair[1]][valdex]:
                closest_pair = [left, right] 
            right -= 1  

    return [arrs[closest_pair[0]], arrs[closest_pair[1]]] if closest_pair is not None else []

def find_best_triplet(arrs, target, sorted, valdex = -1): #Make sure to pass in a sorted list
    assert sorted == True
    if not arrs:
        return []
    if len(arrs)<3:
        return []
    if len(arrs)==3:
        return arrs if arrs[0][valdex]+arrs[1][valdex]+arrs[2][valdex] >= target else []
    closest_sum = float('inf')
    closest_triplet = []
    for i in range(len(arrs) - 2):
        left, right = i + 1, len(arrs) - 1
        while left < right:
            current_sum = arrs[i][valdex] + arrs[left][valdex] + arrs[right][valdex]
            if current_sum > target:
                if current_sum < closest_sum:
                    closest_sum = current_sum
                    closest_triplet = [arrs[i], arrs[left], arrs[right]]
                right -= 1
            else:
                left += 1
    return closest_triplet
    
def decide_kick_idlers(cuda_idlers, vram_to_release, target_size = None, score_upper_thres = 0.8, score_lower_thres = 0.01, number_penalty = 0.05, max_kick = 8):
    target_size = vram_to_release if not target_size else target_size
    idlers_sorted = sorted(cuda_idlers, key=lambda x: x[-1], reverse=False)
    def binary_search_cut_off(sorted_list, value):
        left, right = 0, len(sorted_list) - 1
        while left <= right:
            mid = (left + right) // 2
            if sorted_list[mid][-1] <= value:
                left = mid + 1
            else:
                right = mid - 1
        return sorted_list[:left+1]
    idlers_narrowed = binary_search_cut_off(idlers_sorted, vram_to_release) #discard models each with size larger than needed, except the closest one
    if idlers_narrowed[-1][-1] > vram_to_release:
        kick_choice = [idlers_narrowed[-1]] #add the closest one to kick_choices
        score = (target_size+vram_to_release)/2/kick_choice[-1][-1] #corresponding score
        idlers_narrowed = idlers_narrowed[:-1]
    else:
        kick_choice = []
        score = 0
    #print('score for kicking one model:',score)
    if score >= score_upper_thres:
        return kick_choice
    if len(idlers_narrowed) <2: 
        if score >= score_lower_thres:
            return kick_choice
        else:
            return []
    #then consider kicking a combination of two models
    doublet = find_best_doublet(idlers_narrowed, vram_to_release, True)
    new_score = vram_to_release/sum(map(lambda x: x[-1], doublet))*(1-number_penalty) if doublet else 0
    #print('score for kicking 2 models:',new_score)
    if new_score>score_upper_thres:
        return doublet
    #then consider kicking a combination of 3 models
    if new_score>score:
        score = new_score
        kick_choice = doublet
    if len(idlers_narrowed) <3:
        if score >= score_lower_thres:
            return kick_choice
        else:
            return []
    triplet = find_best_triplet(idlers_narrowed, vram_to_release, True)
    new_score = vram_to_release/sum(map(lambda x: x[-1], triplet))*(1-number_penalty)**2 if triplet else 0
    if new_score>score:
        score = new_score
        kick_choice = triplet
    if score >= score_lower_thres:
        return kick_choice
    if len(idlers_narrowed) <4:
        return []
    #in very bad scenarios when kicking within 3 models won't satisfy score_lower_thres:
    for i in range(4,max_kick+1):
        comb = idlers_narrowed[-i:]
        comb_vram = sum(map(lambda x: x[-1], comb))
        new_score = vram_to_release/comb_vram*(1-number_penalty)**4 if vram_to_release<comb_vram else 0
        if new_score >= score_lower_thres:
            return comb
        if new_score>0:
            #disp.yellow(f"Unable to find a satisfiable combination of cuda idlers to kick for freeing up {vram_to_release}MB, target model will remain on cpu.")
            return []

def find_target_value(data,target_key):
    found = []
    if isinstance(data, dict):  # 如果数据是字典
        if content:=data.get(target_key):  # 检查是否是图像URL类型
            found.append(content)
        else:  # 否则，检查字典的值
            for value in data.values():
                found.extend(find_target_value(value,target_key))
    elif isinstance(data, list):  # 如果数据是列表
        for item in data:
            found.extend(find_target_value(item,target_key))
    return found

def reform_messages(messages,target_keys,reform_fcns):
    data = copy.deepcopy(messages)
    found = []
    founds = []
    def reform_target_value(data,target_key,reform_fcn):
        nonlocal found
        if isinstance(data, dict):  # 如果数据是字典
            for akey in data.keys():
                if isinstance(data[akey], dict):
                    if data[akey].get(target_key):
                        found.append(data[akey][target_key])
                        data[akey] = reform_fcn(data[akey][target_key])
                    else:
                        reform_target_value(data[akey],target_key,reform_fcn)
                elif isinstance(data[akey], list): 
                    reform_target_value(data[akey],target_key,reform_fcn)
        elif isinstance(data, list):  # 如果数据是列表
            for i in range(len(data)):
                if isinstance(data[i], dict):
                    if data[i].get(target_key):
                        found.append(data[i][target_key])
                        data[i] = reform_fcn(data[i][target_key])
                    else:
                        reform_target_value(data[i],target_key,reform_fcn)
                elif isinstance(data[i], list): 
                    reform_target_value(data[i],target_key,reform_fcn)
    assert len(target_keys) == len(reform_fcns)
    for i in range(len(target_keys)):
        found = []
        reform_target_value(data,target_keys[i],reform_fcns[i])
        founds.append((found))
    return data, founds

def get_media(url_data):
    url = url_data
    if isinstance(url_data, dict):
        if "url" in url_data.keys():
            url = url_data['url']
        else:
            url = list(url_data.values())[-1]
    media = None
    try:
        media = Image.open(url).convert('RGB')  
    except Exception as e:
        disp.red(f"cannot open url {url}:", e)
    return media

loadfuncs = {
    'image_url': get_media,
    'question': lambda x: x,
    'text': lambda x: x,
}
def medias_loaded(loadkeys): #decorator to preload medias for multimodal llms
    def wrapper(func):
        def inner_wrapper(*args, **kwargs):
            if not type(args[0]) == list:
                messages = args[1]
                assert type(messages) == list and type(messages[0]) == dict, "decorator medias_loaded requires messages to be the first argument"
                messages, _ = reform_messages(messages, loadkeys, [loadfuncs[akey] for akey in loadkeys])
                return func(*args[0:1], messages, *args[2:], **kwargs)
            else:
                messages = args[0]
                assert type(messages[0]) == dict, "decorator medias_loaded requires messages to be the first argument"
                messages, _ = reform_messages(messages, loadkeys, [loadfuncs[akey] for akey in loadkeys])
                return func(messages, *args[1:], **kwargs)
        return inner_wrapper
    return wrapper