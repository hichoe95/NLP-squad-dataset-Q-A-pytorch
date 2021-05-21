# make batch . 

import numpy as np
import random
import torch
import torch.nn.functional as F

def make_batch(data, batch_size = 64, index = [], random = True, question = False):
    data = np.array(data)
    if random:
        indice = np.random.choice(len(data), batch_size, replace = False)
        for i, idx in enumerate(indice):
            if idx in del_train_data_index:
                indice[i] = 0
        data_batch = data[indice]
        
    else:
        for i,idx in enumerate(index):
            if idx in del_valid_data_index:
                index[i] = 0
        data_batch = data[index]

    context_max_len = 0
    question_max_len = 0
    
    for i in range(batch_size):
        if question:
            context_max_len = max(context_max_len, len(data_batch[i]['context']) + len(data_batch[i]['question']))
        else:
            context_max_len = max(context_max_len, len(data_batch[i]['context']))
    
    context_batch = []
    answer_start_batch = []
    answer_end_batch = []
    context_mask = []
    mask_loc = []
    
    for i, d in enumerate(data_batch):
        if d == []:
            continue
            
        context, questions, answer = wordToid(d)
        
        if question:
            context = np.concatenate([context, questions])
        
        context_len = len(context)
        context_padding = np.zeros(context_max_len - len(context))
        context = np.concatenate([context, context_padding])
        context_batch.append(context)
        
        for answers in data_batch[i]['answers']:
            answer_start_batch.append(answers['start'])
            answer_end_batch.append(answers['end'])
        
        context_mask.append(np.concatenate([np.ones(context_len), np.zeros(len(context_padding))], axis = 0))
        mask_loc.append(context_len)
        
    return torch.LongTensor(context_batch), torch.LongTensor(answer_start_batch), torch.LongTensor(answer_end_batch), torch.LongTensor(context_mask), torch.LongTensor(mask_loc)
