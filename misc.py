import torch
import torch.nn as nn
import numpy as np

device = 'cuda:4' if torch.cuda.is_available() else 'cpu'

# word2id, id2word

def make_dict(train_data, del_train_data_index):

	word2id = {}
	tokens = []
	del_idx = []
	for i in range(len(train_data)):
	    if i in del_train_data_index:
	        continue
	    context_tokens = train_data[i]['context']
	    question_tokens = train_data[i]['question']
	    tokens.extend(context_tokens)
	    tokens.extend(question_tokens)
	    if not train_data[i]['answers']:
	        del_idx.extend([i])

	vocab = ['UNK'] + list(set(tokens))
	    
	word2id = {word : id_ for id_ , word in enumerate(vocab)}
	id2word = {id_ : word for word, id_ in word2id.items()}

	return word2id, id2word

def wordToid(data):
    context = data['context']
    question = data['question']
    
    context = [word2id[word] if word in word2id else 0 for word in context]
    question = [word2id[word] if word in word2id else 0 for word in question]
    
    answer = []
    
    for dic in data['answers']:
        start = dic['start']
        end = dic['end']
        answer.append(context[start:end])
    return context, question, answer



def valid_answers(data_index, index = []):
    data = valid_data[data_index]
    context = data['context']
    tokens_start = data['tokens_start']
    tokens_len = data['tokens_len']
    where_space = data['where_space']
    
    if index[-1] > len(context):
        index[-1] = len(context)
    
    ans_tokens = context[index[0]:index[1]]

    ans = ''
    for i, token in enumerate(ans_tokens):
        if tokens_start[index[0] + i] + tokens_len[index[0] + i] in where_space:
            ans += (token + ' ')
        else:
            ans += token
    
    return ans.strip().lower()
    
def id2word_answer(data_index, start, end):
    ### valid 비어있는지 확인하기 !!!
    
    data = valid_data[data_index] 
    origin_data = squad_dataset['validation'][data_index]
    
    ans = valid_answers(data_index, [start,end])
    
    for i, a in enumerate(origin_data['answers']['text']):
        origin_data['answers']['text'][i] = a.lower()
    
    prediction_ = {'prediction_text': ans, 'id': origin_data['id']}
    reference = {'answers' : origin_data['answers'], 'id': origin_data['id']}
    
    return prediction_, reference

