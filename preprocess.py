# tokenizing context with only space.

def preprocess(example):
    
    out = {}

    context_token = example['context'].strip().split(' ')
    question_token = example['question'].strip().split(' ')
    
    out['context'] = context_token
    out['question'] = question_token
    
    if 'answers' not in example:
        return out
    
    answer_start = example['answers']['answer_start']
    out['answers'] = []
    for i, ans_st in enumerate(answer_start):
        c_token_len = len(example['context'][:ans_st].strip().split(' '))
        a_token_len = len(example['answers']['text'][i].strip().split(' '))
        out['answers'].append({'start' : c_token_len, 'end' : c_token_len + a_token_len})
    
    return out

# tokenizing conext with space and symbolic letter.

def tokenizing1(text):
    text = text.lower()
    tokens = []
    tokens_start = []
    tokens_len = []
    where_space = []
    i, token_start, token_len = 0, 0, 0
    
    while i < len(text):
        # is character alphbet?
        if text[i].isalpha(): # alphabet
            while True:
                token_len += 1
                if i + token_len >= len(text) or not text[i + token_len].isalpha():
                    break
            tokens.append(text[i:i+token_len])
            tokens_len.append(token_len)
            tokens_start.append(i)
            i += token_len
            token_len = 0
        elif text[i] == ' ': # space
            where_space.append(i)
            i += 1
        else: # symbolic char
            tokens.append(text[i])
            tokens_len.append(1)
            tokens_start.append(i)
            i += 1
    return tokens, tokens_start, tokens_len, where_space

def advanced_preprocess1(data):
    
    out = {}
    tokens, tokens_start, tokens_len, where_space = tokenizing1(data['context'])
    tokens_q, _, _, _ = tokenizing1(data['question'])
    
    answer_start = data['answers']['answer_start']
    
    out = {'context' : tokens, 'question' : tokens_q, 'tokens_start' : tokens_start, 'tokens_len' : tokens_len, 'where_space' : where_space}
    
    out['answers'] = []
    
    for i, ans in enumerate(answer_start):
        start_index = tokens_start.index(ans)
        tokens, _, _, _ = tokenizing1(data['answers']['text'][i])
        end_index = start_index + len(tokens)
        out['answers'].append({'start' : start_index, 'end' : end_index})
    return out

def token2string(context, tokens_start, tokens_len, tokens_answers, where_space, test = []):
    
    cont = ''

    for i, token in enumerate(context):
        if tokens_start[i] + tokens_len[i] in where_space:
            cont += (token + ' ')
        else:
            cont += token
    
    if len(test) > 0 :
            start_index = tokens_start[test[0]] if test[0] < len(context) else tokens_start[-1]
            end_index = tokens_start[test[1]] if test[1] < len(context) else tokens_start[-1]
    else:
        try:
            tokens_answers_start = tokens_answers['start']
            tokens_answers_end = tokens_answers['end']
            start_index = tokens_start[tokens_answers_start]
            end_index = tokens_start[tokens_answers_end-1] + tokens_len[tokens_answers_end -1]
        except:
             pass
    
    ans = cont[start_index : end_index]
    return cont, ans






