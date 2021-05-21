def validation(data, model, batch_size = 128, question = False, attention = False):
    device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
    predictions = []
    references = []


    for i in range(len(data)//batch_size):
        c, a_s, a_e, m, l= make_batch(data, batch_size = batch_size, index = np.arange(i*batch_size, (i+1)*batch_size), random = False, question = question)
        c, a_s, a_e, m, l = c.to(device), a_s.to(device), a_e.to(device), m.to(device), l.to(device)
        
        if attention:
            start, end = model(c,m,l)
        else:
            start, end = model(c, m) # batch_size * seq_len
        
        start_index = start.argmax(dim = 1).detach()
        end_index = end.argmax(dim = 1).detach()
        end_index += 1
        
        for j in range(batch_size):
            if i * batch_size + j in del_valid_data_index:
                continue
            if start_index[j] >= end_index[j]:
                continue
            pred, refer = id2word_answer(i * batch_size + j, start_index[j], end_index[j])
            predictions.append(pred)
            references.append(refer)

    results = squad_metric.compute(predictions = predictions, references = references)
    print('validation score : ', results)
    return results['f1']