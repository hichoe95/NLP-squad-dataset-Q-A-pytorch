from tqdm import tqdm

def train(data, model, criterion, optimizer, batch_size = 128, num_iter = 30000, question = False, attention = False):
    device = 'cuda:4' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    model.train()
    loss_ = 0.
    acc_, start_acc, end_acc, val_score = 0, 0, 0, 0
    
    for i in tqdm(range(num_iter)):
        context, answer_start, answer_end, mask, loc = make_batch(data, batch_size, random = True, question = question)
        context, answer_start, answer_end, mask, loc = context.to(device), answer_start.to(device), answer_end.to(device), mask.to(device), loc.to(device)
#         context, answer_start, answer_end = context.to(device), answer_start.to(device), answer_end.to(device), #mask.to(device), loc.to(device) ###
        
        if attention:
            start, end = model(context, mask, loc)
        else:
            start, end = model(context, mask)

        
        loss_start = criterion(start, answer_start)
        loss_end = criterion(end, answer_end-1)
        
        loss = loss_start/2  + loss_end/2
        
        model.zero_grad()
        loss.backward()
        optimizer.step()

        # loss
        loss_ += loss.detach().cpu()
        
        # acc
        start_acc_ = (start.argmax(dim = -1) == answer_start).detach()
        end_acc_ = (end.argmax(dim = -1) == (answer_end -1)).detach()
        start_acc += start_acc_.sum().item()*1./batch_size
        end_acc += end_acc_.sum().item()*1./batch_size
        acc_ += (start_acc_ & end_acc_).sum().item()*1./batch_size
                
        if i % 1000 == 999 :
            print(f'{i + 1 : d}th iters >> loss = {loss_/1000:.4f}, start_acc = {start_acc/1000 * 100 : .4f}, end_acc = {end_acc/1000 * 100 : .4f}, acc = {acc_/1000 * 100 : .4f}')
            loss_, start_acc, end_acc, acc_ = 0., 0, 0, 0
            with torch.no_grad():
                model.eval()
                cur_val_score = validation(valid_data, model = model, batch_size = 128, question = question, attention = attention)
#                 if attention:
#                     if cur_val_score < val_score:
#                         optimizer.factor *= 0.8
#                 val_score = cur_val_score
                model.train()

    
    return model