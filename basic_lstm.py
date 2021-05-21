# 1 - layer lstm with drop out using pytorch lstm
class LSTM_dropout(nn.Module):
    def __init__(self, voca_size, embed_dim, hidden_dim, dropout):
        super(LSTM_dropout, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(voca_size, embed_dim)
        self.lstm = nn.LSTM(input_size = embed_dim, hidden_size = hidden_dim, num_layers = 1,
                            batch_first = True, dropout = dropout)
        self.relu = nn.ReLU()
        self.out_lin = nn.Linear(hidden_dim, 2)
        
        
    def forward(self, x, mask):
        embeded = self.embedding(x)
        batch_size = x.size(0)
        
        h_t, c_t = self.init_state(batch_size)
        
        outputs, (hidden, cell_state)= self.lstm(embeded, (h_t, c_t))
        
        mask = mask.unsqueeze(-1)
         # batch * seq_len * hidden_dim
        
        outputs = self.out_lin(outputs) # batch * seq_len * 2
        
        outputs_masked = outputs * mask
        
        out_start = outputs_masked[:,:,0]
        out_end = outputs_masked[:,:,1]
        
        return out_start, out_end
    
    def init_state(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim).to(device), torch.zeros(1, batch_size, self.hidden_dim).to(device)