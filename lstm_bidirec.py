class LSTM_BI(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, bidirec = True, dropout = 0):
        super(LSTM_BI, self).__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.bidirec = bidirec
        self.num_layers = num_layers
        self.num_direction = 2 if bidirec == True else 1
        
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size = embed_dim, hidden_size = hidden_dim,
                           num_layers = num_layers, batch_first = True, dropout = dropout,
                           bidirectional = bidirec)
        
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim * self.num_direction, 2)

        
    def forward(self, x, mask):
        embed = self.embedding(x)
        h, c = self.init_states(x.size(0))
        
        output, _ = self.lstm(embed, (h,c)) # batch_size * seq * hideen_dim
        mask = mask.unsqueeze(-1)

        
        output_masked = self.relu(self.out(output))
        output_masked = output_masked * mask
        start = output_masked[:,:,0]
        end = output_masked[:,:,1]
        
        return start, end
    
    def init_states(self, batch_size):
        return torch.zeros(self.num_layers * self.num_direction, batch_size, self.hidden_dim).to(device), torch.zeros(self.num_layers * self.num_direction, batch_size, self.hidden_dim).to(device)