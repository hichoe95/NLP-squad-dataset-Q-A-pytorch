from attention_layer import *

class LSTM_attn2(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, bidirec = True, dropout = 0):
        super(LSTM_attn2, self).__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_direction = 2 if bidirec else 1
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.lstm = nn.LSTM(input_size = embed_dim, hidden_size = hidden_dim,
                           num_layers = num_layers, batch_first = True, dropout = dropout,
                           bidirectional = bidirec)
        
        self.lin1 = nn.Linear(hidden_dim * self.num_direction, hidden_dim)
        
        self.attention = Attention(hidden_dim)
        
        self.lstm2 = nn.LSTM(input_size = embed_dim, hidden_size = hidden_dim,
                           num_layers = num_layers, batch_first = True, dropout = dropout,
                            bidirectional = bidirec)
        
        self.relu = nn.ReLU()
        self.out_lin = nn.Linear(hidden_dim * self.num_direction, 2)
    
    def subsequent_mask(self, batch, size, loc):
        attn_shape = (batch, size, size)
#         subsequent_mask = np.triu(np.ones(attn_shape), k = 1).astype('uint8')
        subsequent_mask = np.zeros(attn_shape).astype('uint8')
    
        for i in range(batch):
            subsequent_mask[i,:,loc[i]:] = 1

        return torch.from_numpy(subsequent_mask) == 0

    
    def forward(self, x, mask, loc):
        embed = self.embedding(x)
        h, c = self.init_states(x.size(0), self.num_direction)
        
        output, (hidden, cell) = self.lstm(embed, (h,c))
        output1 = self.lin1(output)
                
        mask = mask.unsqueeze(-1)
        output1 = output1 * mask
        
        attention_mask = self.subsequent_mask(x.size(0), x.size(1), loc) # batch , seq, mask location
        attention_mask = attention_mask.to(device)
        
        attn_output, attn_weight = self.attention(output1, attention_mask)
        
#         h2, c2 = self.init_states(x.size(0), 2)
        output_attn = output1 + attn_output
        
        output_, (hidden2, cell2) = self.lstm2(output_attn, (hidden, cell))
        output_ = self.out_lin(output_)

        start = output_[:,:,0]
        end = output_[:,:,1]
        
        return start, end
        
    def init_states(self, batch_size, num_direction):
        return torch.zeros(self.num_layers * num_direction, batch_size, self.hidden_dim).to(device), torch.zeros(self.num_layers * num_direction, batch_size, self.hidden_dim).to(device)
        