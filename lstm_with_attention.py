from attention_layer import *


class LSTM_attn(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, bidirec = True, dropout = 0):
        super(LSTM_attn, self).__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_direction = 2 if bidirec else 1
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size = embed_dim, hidden_size = hidden_dim,
                           num_layers = num_layers, batch_first = True, dropout = dropout,
                           bidirectional = bidirec)
        
        self.attention = Attention(hidden_dim)
        
        self.lin_hid = nn.Linear(hidden_dim * self.num_direction, hidden_dim)
        
        self.relu = nn.ReLU()
        self.out_lin = nn.Linear(hidden_dim, 2)

    def subsequent_mask(self, batch, size, loc):
        attn_shape = (batch, size, size)
#         subsequent_mask = np.triu(np.ones(attn_shape), k = 1).astype('uint8')
        subsequent_mask = np.zeros(attn_shape).astype('uint8')
    
        for i in range(batch):
            subsequent_mask[i,:,loc[i]:] = 1

        return torch.from_numpy(subsequent_mask) == 0

    
    def forward(self, x, mask, loc):
        embed = self.embedding(x)
        h, c = self.init_states(x.size(0))
        
        output, hidden = self.lstm(embed, (h,c)) # batch * seq * hidden*2
        output = self.lin_hid(output)
        
        mask = mask.unsqueeze(-1)
        output = output * mask
        
        attention_mask = self.subsequent_mask(x.size(0), x.size(1), loc) # batch , seq, mask location
        attention_mask = attention_mask.to(device)
        
        attn_output, attn_weight = self.attention(output, attention_mask)
        
        output = self.out_lin(attn_output)
        
        output = output * mask
        
        start = output[:,:,0]
        end = output[:,:,1]
        
        return start, end
        
    def init_states(self, batch_size):
        return torch.zeros(self.num_layers * self.num_direction, batch_size, self.hidden_dim).to(device), torch.zeros(self.num_layers * self.num_direction, batch_size, self.hidden_dim).to(device)
        