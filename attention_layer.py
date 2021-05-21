class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads = 1, dropout = 0.0, bias = True):
        super(Attention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads 
        self.dropout = dropout
        self.bias = bias
        self.head_dim = embed_dim // num_heads
        
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias = bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias = bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias = bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias = bias)
        self.softmax = nn.Softmax(dim = -1)
    
    def scaled_dot_product(self, q, k, v, mask = None):
        # batch * seq * seq
        attention_weight = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(self.embed_dim)
        
        if mask is not None:
            attention_weight = attention_weight.masked_fill(mask == 0, -1e9)
        
        # 각 token 이 어떤 token에 많이 attn하는지 score 계산.
        attention_weight = self.softmax(attention_weight)
        
        # batch * seq * embed_dim
        attention_output = torch.matmul(attention_weight, v)
        attention_output = self.out_proj(attention_output)
        
        return attention_output, attention_weight
    
    def forward(self, hidden, mask):
        query = self.q_proj(hidden) # batch * seq_len * hidden_dim(num_head * head_dim)
        key = self.k_proj(hidden) # batch * seq_len * hidden_dim(num_head * head_dim)
        value = self.v_proj(hidden) # batch * seq_len * hidden_dim(num_head * head_dim)
        
        
        
        attn_output, attn_weight = self.scaled_dot_product(query, key, value, mask)
        
        return attn_output, attn_weight
    