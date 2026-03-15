import torch.nn as nn
from model.attention import MultiHeadAttention
from model.feedforward import FeedFarward

class TransformerBlock(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super().__init__()

        self.attn=MultiHeadAttention(embed_dim, num_heads)
        self.ff=FeedFarward(embed_dim)

        self.ln1=nn.LayerNorm(embed_dim)
        self.ln2=nn.LayerNorm(embed_dim)

    def forward(self,x):
        x=x+self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))

        return x
     
   