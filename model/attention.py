import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads   

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):

        B, T, C = x.shape

        Q = self.q_proj(x)  
        K = self.k_proj(x)  
        V = self.v_proj(x)  


        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1,2)

        scores = (Q @ K.transpose(-2,-1)) / (self.head_dim ** 0.5)

        
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask==0, float("-inf"))

        attn = F.softmax(scores, dim=-1)

        out = attn @ V   

        out = out.transpose(1,2).contiguous().view(B,T,C)

        return self.out_proj(out)