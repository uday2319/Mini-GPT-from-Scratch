import torch
import torch.nn as nn
from model.transformer_block import TransformerBlock

class MiniGPT(nn.Module):
    def __init__(self,vocab_size,block_size,embed_dim,heads,layers):
        super().__init__()
        self.block_size=block_size
        self.token_embedding=nn.Embedding(vocab_size,embed_dim)
        self.pos_embedding=nn.Embedding(block_size,embed_dim)
        
        self.blocks=nn.Sequential(*[TransformerBlock(embed_dim,heads) for _ in range(layers)])
        self.ln_f=nn.LayerNorm(embed_dim)

        self.lm_head=nn.Linear(embed_dim,vocab_size)

    def forward(self,idx):
        B,T=idx.shape
        pos=torch.arange(T,device=idx.device)
        tok_emb=self.token_embedding(idx)
        pos_emb=self.pos_embedding(pos)

        x=tok_emb+pos_emb
        x=self.blocks(x)
        x=self.ln_f(x)

        logits=self.lm_head(x)

        return logits
    

if __name__ == "__main__":

    vocab_size = 5000
    block_size = 128
    embed_dim = 256
    heads = 4
    layers = 4

    model = MiniGPT(vocab_size, block_size, embed_dim, heads, layers)

    x = torch.randint(0, vocab_size, (2, block_size))

    logits = model(x)

    print(logits.shape)           

