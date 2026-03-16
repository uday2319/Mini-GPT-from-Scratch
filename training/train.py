import torch
import torch.nn.functional as F

from model.config import *
from model.gpt_model import MiniGPT
from training.dataset import get_batch
from tokenizers import Tokenizer

tokenizer=Tokenizer.from_file("tokenizer/tokenizer.json")

with open("data/datasetTinyShakesphere.txt","r",encoding="utf-8") as f:
    text=f.read()
tokens=tokenizer.encode(text).ids
data = torch.tensor(tokens)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=MiniGPT(
    vocab_size,
    block_size,
    embed_dim,
    heads,
    layers
).to(device)

optimizer=torch.optim.AdamW(
    model.parameters(),lr=learning_rate
)

for step in range(max_iters):
    xb,yb=get_batch(data,block_size,batch_size)
    xb = xb.to(device)
    yb = yb.to(device)

    logits=model(xb)

    loss=F.cross_entropy(logits.view(-1,vocab_size),yb.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step%100==0:
        print("step:",step,"loss :",loss.item())

