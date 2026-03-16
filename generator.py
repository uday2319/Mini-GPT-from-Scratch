import torch
from tokenizers import Tokenizer
from model.gpt_model import MiniGPT
from model.config import *
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer=Tokenizer.from_file("tokenizer/tokenizer.json")

model=MiniGPT(
    vocab_size,
    block_size,
    embed_dim,
    heads,
    layers
).to(device)

model.load_state_dict(torch.load("mini_gpt.pth", map_location=device))
model.eval()

prompt="KING:"
tokens=tokenizer.encode(prompt).ids
context = torch.tensor([tokens], dtype=torch.long).to(device)

max_new_tokens = 100

for _ in range(max_new_tokens):

    for _ in range(max_new_tokens):

     context = context[:, -block_size:]

    logits = model(context)

    logits = model(context)

    logits = logits[:, -1, :]  

    probs = torch.softmax(logits, dim=-1)

    next_token = torch.multinomial(probs, num_samples=1)

    context = torch.cat([context, next_token], dim=1)

generated_ids = context[0].tolist()

print(tokenizer.decode(generated_ids))

