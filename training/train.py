import torch
import torch.nn.functional as F
import mlflow
import mlflow.pytorch
from model.config import *
from model.gpt_model import MiniGPT
from training.dataset import get_batch
from tokenizers import Tokenizer

tokenizer=Tokenizer.from_file("tokenizer/tokenizer.json")

with open("data/datasetTinyShakesphere.txt","r",encoding="utf-8") as f:
    text=f.read()
tokens=tokenizer.encode(text).ids

data = torch.tensor(tokens)
n1=int(0.8*len(data))
n2=int(0.9*len(data))
train_data = data[:n1]
val_data   = data[n1:n2]
test_data  = data[n2:]

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def estimate_loss(model,data_split):

    model.eval()
    losses=[]
    with torch.no_grad():
        for _ in range(50):
            xb, yb = get_batch(data_split, block_size, batch_size)
            xb = xb.to(device)
            yb = yb.to(device)
            logits= model(xb)

            loss=F.cross_entropy(logits.view(-1,vocab_size),yb.view(-1))
            losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)

def train():
    mlflow.set_tracking_uri("file:///D:/MINI-GPT/mlruns")
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
    with mlflow.start_run():
        mlflow.log_param("vocab_size", vocab_size)
        mlflow.log_param("block_size", block_size)
        mlflow.log_param("embed_dim", embed_dim)
        mlflow.log_param("heads", heads)
        mlflow.log_param("layers", layers)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)

        for step in range(max_iters):
            xb,yb=get_batch(train_data,block_size,batch_size)
            xb = xb.to(device)
            yb = yb.to(device)

            logits=model(xb)

            loss=F.cross_entropy(logits.view(-1,vocab_size),yb.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step%100==0:
                val_loss=estimate_loss(model,val_data)
                mlflow.log_metric("train_loss",loss.item(),step=step)
                mlflow.log_metric("val Loss" ,val_loss,step=step)
                print(f"step {step} | train {loss.item():.3f} | val {val_loss:.3f}")
        
        test_loss=estimate_loss(model,test_data)
        mlflow.log_metric("test loss",test_loss)
        print(f"final test loss : {test_loss:.3f}")

        torch.save(model.state_dict(), "mini_gpt.pth")
        mlflow.pytorch.log_model(model,"model",pickle_module=None)

        mlflow.end_run()
        print("Model saved as mini_gpt.pth")

if __name__=="__main__":
    train()
