import torch
def get_batch(data,block_size,batch_size):
    ix=torch.randint(len(data)-block_size,(batch_size,))

    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1]for i in ix])
    
    return x,y
 
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print("Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

