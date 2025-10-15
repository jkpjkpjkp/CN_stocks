import torch
torch.manual_seed(42)
torch.cuda.manual_seed(42)

import random
random.seed(42)

import numpy as np
np.random.seed(42)


from torch import nn
import math



from model import Transformer, ModelArgs

model = Transformer(args).to('cuda')



from tqdm import tqdm
def train(
    model,
    dataloader,
    lr,
    n_epochs,
    f_loss
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to('cuda').train()
    for epoch in range(n_epochs):
        for x in tqdm(dataloader):
            x = x.to('cuda')

            scaling_factors = x[:, -30].unsqueeze(1)  # Get last position and add dimension for broadcasting
            x = x / scaling_factors

            y = x[:,-1]
            # print(y)
            x = x[:,:-30]

            # x = torch.ones_like(x)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = f_loss(y_hat, y)
            loss.backward()
            optimizer.step()
            break
        print(f"Epoch {epoch} loss: {loss.item()}")


def val(
    model,
    dataloader,
    f_loss
):
    model.to('cuda').eval()
    with torch.no_grad():
        for x in tqdm(dataloader):
            x = x.to('cuda')

            scaling_factors = x[:, -2].unsqueeze(1)  # Get last position and add dimension for broadcasting
            x = x / scaling_factors

            y = x[:,-1]
            x = x[:,:-1]
            y_hat = model(x)
            loss = f_loss(y_hat, y)

            print(x, y, y_hat)
            breakpoint()
        print(f"Val loss: {loss.item()}")

# In[48]:


import polars as pl

df = pl.scan_parquet('../data/a_30min.pq')
df = df.drop(['open', 'high', 'low', 'close', 'volume'])

df = df.rename({
    'open_post': 'open',
    'high_post': 'high',
    'low_post': 'low',
    'close_post': 'close',
    'volume_post': 'volume',
})

# In[49]:


from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# val(
#     model=model,
#     dataloader=DataLoader(data, batch_size=1, shuffle=True),
#     f_loss=F.huber_loss
# )

# In[ ]:


train(
    model=model,
    dataloader=DataLoader(data, batch_size=100, shuffle=True),
    lr=3e-4,
    n_epochs=100,
    f_loss=F.huber_loss
)
