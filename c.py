#!/usr/bin/env python
# coding: utf-8

# 2. Set up the end-to-end training/evaluation skeleton + get dumb baselines

# verify loss @ init. Verify that your loss starts at the correct loss value. E.g. if you initialize your final layer correctly you should measure -log(1/n_classes) on a softmax at initialization. The same default values can be derived for L2 regression, Huber losses, etc.
# init well. Initialize the final layer weights correctly. E.g. if you are regressing some values that have a mean of 50 then initialize the final bias to 50. If you have an imbalanced dataset of a ratio 1:10 of positives:negatives, set the bias on your logits such that your network predicts probability of 0.1 at initialization. Setting these correctly will speed up convergence and eliminate “hockey stick” loss curves where in the first few iteration your network is basically just learning the bias.
# human baseline. Monitor metrics other than loss that are human interpretable and checkable (e.g. accuracy). Whenever possible evaluate your own (human) accuracy and compare to it. Alternatively, annotate the test data twice and for each example treat one annotation as prediction and the second as ground truth.
# input-indepent baseline. Train an input-independent baseline, (e.g. easiest is to just set all your inputs to zero). This should perform worse than when you actually plug in your data without zeroing it out. Does it? i.e. does your model learn to extract any information out of the input at all?
# overfit one batch. Overfit a single batch of only a few examples (e.g. as little as two). To do so we increase the capacity of our model (e.g. add layers or filters) and verify that we can reach the lowest achievable loss (e.g. zero). I also like to visualize in the same plot both the label and the prediction and ensure that they end up aligning perfectly once we reach the minimum loss. If they do not, there is a bug somewhere and we cannot continue to the next stage.
# verify decreasing training loss. At this stage you will hopefully be underfitting on your dataset because you’re working with a toy model. Try to increase its capacity just a bit. Did your training loss go down as it should?
# visualize just before the net. The unambiguously correct place to visualize your data is immediately before your y_hat = model(x) (or sess.run in tf). That is - you want to visualize exactly what goes into your network, decoding that raw tensor of data and labels into visualizations. This is the only “source of truth”. I can’t count the number of times this has saved me and revealed problems in data preprocessing and augmentation.
# visualize prediction dynamics. I like to visualize model predictions on a fixed test batch during the course of training. The “dynamics” of how these predictions move will give you incredibly good intuition for how the training progresses. Many times it is possible to feel the network “struggle” to fit your data if it wiggles too much in some way, revealing instabilities. Very low or very high learning rates are also easily noticeable in the amount of jitter.
# use backprop to chart dependencies. Your deep learning code will often contain complicated, vectorized, and broadcasted operations. A relatively common bug I’ve come across a few times is that people get this wrong (e.g. they use view instead of transpose/permute somewhere) and inadvertently mix information across the batch dimension. It is a depressing fact that your network will typically still train okay because it will learn to ignore data from the other examples. One way to debug this (and other related problems) is to set the loss to be something trivial like the sum of all outputs of example i, run the backward pass all the way to the input, and ensure that you get a non-zero gradient only on the i-th input. The same strategy can be used to e.g. ensure that your autoregressive model at time t only depends on 1..t-1. More generally, gradients give you information about what depends on what in your network, which can be useful for debugging.
# generalize a special case. This is a bit more of a general coding tip but I’ve often seen people create bugs when they bite off more than they can chew, writing a relatively general functionality from scratch. I like to write a very specific function to what I’m doing right now, get that to work, and then generalize it later making sure that I get the same result. Often this applies to vectorizing code, where I almost always write out the fully loopy version first and only then transform it to vectorized code one loop at a time.

# In[45]:


import torch
torch.manual_seed(42)
torch.cuda.manual_seed(42)

import random
random.seed(42)

import numpy as np
np.random.seed(42)

# In[46]:


import torch
from torch import nn
import math


class BatchedDummyModule(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        val = torch.randn(int(math.log2(window_size)) + 1)
        self.weights = nn.Parameter(val / val.sum())
        # self.weights = nn.Parameter(val)
    
    def forward(self, series):
        assert series.shape[1] == self.window_size, f'{series.shape} {self.window_size}'
        
        device = series.device
        batch_size = series.shape[0]
        
        indices = torch.arange(self.window_size, device=device)
        scales = torch.floor(torch.log2(indices + 1)).long()
        factors = 2.0 ** scales
        mul = self.weights[scales] / factors  # Shape: [window_size]
        
        reversed_series = torch.flip(series, [1])  # Shape: [batch_size, window_size]
        
        weighted_series = reversed_series * mul.unsqueeze(0)
        
        return torch.sum(weighted_series, dim=1)

window_size = 512
model = BatchedDummyModule(window_size-1).to('cuda')

# In[47]:


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

            scaling_factors = x[:, -2].unsqueeze(1)  # Get last position and add dimension for broadcasting
            x = x / scaling_factors

            y = x[:,-1]
            x = x[:,:-1]

            optimizer.zero_grad()
            y_hat = model(x)
            loss = f_loss(y_hat, y)
            loss.backward()
            optimizer.step()
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
            y = x[:,-1]
            x = x[:,:-1]
            y_hat = model(x)
            loss = f_loss(y_hat, y)
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


from torch.utils.data import Dataset, ConcatDataset
class rollingWindowDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
    
    def __len__(self):
        return self.data.shape[0] - self.window_size + 1

    def __getitem__(self, idx):
        return self.data[idx:idx+self.window_size]
datasets = []
for x in df.collect().group_by('order_book_id'):
    datasets.append(rollingWindowDataset(x[1]['close'].to_torch(), window_size))

    if len(datasets) > 10:
        break
data = ConcatDataset(datasets)

# In[50]:


model.parameters()

# In[ ]:


from torch.utils.data import DataLoader
import torch.nn.functional as F

train(
    model=model,
    dataloader=DataLoader(data, batch_size=2048, shuffle=True),
    lr=3e-4,
    n_epochs=10,
    f_loss=F.mse_loss
)
