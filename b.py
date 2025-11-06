import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset
from functorch.experimental.control_flow import map
import numpy as np
from einops import rearrange
from lightning import LightningModule, LightningDataModule
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import Callback, RichProgressBar, ModelCheckpoint

def c_ret(x): # of shape f t i
    x = x[4].to('cuda')
    ret1d = torch.roll(x, 1, dims=0) / x
    nnan = ret1d[ret1d.isnan().logical_not()]
    ret1d = torch.clamp(ret1d, nnan.quantile(0.01), nnan.quantile(0.99))
    m = ret1d.nanmean(dim=-1)
    v = ret1d.square().nanmean(dim=-1) - m.square()
    ret1d = (ret1d - m.unsqueeze(-1)) / v.sqrt().unsqueeze(-1)
    return ret1d.cpu()


class mod(LightningModule):
    def __init__(self, n_feat=234):
        super().__init__()
        self.n1 = nn.LayerNorm(n_feat)
        self.ff1 = nn.Conv1d(n_feat, 256, 1)

        self.n2 = nn.LayerNorm(n_feat)
        self.ff2 = nn.Conv1d(256, 256, 1)

        self.ff3 = nn.Conv1d(512, 1, 1)

    def forward(self, x):
        f, t, i = x.shape
        breakpoint()
        x = self.n1(rearrange(x, 'f t i -> t i f'))
        x = self.ff1(rearrange(x, 't i f -> t f i'))
        
        l = F.silu(x).log1p()
        l = self.n2(rearrange(l, 't f i -> t i f'))
        l = self.ff2(rearrange(l, 't i f -> t f i'))
        l = F.softmax(rearrange(l, 't f i -> f (t i)'), dim=1)

        x = torch.concat(
            rearrange(x, 't f i -> f t i'), 
            rearrange(l, 'f (t i) -> f t i', i=i),
        )

        y = self.ff3(rearrange(x, 'f t i -> t f i')).squeeze(1)
        return y

    def step(self, x):
        y = self(x[0])
        loss = F.huber_loss(x[1][1:, :], y[:-1, :])
        return loss
    
    def training_step(self, batch, batch_id):
        breakpoint()
        return self.step(self, batch)

    def validation_step(self, batch, batch_id):
        breakpoint()
        return self.step(self, batch)

    def configure_optimizers(self):
        return AdamW(
            [
                {'params': self.n1.parameters(), 'weight_decay': 0.0},
                {'params': [p for n, p in self.named_parameters() if 'n1' not in n]}
            ],
            lr=3e-4,
            betas=(0.9, 0.95),
        )

class dat(Dataset):
    def __init__(self, c, p, v):
        self.ret = c_ret(p)[4]
        self.c= c
        self.v = v
    
    def __len__(self):
        return len(self.c) - 1

    def __getitem__(self, idx):
        return self.c[idx], self.ret[idx], self.v[idx]
    
def datasets():
    c = torch.load('company_basics.pt')
    p = torch.load('prices.pt')
    v = torch.load('valid_mask.pt')

    o = 3200
    return dat(c[:,:o,:], p[:,:o,:], v[:o, :]), dat(c[:,o:,:], p[:,o:,:], v[o:, :])

def main():
    trainer = Trainer(
        max_epochs=42,
        gradient_clip_val=1.,
        callbacks=[
            RichProgressBar(),
            ModelCheckpoint(dirpath="./ml-runs/models/", save_top_k=2, monitor="val/loss"),
        ],
        logger=MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./ml-runs", artifact_location='./ml-runs/artifacts/'),
    )

    model = mod()
    data = LightningDataModule.from_datasets(*datasets(), batch_size=3072, num_workers=16)

    trainer.fit(model, data)
    breakpoint()

main()