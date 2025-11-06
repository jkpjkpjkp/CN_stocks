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

torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision('high')

class NeierNorm(nn.LayerNorm):
    def forward(self, x):
        x = x - x.nanmean(dim=-1).unsqueeze(-1)
        x = x / (x.square().nanmean(dim=-1).sqrt().unsqueeze(-1) + self.eps)
        return x * self.weight + self.bias

def c_ret(x): # of shape f t i
    x = x[4].to('cuda')
    ret1d = torch.roll(x, 1, dims=0) / x
    nnan = ret1d[ret1d.isnan().logical_not()]
    ret1d = torch.clamp(ret1d, nnan.quantile(0.01), nnan.quantile(0.99))
    with torch.no_grad():
        norm = NeierNorm(5327)
        norm.to('cuda')
        norm.requires_grad_(False)
        ret1d = norm(ret1d).detach().cpu()
        ret1d.requires_grad_(False)
        return ret1d


class mod(LightningModule):
    def __init__(self, n_feat=234, medi=16):
        super().__init__()
        # self.automatic_optimization = False
        self.n1 = nn.BatchNorm1d(n_feat)
        self.ff1 = nn.Conv1d(n_feat, medi, 1)

        self.n2 = nn.BatchNorm1d(medi)
        self.ff2 = nn.Conv1d(medi, medi, 1)

        self.ff3 = nn.Conv1d(medi * 2, 1, 1)

    def forward(self, x):
        f, t, i = x.shape
        cnt = x.isnan().sum()
        x = self.n1(x)
        assert x.isnan().sum() == cnt
        x2 = self.ff1(rearrange(x.nan_to_num(0), 't f i -> t f i'))

        assert x2.isnan().logical_not().all()
        x = x2
        l = F.silu(x).log1p()
        l = self.n2(l)
        assert l.isnan().logical_not().all()
        l = self.ff2(l)
        assert l.isnan().logical_not().all()
        l = F.softmax(rearrange(l, 't f i -> f (t i)'), dim=1)
        assert l.isnan().logical_not().all()

        x = torch.concat((
            rearrange(x, 't f i -> f t i'), 
            rearrange(l, 'f (t i) -> f t i', i=i),
        ))

        y = self.ff3(rearrange(x, 'f t i -> t f i')).squeeze(1)
        return y

    def step(self, x):
        y = self(x[0].nan_to_num(0))
        assert y.isnan().logical_not().all()
        mask = x[1].isnan().logical_not()
        assert (mask.sum(dim=1) > 0).all()
        loss = F.huber_loss(x[1][mask].detach(), y[mask])
        return loss
    
    def training_step(self, batch, batch_id):
        loss = self.step(batch)
        assert loss.isfinite()
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_id):
        loss = self.step(batch)
        self.log('val/loss', loss)
        return loss

    def configure_optimizers(self):
        n1_params = list(self.n1.parameters())
        n1_param_ids = set(id(p) for p in n1_params)
        
        other_params = [p for p in self.parameters() if id(p) not in n1_param_ids]
        
        return AdamW(
            # [
            #     {'params': n1_params, 'weight_decay': 0.0},
            #     {'params': other_params, 'weight_decay': 0.01}  # Explicit weight decay
            # ],
            self.parameters(),
            lr=3e-4,
            betas=(0.9, 0.95),
        )

class dat(Dataset):
    def __init__(self, c, p, v):
        self.ret = c_ret(p)
        self.c= c
        self.v = v

        self.ret.requires_grad_(False)
        
    def __len__(self):
        return self.c.shape[1] - 1

    def __getitem__(self, idx):
        return self.c[:, idx, :], self.ret[idx, :], self.v[idx, :]
    
def datasets():
    c = torch.load('company_basics.pt')
    p = torch.load('prices.pt')
    v = torch.load('valid_mask.pt')

    c.requires_grad_(False)
    p.requires_grad_(False)
    v.requires_grad_(False)

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
        log_every_n_steps=1,
    )

    model = mod()
    data = LightningDataModule.from_datasets(*datasets(), batch_size=2, num_workers=16)

    trainer.fit(model, data)
    breakpoint()

main()