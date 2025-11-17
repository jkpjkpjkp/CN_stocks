import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from lightning import LightningModule as Module, LightningDataModule as DataModule
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
import numpy as np
import random
import os
import subprocess

from min.models.quantile_quantized.t1p1 import decoderLayer, loggingMixin, transformerConfig
from ..embeddings.quantile import quantile_30min as ds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

class t30m(Module):
    def __init__(self, config: transformerConfig):
        super().__init__()
        self.emb1 = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb30 = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([decoderLayer(config) for _ in range(config.layers)])
        
        self.readout = nn.Linear(config.hidden_size, config.vocab_size)

        self.config = config
    
    def forward(self, x1, x30):
        b = x1.shape[0]
        x1 = self.emb1(x1)
        x30 = self.emb30(x30)
        x = x1 + torch.concat((torch.zeros((b, 29, self.config.vocab_size), device=x30.device, dtype=x30.dtype), x30), dim=1)
        
        for layer in self.layers:
            x = layer(x)
        x = self.readout(x)

        return x
    
    def training_step(self, batch, batch_idx, train = 'train'):
        x, y = batch
        y_hat = self(x[:, :-30], y[:, :-30])
        loss = nn.CrossEntropyLoss()(y_hat.view(-1, 128), y[:, 1:].contiguous().view(-1))
        if batch_idx % 10 == 0:
            self.log(f'{train}/loss', loss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        return {
            'loss': loss,
            'logits': y_hat,
        }
    
    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, train='val')
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        def warmup_lambda(step):
            return min(step / self.config.warmup_tokens, 1.0)
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=warmup_lambda
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
         }

def main(checkpoint_dir):
    config = transformerConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_ratio=4,
        layers=6,
        lr=3e-4,
        warmup_tokens=1000,
    )
    model = t30m(config)

    torch.set_float32_matmul_precision('medium')
    data = DataModule.from_datasets(
        ds('../data/train.npy'), 
        ds('../data/eval.npy'), 
        batch_size=2048,
        num_workers=16,
    )
    trainer = Trainer(
        max_epochs=42,
        gradient_clip_val=1.,
        precision="bf16-mixed",
        accelerator="gpu",
        devices=1,
        num_nodes=1,
        callbacks=[
            RichProgressBar(),
            loggingMixin(every_n_steps=100),
            ModelCheckpoint(dirpath=checkpoint_dir, save_top_k=2, monitor="val/loss"),
        ],
        logger=MLFlowLogger(experiment_name="lightning_logs", tracking_uri=f"file:{config.mlflow_dir}", artifact_location=f'{config.mlflow_dir}/artifacts/'),
    )

    trainer.fit(model, data)

if __name__ == '__main__':
    checkpoint_dir = '.checkpoint/t1p30'
    os.makedirs(checkpoint_dir, exist_ok=True)
    subprocess.run(['cp', __file__, checkpoint_dir])
    main(checkpoint_dir)