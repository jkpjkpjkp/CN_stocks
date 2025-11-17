import torch
from torch import nn
from torch.nn import functional as F, Module
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ChainedScheduler
import numpy as np
from einops import rearrange
import random
import matplotlib.pyplot as plt
from transformers import PreTrainedModel, PretrainedConfig
from ..embedding.quantile import quantile_1min as ds

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)



class mha(Module):
    def __init__(self, config: transformerConfig):
        super().__init__()
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)
        device = config.device
        channel_range = torch.arange(0, config.hidden_size, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (10000 ** (channel_range / config.hidden_size))
        t = torch.arange(config.seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        self.cos, self.sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()

        self.num_heads = config.num_heads
        assert config.hidden_size % config.num_heads == 0
        self.head_dim = config.hidden_size // config.num_heads
        self.hidden_size = config.hidden_size

    def forward(self, x):
        q, k, v = torch.chunk(self.qkv(x), 3, -1)
        q = apply_rotary_emb(q, self.cos, self.sin)
        k = apply_rotary_emb(k, self.cos, self.sin)
        b = x.shape[0]
        
        shape = (b, -1, self.head_dim, self.num_heads)
        q = rearrange(q.view(*shape), 'b l d h -> b h l d')
        k = rearrange(k.view(*shape), 'b l d h -> b h l d')
        v = rearrange(v.view(*shape), 'b l d h -> b h l d')

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous()
        y = y.view(b, -1, self.hidden_size)
        y = self.fc2(y)
        y = F.silu(y)
        return y


class tm(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([decoderLayer(config) for _ in range(config.layers)])
        self.readout = nn.Linear(config.hidden_size, config.vocab_size)

        self.config = config
    
    def forward(self, x):
        b = x.shape[0]
        # assert x.shape == (b, 118)
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x)
        x = self.readout(x)

        return x
    
    def training_step(self, batch, batch_idx, train='train'):
        y = self(batch[:, :-1])
        loss = nn.CrossEntropyLoss()(y.view(-1, self.config.vocab_size), batch[:, 1:].contiguous().view(-1))
        if batch_idx % 10 == 0:
            self.log(f'{train}/loss', loss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        idx = random.randint(0, batch.shape[0] - 1)
        return {
            'loss': loss,
            'sample': batch[idx].detach().cpu().numpy(),
            'pred': y[idx].detach().cpu().numpy(),
        }

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, train='val')
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        return {
            'optimizer': optimizer,
        }

class loggingMixin(Callback):
    def __init__(self, every_n_steps):
        super().__init__()
        self.every_n_steps = every_n_steps

    def on_before_optimizer_step(self, model, closure, optimizer):
        layer_norms = grad_norm(closure, norm_type=2)
        self.log_dict(layer_norms, on_step=True, on_epoch=True, logger=True)
    
    # def on_train_batch_end(self, trainer: Trainer, pl_module: Module, outputs, batch, batch_idx: int) -> None:
    #     fig, ax = plt.subplots()
    #     ax.plot(np.arange(118), outputs['sample'][1:], label='gt')
    #     ax.plot(np.arange(118), outputs['pred'], label='pred')
    #     trainer.logger.experiment.log_figure(trainer.logger.run_id, fig, f"train/plt{batch_idx}.png") 
    #     plt.close(fig)
    
    # def on_val_batch_end(self, trainer: Trainer, pl_module: Module, outputs, batch, batch_idx: int) -> None:
    #     fig, ax = plt.subplots()
    #     ax.plot(np.arange(118), outputs['sample'][1:], label='gt')
    #     ax.plot(np.arange(118), outputs['pred'], label='pred')
    #     trainer.logger.experiment.log_figure(trainer.logger.run_id, fig, f"val/plt{batch_idx}.png") 
    #     plt.close(fig)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    config = transformerConfig()
    model = tm(config)
    data = DataModule.from_datasets(ds(config, '../data/train.npy'), ds(config, '../data/eval.npy'), batch_size=4096, num_workers=32)
    trainer = Trainer(
        max_epochs=42,
        gradient_clip_val=1.,
        callbacks=[
            RichProgressBar(), 
            loggingMixin(every_n_steps=20),
            ModelCheckpoint(dirpath="./.checkpoints/mlruns/models/", save_top_k=2, monitor="val/loss"),   # TODO: dirpath
        ],
        logger=MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./.checkpoints/mlruns", artifact_location='./ml-runs/artifacts/'),
    )

    trainer.fit(model, data)