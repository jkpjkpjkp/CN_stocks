import torch
import torch.distributed as dist
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
import mlflow
import random
import os
from datetime import timedelta
from rich.progress import Progress, TextColumn, BarColumn, ProgressColumn, Task
from rich.text import Text
from dataclasses import dataclass


class BatchesProcessedColumn(ProgressColumn):
    """completed batches / total batches."""
    def render(self, task: Task) -> Text:
        total = int(task.total) if task.total is not None else "--"
        return Text(f"{int(task.completed)}/{total}", style="progress.download")


class CustomTimeColumn(ProgressColumn):
    """elapsed time and remaining time as 'HH:MM:SS • HH:MM:SS'."""
    def render(self, task: Task) -> Text:
        elapsed = timedelta(seconds=int(task.elapsed)) if task.elapsed is not None else timedelta(0)
        remaining = timedelta(seconds=int(task.time_remaining)) if task.time_remaining is not None else "--:--:--"
        return Text(f"{str(elapsed)} • {str(remaining)}", style="progress.elapsed")


class ProcessingSpeedColumn(ProgressColumn):
    """processing speed (it/s)."""
    def render(self, task: Task) -> Text:
        if task.speed is None:
            return Text("?", style="progress.data.speed")
        return Text(f"{task.speed:.2f}it/s", style="progress.data.speed")


class MetricsTextColumn(ProgressColumn):
    """metrics logged with prog_bar=True."""
    def render(self, task: Task) -> Text:
        metrics = task.fields.get("metrics", {})
        if not metrics:
            return Text("")

        # Format metrics with special handling for different types
        formatted_metrics = []
        for name, value in sorted(metrics.items()):
            if isinstance(value, float):
                formatted_metrics.append(f"{name}: {value:.4f}")
            elif isinstance(value, (int, str)):
                formatted_metrics.append(f"{name}: {value}")

        metrics_str = " • ".join(formatted_metrics)
        return Text(metrics_str, style="progress.data.speed")


def create_progress_bar():
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        BatchesProcessedColumn(),
        CustomTimeColumn(),
        ProcessingSpeedColumn(),
        MetricsTextColumn(),
        expand=True,
        transient=False,  # Keep progress bars visible after completion
    )


class dummyLightning(Module):
    def __init__(self, config):
        super().__init__()
        assert config
        self.config = config
        os.environ['MASTER_ADDR'] = config.master_addr
        os.environ['MASTER_PORT'] = config.port

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            try:
                return vars(self.config)[name]
            except KeyError:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, step / (1e-10 + self.warmup_steps)))
        return {'optimizer': optimizer, 'scheduler': scheduler}

    def get_dataloader(self, dataset, shuffle=False):
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=True,
        )
        # When using DistributedSampler, shuffle must be False in DataLoader
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    @property
    def train_dataloader(self):
        return self.get_dataloader(self.train_dataset, shuffle=True)

    @property
    def val_dataloader(self):
        return self.get_dataloader(self.val_dataset, shuffle=False)

    def setup_ddp(self):
        """Initialize DDP - always use DDP even for single GPU."""
        if not dist.is_initialized():
            self._init_distributed()
        self.to(self.device)
        return DDP(self, device_ids=[self.local_rank], output_device=self.local_rank)

    def _init_distributed(self):
        """Initialize distributed environment without DDP wrapper."""
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))

        torch.cuda.set_device(self.local_rank)
        self.device = f'cuda:{self.local_rank}'

        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                rank=self.rank,
                world_size=self.world_size,
                device_id=torch.device(f'cuda:{self.local_rank}')
            )

    def is_root(self):
        return self.rank == 0

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        if hasattr(self, 'scheduler') and self.scheduler:
            self.scheduler.step()

        for name, module in self.named_children():
            if isinstance(module, dummyLightning):
                module.optimizer_step()

    def _activate(self):
        opt = self.optimizers()
        if isinstance(opt, dict):
            self.optimizer = opt['optimizer']
            self.scheduler = opt.get('scheduler', None)
        else:
            self.optimizer = opt

        for name, module in self.named_children():
            if isinstance(module, dummyLightning):
                module._activate()

    def activate(self):
        self._activate()
        for name, param in self.named_parameters():
            if name[-5:] == '.bias':
                param.data.zero_()
        if not self.no_compile:
            self.forward = torch.autocast(
                device_type=self.device, dtype=torch.bfloat16
            )(torch.compile(self.forward))

    def _iteration(self, ddp_model, dataloader, progress, epoch):
        train = self.is_train
        num_epochs = self.epochs
        if train:
            self.train()
        else:
            self.eval()

        # Initialize epoch loss tracking
        epoch_losses = []

        task = progress.add_task(
            f"{'Train' if train else 'Val'} {epoch + 1}/{num_epochs}",
            total=len(dataloader),
            metrics={}
        )

        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                batch = [item.to(self.device) for item in batch]
            elif isinstance(batch, dict):
                batch = {k: v.to(self.device) for k, v in batch.items()}
            else:
                batch = batch.to(self.device)

            self.optimizer.zero_grad()
            outputs = ddp_model.module.step(batch) if hasattr(ddp_model, 'module') else self.step(batch)
            loss = outputs['loss']

            if train:
                loss.backward()
                clip_grad_norm_(self.parameters(), self.grad_clip)
                self.optimizer_step()

            epoch_losses.append(loss.item())

            if random.randint(0, 9) == 0 or batch_idx == len(dataloader) - 1:
                for key, value in outputs.items():
                    self.log(key, value)

            self.global_step += 1
            progress.update(task, advance=1, metrics=self.prog_bar_metrics)
        epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
        return epoch_avg_loss

    def fit(self):
        mlflow.set_experiment(self.__class__.__name__)
        ddp_model = self.setup_ddp()
        self.activate()
        torch.set_float32_matmul_precision('medium')

        td = self.train_dataloader
        vd = self.val_dataloader

        best_val_loss = float('inf')

        self.global_step = 0
        progress = create_progress_bar()
        with progress:
            for epoch in range(self.epochs):
                self.prog_bar_metrics = {}
                self.is_train = True
                train_epoch_loss = self._iteration(ddp_model, td, progress, epoch)
                self.log('epoch_loss', train_epoch_loss)
                self.prog_bar_metrics = {}
                self.is_train = False
                val_epoch_loss = self._iteration(ddp_model, vd, progress, epoch)
                self.log('epoch_loss', val_epoch_loss)

                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    self.save_checkpoint(f'./.checkpoints/{self.__class__.__name__}/_epoch={epoch}_step={self.global_step}_loss={val_epoch_loss:.4f}.pt', epoch, val_epoch_loss, train_epoch_loss)

                progress.console.print(f"Epoch {epoch + 1}/{self.epochs} completed - Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")

        dist.destroy_process_group()

    def log(self, name, value):
        name = ('train' if self.is_train else 'val') + '/' + name
        mlflow.log_metric(name, value, step=self.global_step)
        self.prog_bar_metrics[name] = value

    def save_checkpoint(self, path, epoch, val_epoch_loss, train_epoch_loss):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save({
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_epoch_loss': val_epoch_loss,
            'train_epoch_loss': train_epoch_loss,
            'config': self.config,
        }, path)

    @classmethod
    def load_checkpoint(cls, path):
        checkpoint = torch.load(path, weights_only=False)
        config = checkpoint['config']
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.global_step = checkpoint['global_step']
        return model


@dataclass
class dummyConfig:
    def __post_init__(self):
        # Dataloader
        if self.num_workers is None:
            self.num_workers = os.cpu_count()
