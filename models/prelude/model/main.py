import torch
from torch.nn import Module
from torch.utils.data import DataLoader
import mlflow
import random
from datetime import timedelta
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, ProgressColumn, Task, TimeRemainingColumn
from rich.text import Text

class BatchesProcessedColumn(ProgressColumn):
    """Renders completed batches / total batches."""

    def render(self, task: Task) -> Text:
        total = int(task.total) if task.total is not None else "--"
        return Text(f"{int(task.completed)}/{total}", style="progress.download")

class CustomTimeColumn(ProgressColumn):
    """Renders elapsed time and remaining time as 'HH:MM:SS • HH:MM:SS'."""

    def render(self, task: Task) -> Text:
        elapsed = timedelta(seconds=int(task.elapsed)) if task.elapsed is not None else timedelta(0)
        remaining = timedelta(seconds=int(task.time_remaining)) if task.time_remaining is not None else "--:--:--"
        return Text(f"{str(elapsed)} • {str(remaining)}", style="progress.elapsed")

class ProcessingSpeedColumn(ProgressColumn):
    """Renders processing speed (it/s)."""

    def render(self, task: Task) -> Text:
        if task.speed is None:
            return Text("?", style="progress.data.speed")
        return Text(f"{task.speed:.2f}it/s", style="progress.data.speed")

class MetricsTextColumn(ProgressColumn):
    """Renders metrics logged with prog_bar=True."""

    def render(self, task: Task) -> Text:
        metrics = task.fields.get("metrics", {})
        if not metrics:
            return Text("")
        metrics_str = " • ".join([f"{name}: {value:.4f}" if not isinstance(value, (int, str)) else f"{name}: {value}" for name, value in sorted(metrics.items())])
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
    )

class dummyLightning(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def training_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
        )
    
    def validation_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )
    
    def optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, step / (1e-10 + self.config.warmup_steps)))
        return {'optimizer': optimizer, 'scheduler': scheduler}
    
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
        self.forward = torch.autocast(device_type=self.config.device, dtype=torch.bfloat16)(torch.compile(self.forward))
    
    def _iteration(self, dataloader, progress, epoch, num_epochs, train=True):
        if train:
            self.train()
        else:
            self.eval()
        task = progress.add_task(
            f"{'Training' if train else 'Validation'} Epoch {epoch + 1}/{num_epochs}",
            total=len(dataloader),
            metrics={}
        )
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = [item.to(self.config.device) for item in batch]
            elif isinstance(batch, dict):
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
            else:
                batch = batch.to(self.config.device)
            
            self.optimizer.zero_grad()
            outputs = self.step(batch)
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs
            if train:
                loss.backward()
                self.optimizer_step()
            if random.randint(0, 9) == 0:
                self.log(f'{"train" if train else "val"}/loss', loss.item())
            self.global_step += 1
            progress.update(task, advance=1, metrics=self.prog_bar_metrics)

    def fit(self):
        mlflow.set_experiment(self.__class__.__name__)
        self.to(self.config.device)
        self.activate()
        torch.set_float32_matmul_precision('medium')
        
        train_dataloader = self.training_dataloader()
        val_dataloader = self.validation_dataloader()
        
        self.global_step = 0
        progress = create_progress_bar()
        with progress:
            self.prog_bar_metrics = {}
            for epoch in range(self.config.epochs):
                self._iteration(train_dataloader, progress, epoch, self.config.epochs, train=True)
                self._iteration(val_dataloader, progress, epoch, self.config.epochs, train=False)
    
    def log(self, name, value, prog_bar=True, logger=True):
        if logger:
            mlflow.log_metric(name, value, step=self.global_step)
        if prog_bar:
            self.prog_bar_metrics[name] = value