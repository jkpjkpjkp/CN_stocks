from torch.nn import Module
import mlflow

class dummyLightning(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.scheduler:
            self.scheduler.step()
        
        for name, module in self.named_children():
            if isinstance(module, dummyLightning):
                module.optimizer_step()
    
    def activate(self):
        opt = self.configure_optimizers()
        if isinstance(opt, dict):
            self.optimizer = opt['optimizer']
            self.scheduler = opt.get('scheduler', None)
        else:
            self.optimizer = opt
        
        for name, module in self.named_children():
            if isinstance(module, dummyLightning):
                module.activate()