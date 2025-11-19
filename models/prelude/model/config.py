from transformers import PretrainedConfig

class transformerConfig(PretrainedConfig):
    model_type = "t1p30"
    num_layers=7
    hidden_dim=16
    intermediate_ratio=2.5
    num_heads=2
    lr=3e-4
    norm='LayerNorm'
    device='cuda'
    warmup_steps=1000
    batch_warmup=0
    seq_len=119
    vocab_size=128
    mlflow_dir='./.checkpoint/mlruns'
    
    batch_size=128
    num_workers=2
    epochs=32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.intermediate_size = int(self.hidden_dim * self.intermediate_ratio)
        self.intermediate_dim = self.intermediate_size
        self.head_dim = self.hidden_dim // self.num_heads
