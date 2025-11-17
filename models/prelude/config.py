from transformers import PretrainedConfig

class transformerConfig(PretrainedConfig):
    model_type = "t1p30"
    layers=7
    hidden_size=256
    intermediate_ratio=2.5
    num_heads=4
    lr=3e-3
    weight_decay=3e-4
    norm='LayerNorm'
    device='cuda'
    warmup_steps=0
    batch_warmup=0
    seq_len=119
    vocab_size=128
    mlflow_dir='./.checkpoint/mlruns'
    
    batch_size=128
    num_workers=16
    epochs=32

    def __init__(self, intermediate_ratio=1.5, **kwargs):
        self.intermediate_ratio = intermediate_ratio
        super().__init__(**kwargs)
        self.intermediate_size = int(self.hidden_size * self.intermediate_ratio)
