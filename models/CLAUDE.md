use `uv run -m`

data is stored in /home/jkp/h/data/a_1min.pq

when implementing attention, use modern best practices of F.scaled_dot_product_attention; avoid using legacy classes (e.g. nn.MultiheadAttention, nn.TransformerDecoderLayer)
