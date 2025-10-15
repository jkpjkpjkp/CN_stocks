import torch, torch.nn as nn, torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        nheads: int,
        bias=True,
    ):
        super().__init__()
        self.nheads = nheads
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias)
        else:
            self.q_proj = nn.Linear(E_q, E_total, bias=bias)
            self.k_proj = nn.Linear(E_k, E_total, bias=bias)
            self.v_proj = nn.Linear(E_v, E_total, bias=bias)
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out, bias=bias)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads
        self.bias = bias

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal=False,
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (``N``, ``L_q``, ``E_qk``)
            key (torch.Tensor): key of shape (``N``, ``L_kv``, ``E_qk``)
            value (torch.Tensor): value of shape (``N``, ``L_kv``, ``E_v``)
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        # Step 1. Apply input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(
                    self.packed_proj.weight, 3, dim=0
                )
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(
                        self.packed_proj.bias, 3, dim=0
                    )
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    F.linear(query, q_weight, q_bias),
                    F.linear(key, k_weight, k_bias),
                    F.linear(value, v_weight, v_bias),
                )

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention(
            query, key, value, is_causal=is_causal
        )
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output


class RMSNorm(nn.Module):
    def __init__(self, d_model: int):
        self.mean = nn.Parameter(torch.zeros(d_model))
        self.var = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_detach = x.detach()
        x = (x - x_detach.mean(dim=-1, keepdim=True)) / torch.sqrt(x_detach.var(dim=-1, keepdim=True))
        x = x * self.var + self.mean
        return x


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer
    
    Args:
        d_model (int): Dimension of the input embedding
        nheads (int): Number of attention heads
        dim_feedforward (int, optional): Dimension of the feedforward network. Default: 2048
        bias (bool, optional): Whether to use bias in linear layers. Default: True
    """
    def __init__(
        self,
        d_model: int,
        nheads: int,
        dim_feedforward: int = 2048,
        bias: bool = True,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        # Self-attention sublayer components
        self.self_attn = MultiHeadAttention(
            E_q=d_model,
            E_k=d_model,
            E_v=d_model,
            E_total=d_model,
            nheads=nheads,
            bias=bias,
            **factory_kwargs
        )
        self.norm1 = nn.RMSNorm(d_model)  # Prenorm for self-attentioner attention
        
        # Feed-forward sublayer components
        self.ffnn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),  # Modern activation (better than ReLU)
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.RMSNorm(d_model)  # Prenorm for feed-forward
        
        # Initialize weights properly
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters with proper initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  # Better for transformers than default init

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the decoder layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, L, d_model)
                where N = batch size, L = sequence length
            attn_mask (Optional[torch.Tensor]): Attention mask of shape (N, L, L). 
                Default: None
                
        Returns:
            torch.Tensor: Output tensor of shape (N, L, d_model)
        """
        # Self-attention sublayer with prenorm
        x_norm = self.norm1(x)  # Apply norm first (prenorm)
        attn_output = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            is_causal=True  # Decoder uses causal mask to prevent future token access
        )
        x = x + attn_output  # Residual connection
        
        # Feed-forward sublayer with prenorm
        x_norm2 = self.norm2(x)  # Apply norm first (prenorm)
        ff_output = self.ffnn(x_norm2)  # Feed-forward
        x = x + ff_output  # Residual connection
        
        return x

