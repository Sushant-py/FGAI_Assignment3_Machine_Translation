"""
transformer_scratch.py
Complete PyTorch implementation of the Basic Transformer (Encoder-Decoder Architecture)
from scratch based on 'Attention Is All You Need' (Vaswani et al., 2017).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding."""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-Head Self and Cross Attention Mechanism."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)

        # 1. Linear projections: (batch_size, seq_len, d_model) -> (batch_size, n_heads, seq_len, d_k)
        q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2. Scaled Dot-Product Attention: Q * K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # Masking with large negative float
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 3. Multiply weights by V
        context = torch.matmul(attention_weights, v)  # (batch_size, n_heads, seq_len, d_k)
        
        # 4. Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(context)
        return output, attention_weights


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Single Transformer Encoder Layer."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN / Post-LN: Standard Vaswani uses Post-LN (SubLayer(x) + x -> LayerNorm)
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x


class DecoderLayer(nn.Module):
    """Single Transformer Decoder Layer with Masked Self-Attention and Cross-Attention."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        enc_output: torch.Tensor, 
        tgt_mask: Optional[torch.Tensor] = None, 
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 1. Masked Self-Attention (Causal)
        self_attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))
        
        # 2. Cross-Attention over Encoder outputs (Query from decoder, Key/Value from encoder)
        cross_attn_out, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))
        
        # 3. Position-wise Feed Forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_out))
        return x


class BasicTransformer(nn.Module):
    """Full Encoder-Decoder Transformer Architecture."""
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_decoder_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        # src: (batch_size, src_len) -> mask: (batch_size, 1, 1, src_len)
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        # tgt: (batch_size, tgt_len)
        batch_size, tgt_len = tgt.size()
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, tgt_len)
        # Causal triangular look-ahead mask
        causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = tgt_pad_mask & causal_mask.unsqueeze(0).unsqueeze(1)
        return tgt_mask

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(
        self, 
        tgt: torch.Tensor, 
        enc_output: torch.Tensor, 
        tgt_mask: torch.Tensor, 
        src_mask: torch.Tensor
    ) -> torch.Tensor:
        x = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder_layers:
            x = layer(x, enc_output, tgt_mask, src_mask)
        return x

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, tgt_mask, src_mask)
        output = self.fc_out(dec_output)
        return output

    def count_parameters(self) -> int:
        """Return total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def greedy_decode(
    model: BasicTransformer,
    src: torch.Tensor,
    sos_idx: int,
    eos_idx: int,
    max_len: int = 50,
    device: str = "cpu"
) -> torch.Tensor:
    """Greedy autoregressive decoding for basic transformer."""
    model.eval()
    src = src.to(device)
    src_mask = model.make_src_mask(src)
    
    with torch.no_grad():
        enc_output = model.encode(src, src_mask)
        
    tgt = torch.tensor([[sos_idx]], dtype=torch.long, device=device)
    
    for _ in range(max_len):
        tgt_mask = model.make_tgt_mask(tgt)
        with torch.no_grad():
            dec_output = model.decode(tgt, enc_output, tgt_mask, src_mask)
            logits = model.fc_out(dec_output[:, -1, :])
            next_token = logits.argmax(dim=-1).item()
            
        tgt = torch.cat([tgt, torch.tensor([[next_token]], device=device)], dim=1)
        if next_token == eos_idx:
            break
            
    return tgt.squeeze(0)


if __name__ == "__main__":
    src_vocab, tgt_vocab = 500, 600
    model = BasicTransformer(src_vocab, tgt_vocab, d_model=128, n_heads=4, num_encoder_layers=2, num_decoder_layers=2)
    print(f"Basic Transformer created with {model.count_parameters():,} trainable parameters.")
    dummy_src = torch.randint(1, src_vocab, (2, 10))
    dummy_tgt = torch.randint(1, tgt_vocab, (2, 8))
    out = model(dummy_src, dummy_tgt)
    print("Forward pass successful. Output shape:", out.shape)
