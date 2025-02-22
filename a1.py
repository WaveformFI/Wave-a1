import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, f"d_model {d_model} must be divisible by num_heads {num_heads}"
        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.head_dim: int = d_model // num_heads
        
        self.q_linear: nn.Linear = nn.Linear(d_model, d_model)
        self.k_linear: nn.Linear = nn.Linear(d_model, d_model)
        self.v_linear: nn.Linear = nn.Linear(d_model, d_model)
        
        self.out_linear: nn.Linear = nn.Linear(d_model, d_model)
        
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.scale: float = math.sqrt(self.head_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size: int = query.shape[0]
        # At some point, we experimented with rotary embeddings, but it didn't seem to impact the model in a meaningful way
        # the current implementation could (should?) be replaced with n.MultiheadAttention
        Q: torch.Tensor = self.q_linear(query)
        K: torch.Tensor = self.k_linear(key)
        V: torch.Tensor = self.v_linear(value)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores: torch.Tensor = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights: torch.Tensor = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context: torch.Tensor = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output: torch.Tensor = self.out_linear(context)
        return output, attention_weights

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1: nn.Linear = nn.Linear(d_model, d_ff)
        self.linear2: nn.Linear = nn.Linear(d_ff, d_model)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        position: torch.Tensor = torch.arange(max_seq_length).unsqueeze(1)
        div_term: torch.Tensor = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe: torch.Tensor = torch.zeros(1, max_seq_length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term[: d_model // 2])
        pe[0, :, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len: int = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(f'Sequence length {seq_len} exceeds maximum {self.pe.size(1)}')
        return self.dropout(x + self.pe[:, :seq_len, :])

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn: Attention = Attention(d_model, num_heads, dropout)
        self.ff: PositionwiseFeedForward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1: nn.LayerNorm = nn.LayerNorm(d_model)
        self.norm2: nn.LayerNorm = nn.LayerNorm(d_model)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output: torch.Tensor = self.ff(x)
        return self.norm2(x + self.dropout(ff_output))

class WaveformA1(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        d_ff: int = 1024,
        max_seq_length: int = 512,
        num_sentiment_classes: int = 3,
        num_market_classes: int = 5,
        dropout: float = 0.1,
        base_model_name: Optional[str] = None,
        vocab_size: int = 128000,
        padding_idx: int = 0
    ) -> None:
        super().__init__()
        self.use_base: bool = base_model_name is not None
        print(f"Using base model: {base_model_name if self.use_base else 'None'}")
        if self.use_base:
            self.base: nn.Module = AutoModel.from_pretrained(base_model_name)  
            base_hidden: int = self.base.config.hidden_size
            self.proj: nn.Module = nn.Linear(base_hidden, d_model) if base_hidden != d_model else nn.Identity()
        else:
            # During training, we observed similar, or better, results than using a pre-trained model, however, during real world tests, we often
            # see that the (supervised) model doesn't generalize well to unseen data. We need to run more tests on the semi supervised pipeline to 
            # confirm whether the trend continues.
            self.embedding: nn.Embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=d_model,
                padding_idx=padding_idx
            )
            
        self.positional_encoding: PositionalEncoding = PositionalEncoding(d_model, max_seq_length, dropout)
        self.encoder_layers: nn.ModuleList = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.sentiment_classifier: nn.Sequential = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_sentiment_classes)
        )
        self.market_classifier: nn.Sequential = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_market_classes)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_base:
            outputs: torch.Tensor = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            pooled_output: torch.Tensor = outputs[:, 0]  # use first token as pooled representation
            x: torch.Tensor = self.proj(pooled_output).unsqueeze(1)
            x = self.positional_encoding(x)
            for layer in self.encoder_layers:
                x = layer(x)
            x = x.squeeze(1)
        else:
            x = self.embedding(input_ids)
            x = self.positional_encoding(x)
            mask: Optional[torch.Tensor] = None
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(1).unsqueeze(2)
            for layer in self.encoder_layers:
                x = layer(x, mask)
            x = x.mean(dim=1)
        sentiment_logits: torch.Tensor = self.sentiment_classifier(x)
        market_logits: torch.Tensor = self.market_classifier(x)
        return sentiment_logits, market_logits