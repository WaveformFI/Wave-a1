from dataclasses import dataclass   

@dataclass
class ModelConfig:
    base_model: str = "microsoft/deberta-v3-small"
    use_pretrained: bool = True
    max_length: int = 512
    batch_size: int = 12
    learning_rate: float = 1e-5
    num_epochs: int = 5
    num_heads: int = 4
    num_layers: int = 3
    d_ff: int = 1024
    warmup_steps: int = 50
    num_sentiment_classes: int = 3  # positive, neutral, negative
    num_market_classes: int = 5     # strong bullish to strong bearish
    seed: int = 42
    dropout_rate: float = 0.25
    confidence_threshold: float = 0.75
    hidden_size: int = 256
    model_path: str = 'model.pth'
    vocab_size: int = 128000
    paddinx_idx: int = 0