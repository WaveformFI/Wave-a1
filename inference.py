from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import ModelConfig
from a1 import WaveformA1

app = FastAPI()

config = ModelConfig()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=False)

model = WaveformA1(config).to(device)
    
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

class PredictionResponse(BaseModel):
    sentiment: str
    classification: str
    confidence_sentiment: float
    confidence_classification: float

def decode_sentiment(idx: int) -> str:
    sentiment_map = {0: 'positive', 1: 'neutral', 2: 'negative'}
    return sentiment_map[idx]

def decode_market(idx: int) -> str:
    market_map = {
        0: 'strong bullish',
        1: 'bullish',
        2: 'neutral',
        3: 'bearish',
        4: 'strong bearish'
    }
    return market_map[idx]

@app.get("/predict", response_model=PredictionResponse)
async def predict(text: str):
    try:
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )

        inputs = {k: v.to(device) for k, v in inputs.items() 
                 if k in ['input_ids', 'attention_mask']}
    
        with torch.no_grad():
            sentiment_logits, market_logits = model(**inputs)
            
            sentiment_probs = F.softmax(sentiment_logits, dim=1)
            market_probs = F.softmax(market_logits, dim=1)
            
            sentiment_conf, sentiment_pred = torch.max(sentiment_probs, dim=1)
            market_conf, market_pred = torch.max(market_probs, dim=1)
        
        return {
            "sentiment": decode_sentiment(sentiment_pred.item()),
            "classification": decode_market(market_pred.item()),
            "confidence_sentiment": sentiment_conf.item(),
            "confidence_classification": market_conf.item()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)