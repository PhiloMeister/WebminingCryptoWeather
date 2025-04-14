from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from typing import List
import os

# Define the data model for incoming request
class Article(BaseModel):
    title: str
    content: str

class ArticlesRequest(BaseModel):
    articles: List[Article]

# Init app
app = FastAPI()

# Load model and tokenizer
MODEL_PATH = os.path.join("model", "training", "bert_sentiment_regression_model")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_sentiment_status(score: float) -> str:
    if score <= -0.6:
        return "Extreme fear"
    elif -0.6 < score <= -0.2:
        return "Fear"
    elif -0.2 < score <= 0.2:
        return "Neutral"
    elif 0.2 < score <= 0.6:
        return "Greed"
    else:
        return "Extreme greed"

@app.post("/predict_sentiment")
async def predict_sentiment(request: ArticlesRequest):
    texts = [article.content for article in request.articles]
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt", max_length=512)
    encodings = {k: v.to(model.device) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
        scores = outputs.logits.squeeze().cpu().tolist()

    # If only one item, convert to list
    if isinstance(scores, float):
        scores = [scores]

    avg_score = sum(scores) / len(scores)
    status = get_sentiment_status(avg_score)

    return {
        "average_sentiment_score": round(avg_score, 4),
        "status": status,
        "individual_scores": [round(s, 4) for s in scores]
    }