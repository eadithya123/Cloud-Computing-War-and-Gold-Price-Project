import json
import boto3
import joblib
import pandas as pd
import numpy as np
import feedparser
import yfinance as yf
from textblob import TextBlob
from datetime import datetime
from pathlib import Path
import tempfile

S3_BUCKET = "cc-bucket-4821"

FEATURES = [
    "price_change", "ma7", "ma14", "high_low_diff",
    "open_close_diff", "volume", "sentiment_mean",
    "news_count", "has_news",
]

FEEDS = [
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
]

KEYWORDS = ["war", "conflict", "attack", "military", "invasion", "battle", "troops"]


def lambda_handler(event, context):

    s3 = boto3.client("s3")
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        model_path = tmp.name

    s3.download_file(S3_BUCKET, "models/best/gold_model.pkl", model_path)
    model = joblib.load(model_path)
    print("Model loaded successfully")

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        gold_path = tmp.name

    s3.download_file(S3_BUCKET, "raw/gold/gold_prices.csv", gold_path)
    gold = pd.read_csv(gold_path, parse_dates=["date"])
    gold = gold.sort_values("date").reset_index(drop=True)

    today = pd.Timestamp.today().normalize()
    today_str = today.strftime("%Y-%m-%d")

    new_data = yf.download(
        "GC=F",
        start=today_str,
        end=(today + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False,
        multi_level_index=False,
    )

    if new_data.empty:
        return {
            "statusCode": 200,
            "body": json.dumps({"message": "No gold price data available for today yet."})
        }

    new_data = new_data.reset_index()
    new_data = new_data[["Date", "Open", "High", "Low", "Close", "Volume"]]
    new_data.columns = ["date", "open", "high", "low", "close", "volume"]
    new_data["date"] = pd.to_datetime(new_data["date"])

    gold = pd.concat([gold, new_data], ignore_index=True)
    gold = gold.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)

    articles = []
    for url in FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            text = f"{title} {summary}".lower()
            if any(kw in text for kw in KEYWORDS):
                articles.append({"title": title, "summary": summary})

    if articles:
        sentiments = [
            TextBlob(f"{a['title']} {a['summary']}").sentiment.polarity
            for a in articles
        ]
        sentiment_mean = float(np.mean(sentiments))
        news_count = len(articles)
        has_news = 1
    else:
        sentiment_mean = 0.0
        news_count = 0
        has_news = 0

    print(f"News articles found: {news_count}, Sentiment: {sentiment_mean:.4f}")

    gold["price_change"] = gold["close"].pct_change()
    gold["ma7"]          = gold["close"].rolling(7).mean()
    gold["ma14"]         = gold["close"].rolling(14).mean()
    gold["high_low_diff"]   = gold["high"] - gold["low"]
    gold["open_close_diff"] = gold["close"] - gold["open"]

    latest = gold.iloc[-1].copy()
    latest["sentiment_mean"] = sentiment_mean
    latest["news_count"]     = news_count
    latest["has_news"]       = has_news

    input_df = pd.DataFrame([latest[FEATURES]])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    result = "UP 📈" if prediction == 1 else "DOWN 📉"
    confidence = float(max(probability)) * 100

    print(f"Prediction: {result} | Confidence: {confidence:.1f}%")

    return {
        "statusCode": 200,
        "body": json.dumps({
            "date": today_str,
            "prediction": result,
            "confidence_pct": round(confidence, 1),
            "sentiment_mean": round(sentiment_mean, 4),
            "news_count": news_count,
        })
    }