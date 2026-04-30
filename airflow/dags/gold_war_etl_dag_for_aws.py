from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from datetime import datetime, timedelta
from pathlib import Path

import feedparser
import joblib
import pandas as pd
import yfinance as yf
import boto3

from botocore.exceptions import ClientError
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from textblob import TextBlob


# -----------------------------------------------------------------------------
# S3 settings
# -----------------------------------------------------------------------------
S3_BUCKET = "cc-bucket-4821"
S3_PREFIX = ""


def get_s3_client():
    return boto3.client("s3")


def s3_key(path):
    if S3_PREFIX:
        return f"{S3_PREFIX}/{path}"
    return path


def upload_file_to_s3(local_path, s3_path):
    s3 = get_s3_client()
    s3.upload_file(str(local_path), S3_BUCKET, s3_key(s3_path))
    print(f"Uploaded to s3://{S3_BUCKET}/{s3_key(s3_path)}")


def download_file_from_s3(s3_path, local_path):
    s3 = get_s3_client()
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        s3.download_file(S3_BUCKET, s3_key(s3_path), str(local_path))
        print(f"Downloaded s3://{S3_BUCKET}/{s3_key(s3_path)}")
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ["404", "NoSuchKey"]:
            print(f"S3 file not found: {s3_path}")
            return False
        raise


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
LOCAL_DIR = Path("/tmp/gold_war_pipeline")
MODEL_DIR = LOCAL_DIR / "models"
SNAPSHOT_DIR = LOCAL_DIR / "snapshots"

GOLD_FILE = LOCAL_DIR / "gold_prices.csv"
NEWS_FILE = LOCAL_DIR / "war_news.csv"
TRAIN_FILE = LOCAL_DIR / "training_data.csv"
BEST_MODEL_FILE = MODEL_DIR / "gold_model.pkl"


def prepare_local_dirs():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "price_change",
    "ma7",
    "ma14",
    "high_low_diff",
    "open_close_diff",
    "volume",
    "sentiment_mean",
    "news_count",
    "has_news",
]

FEEDS = [
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
]

KEYWORDS = ["war", "conflict", "attack", "military", "invasion", "battle", "troops"]


# -----------------------------------------------------------------------------
# Task 1: Fetch gold prices
# -----------------------------------------------------------------------------
def fetch_gold_prices():
    prepare_local_dirs()
    download_file_from_s3("raw/gold/gold_prices.csv", GOLD_FILE)

    if GOLD_FILE.exists():
        existing = pd.read_csv(GOLD_FILE, parse_dates=["date"])
        last_date = existing["date"].max().normalize()
        next_date = last_date + pd.Timedelta(days=1)

        today = pd.Timestamp.today().normalize()

        if next_date > today:
            print("No new gold price data available (already up to date)")
            return

        print(f"Existing data found. Fetching from {next_date.date()} to {today.date()}")
        new_data = yf.download(
            "GC=F",
            start=next_date.strftime("%Y-%m-%d"),
            end=(today + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            multi_level_index=False,
        )
    else:
        print("First run. Fetching all data from 2024-01-01")
        new_data = yf.download(
            "GC=F",
            start="2024-01-01",
            progress=False,
            multi_level_index=False,
        )

    if new_data.empty:
        print("No new gold data available")
        return

    new_data = new_data.reset_index()
    new_data = new_data[["Date", "Open", "High", "Low", "Close", "Volume"]]
    new_data = new_data.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    new_data["date"] = pd.to_datetime(new_data["date"]).dt.strftime("%Y-%m-%d")

    if GOLD_FILE.exists():
        existing = pd.read_csv(GOLD_FILE)
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset="date").sort_values("date")
        combined.to_csv(GOLD_FILE, index=False)
    else:
        new_data.to_csv(GOLD_FILE, index=False)

    print(f"Saved gold data to: {GOLD_FILE}")
    upload_file_to_s3(GOLD_FILE, "raw/gold/gold_prices.csv")


# -----------------------------------------------------------------------------
# Task 2: Fetch war news from NYT RSS
# -----------------------------------------------------------------------------
def fetch_war_news():
    prepare_local_dirs()
    download_file_from_s3("raw/news/war_news.csv", NEWS_FILE)

    articles = []

    for url in FEEDS:
        feed = feedparser.parse(url)
        print(f"{url} -> {len(feed.entries)} articles received")

        for entry in feed.entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            text = f"{title} {summary}".lower()

            if any(keyword in text for keyword in KEYWORDS):
                try:
                    published = entry.get("published_parsed")
                    date = datetime(*published[:3]).strftime("%Y-%m-%d")
                except Exception:
                    date = datetime.today().strftime("%Y-%m-%d")

                articles.append(
                    {
                        "date": date,
                        "title": title,
                        "summary": summary,
                    }
                )

    if not articles:
        print("No war-related articles found")
        new_df = pd.DataFrame(columns=["date", "title", "summary"])
    else:
        new_df = pd.DataFrame(articles)

    if NEWS_FILE.exists():
        existing = pd.read_csv(NEWS_FILE)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "title"])
        combined.to_csv(NEWS_FILE, index=False)
    else:
        new_df.to_csv(NEWS_FILE, index=False)

    print(f"Saved war news to: {NEWS_FILE}")
    upload_file_to_s3(NEWS_FILE, "raw/news/war_news.csv")


# -----------------------------------------------------------------------------
# Task 3: Sentiment + merge + feature engineering
# -----------------------------------------------------------------------------
def compute_sentiment_and_merge():
    prepare_local_dirs()

    download_file_from_s3("raw/gold/gold_prices.csv", GOLD_FILE)
    download_file_from_s3("raw/news/war_news.csv", NEWS_FILE)

    if not GOLD_FILE.exists():
        raise FileNotFoundError(f"Missing file: {GOLD_FILE}")
    if not NEWS_FILE.exists():
        raise FileNotFoundError(f"Missing file: {NEWS_FILE}")

    gold = pd.read_csv(GOLD_FILE, parse_dates=["date"])
    news = pd.read_csv(NEWS_FILE, parse_dates=["date"])

    if news.empty:
        news_agg = pd.DataFrame(
            columns=["date", "sentiment_mean", "news_count", "has_news"]
        )
    else:
        def get_sentiment(text):
            return TextBlob(str(text)).sentiment.polarity

        news["text"] = news["title"].fillna("") + " " + news["summary"].fillna("")
        news["sentiment"] = news["text"].apply(get_sentiment)

        news_agg = (
            news.groupby("date")
            .agg(
                sentiment_mean=("sentiment", "mean"),
                news_count=("title", "count"),
            )
            .reset_index()
        )

        news_agg["has_news"] = 1

    merged = pd.merge(gold, news_agg, on="date", how="left")
    merged["sentiment_mean"] = merged["sentiment_mean"].fillna(0)
    merged["news_count"] = merged["news_count"].fillna(0)
    merged["has_news"] = merged["has_news"].fillna(0)

    merged = merged.sort_values("date").reset_index(drop=True)

    merged["price_change"] = merged["close"].pct_change()
    merged["ma7"] = merged["close"].rolling(7).mean()
    merged["ma14"] = merged["close"].rolling(14).mean()
    merged["high_low_diff"] = merged["high"] - merged["low"]
    merged["open_close_diff"] = merged["close"] - merged["open"]

    merged["target"] = (merged["close"].shift(-1) > merged["close"]).astype(float)
    merged = merged.dropna().copy()
    merged["target"] = merged["target"].astype(int)

    output = merged[
        [
            "date",
            "price_change",
            "ma7",
            "ma14",
            "high_low_diff",
            "open_close_diff",
            "volume",
            "sentiment_mean",
            "news_count",
            "has_news",
            "target",
        ]
    ]

    output.to_csv(TRAIN_FILE, index=False)

    date_str = datetime.today().strftime("%Y%m%d")
    snapshot_path = SNAPSHOT_DIR / f"training_data_{date_str}.csv"
    output.to_csv(snapshot_path, index=False)

    print(f"Saved training data to: {TRAIN_FILE}")
    print(f"Snapshot saved to: {snapshot_path}")
    print(f"Total rows: {len(output)}")

    upload_file_to_s3(TRAIN_FILE, "processed/training_data.csv")
    upload_file_to_s3(
        snapshot_path,
        f"processed/snapshots/training_data_{date_str}.csv",
    )


# -----------------------------------------------------------------------------
# Task 4: Train model
# -----------------------------------------------------------------------------
def train_model():
    prepare_local_dirs()

    download_file_from_s3("processed/training_data.csv", TRAIN_FILE)

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Missing file: {TRAIN_FILE}")

    df = pd.read_csv(TRAIN_FILE, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    X = df[FEATURES]
    y = df["target"]

    split_idx = int(len(df) * 0.8)
    if split_idx == 0 or split_idx == len(df):
        raise ValueError("Not enough data to perform train/test split.")

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    model = RandomForestClassifier(
        n_estimators=20,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    new_acc = accuracy_score(y_test, preds)
    print(f"New model accuracy: {new_acc:.4f}")

    date_str = datetime.today().strftime("%Y%m%d")
    versioned_model_path = MODEL_DIR / f"gold_model_{date_str}.pkl"

    joblib.dump(model, versioned_model_path)
    print(f"Versioned model saved to: {versioned_model_path}")

    upload_file_to_s3(
        versioned_model_path,
        f"models/archive/gold_model_{date_str}.pkl",
    )

    has_old_model = download_file_from_s3(
        "models/best/gold_model.pkl",
        BEST_MODEL_FILE,
    )

    if has_old_model:
        old_model = joblib.load(BEST_MODEL_FILE)
        old_preds = old_model.predict(X_test)
        old_acc = accuracy_score(y_test, old_preds)
        print(f"Old model accuracy: {old_acc:.4f}")

        if new_acc >= old_acc:
            joblib.dump(model, BEST_MODEL_FILE)
            upload_file_to_s3(BEST_MODEL_FILE, "models/best/gold_model.pkl")
            print("Updated best gold_model.pkl")
        else:
            print("Kept existing best gold_model.pkl")
    else:
        joblib.dump(model, BEST_MODEL_FILE)
        upload_file_to_s3(BEST_MODEL_FILE, "models/best/gold_model.pkl")
        print(f"First run. Saved best model to: {BEST_MODEL_FILE}")


# -----------------------------------------------------------------------------
# Airflow DAG
# -----------------------------------------------------------------------------
default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
}

with DAG(
    dag_id="gold_war_pipeline_aws",
    default_args=default_args,
    start_date=datetime(2026, 4, 26),
    schedule="0 2 * * *", #everyday 2am
    #schedule="@daily",
    catchup=False,
    tags=["gold", "mid", "aws"],
) as dag:

    t1 = PythonOperator(
        task_id="fetch_gold_prices",
        python_callable=fetch_gold_prices,
    )

    t2 = PythonOperator(
        task_id="fetch_war_news",
        python_callable=fetch_war_news,
    )

    t3 = PythonOperator(
        task_id="compute_sentiment_and_merge",
        python_callable=compute_sentiment_and_merge,
    )

    t4 = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    [t1, t2] >> t3 >> t4