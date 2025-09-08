import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords once
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """Clean claim description text for embeddings/ML."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\d+", "", text)      # remove numbers
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create structured ML features from raw claim/order dataset.
    """
    df = df.copy()

    # --- Temporal features ---
    df["claim_hour"] = df["claim_timestamp"].dt.hour
    df["claim_dayofweek"] = df["claim_timestamp"].dt.dayofweek
    df["is_weekend"] = df["claim_dayofweek"].apply(lambda x: 1 if x >= 5 else 0)
    df["days_since_order"] = (df["claim_timestamp"] - df["order_timestamp"]).dt.days
    df["days_since_delivery"] = (df["claim_timestamp"] - df["delivery_timestamp"]).dt.days

    # --- Customer history aggregates ---
    customer_stats = df.groupby("customer_id").agg(
        total_orders=("order_id", "count"),
        total_claims=("claim_id", "count"),
        lifetime_order_value=("total_order_value", "sum"),
        lifetime_refund_value=("refund_amount", "sum"),
    ).reset_index()
    customer_stats["claim_rate"] = customer_stats["total_claims"] / customer_stats["total_orders"]
    customer_stats["refund_to_order_ratio"] = (
        customer_stats["lifetime_refund_value"] / customer_stats["lifetime_order_value"]
    )

    df = df.merge(customer_stats, on="customer_id", how="left")

    # --- Text cleaning ---
    df["cleaned_description"] = df["claim_description"].apply(clean_text)

    # --- Text numeric features ---
    df["desc_len"] = df["cleaned_description"].apply(lambda x: len(x) if isinstance(x, str) else 0)
    suspicious_words = {"never", "lost", "refund", "missing", "damaged", "expired", "defective"}
    def count_suspicious(text: str) -> int:
        if not isinstance(text, str):
            return 0
        words = set(text.split())
        return sum(1 for w in suspicious_words if w in words)
    df["desc_suspicious_count"] = df["cleaned_description"].apply(count_suspicious)

    # --- History-derived features ---
    def hist_features(text: str) -> dict:
        if not isinstance(text, str):
            return {"history_len": 0, "history_refund_mentions": 0, "history_repeat_cues": 0}
        t = text.lower()
        refund_mentions = t.count("refund")
        repeat_cues = sum(1 for kw in ["again", "multiple", "repeat"] if kw in t)
        return {
            "history_len": len(t.split()),
            "history_refund_mentions": refund_mentions,
            "history_repeat_cues": repeat_cues,
        }

    hist_df = df.get("user_history")
    if hist_df is not None:
        feats = df["user_history"].apply(hist_features).apply(pd.Series)
        df = pd.concat([df, feats], axis=1)

    # --- Fill NaNs ---
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna("unknown")

    return df


if __name__ == "__main__":
    from RefundFraudDetectionAI.data.generate_synthetic_data import generate_synthetic_data

    df = generate_synthetic_data(10, 0.2)
    processed = engineer_features(df)
    print(processed.head())
    print("Columns:", processed.columns.tolist())
