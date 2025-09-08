import os
import joblib
from pathlib import Path

from RefundFraudDetectionAI.data.generate_synthetic_data import generate_synthetic_data
from RefundFraudDetectionAI.core.features import engineer_features
from RefundFraudDetectionAI.core.model import FraudModel
from RefundFraudDetectionAI.core.vectorstore import ClaimVectorStore


# ---------------------------
# Paths
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
INDEX_PATH = BASE_DIR / "models" / "faiss_index"


def train_pipeline(n_samples: int = 200, fraud_ratio: float = 0.15):
    """
    Train FraudModel + build FAISS vector index
    """
    print("📊 Generating synthetic data...")
    df = generate_synthetic_data(n_samples, fraud_ratio)
    df = engineer_features(df)

    # 1. Train ML model
    print("🤖 Training FraudModel...")
    model = FraudModel()
    model.train(df)

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved at {MODEL_PATH}")

    # 2. Build FAISS vector store
    print("🔎 Building FAISS index...")
    store = ClaimVectorStore()
    store.build_index(df)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    store.save(str(INDEX_PATH))
    print(f"✅ FAISS index saved at {INDEX_PATH}.faiss and {INDEX_PATH}.claims.txt")


if __name__ == "__main__":
    train_pipeline()
