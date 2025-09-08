import joblib
from pathlib import Path

from RefundFraudDetectionAI.core.model import FraudModel
from RefundFraudDetectionAI.core.vectorstore import ClaimVectorStore
from RefundFraudDetectionAI.core.langgraph_pipeline import FraudDetectionGraph


# ---------------------------
# Paths
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
INDEX_PATH = BASE_DIR / "models" / "faiss_index"


class FraudPipeline:
    def __init__(self):
        # Load ML model
        if MODEL_PATH.exists():
            self.model: FraudModel = joblib.load(MODEL_PATH)
            print("✅ Loaded FraudModel")
        else:
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

        # Load FAISS index
        self.vector_store = ClaimVectorStore()
        if (INDEX_PATH.with_suffix(".faiss").exists() 
                and INDEX_PATH.with_suffix(".claims.txt").exists()):
            self.vector_store.load(str(INDEX_PATH))
            print("✅ Loaded FAISS index")
        else:
            raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")

        # Initialize LangGraph pipeline
        self.graph = FraudDetectionGraph(self.model, self.vector_store)

    def run(self, claim_text: str, user_history: str = "") -> dict:
        """
        Run inference pipeline on new claim
        """
        return self.graph.run(claim_text, user_history)


if __name__ == "__main__":
    pipeline = FraudPipeline()

    test_claim = "I never received my package, but tracking shows it was delivered."
    test_history = "Past 3 refund requests in last 2 months for similar reasons."

    result = pipeline.run(test_claim, test_history)

    print("\n🚨 Prediction Result 🚨")
    print(result)
