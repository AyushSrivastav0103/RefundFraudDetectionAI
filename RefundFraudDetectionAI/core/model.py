import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class FraudModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            class_weight="balanced"
        )
        self.feature_cols = None

    def train(self, df: pd.DataFrame):
        """
        Train RandomForest on engineered features.
        """
        # Extract features and labels
        # Align with current schema
        drop_cols = [c for c in ["fraud_flag", "claim_description", "cleaned_description", "user_history", "customer_id", "order_id", "claim_id", "product_category", "claim_reason", "order_timestamp", "delivery_timestamp", "claim_timestamp"] if c in df.columns]
        X = df.drop(columns=drop_cols)
        # Keep only numeric columns
        X = X.select_dtypes(include=["number"]).copy()
        y = df["fraud_flag"]

        self.feature_cols = X.columns

        # Train/test split (just for evaluation, model is trained on full set after)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)

        # Quick evaluation
        preds = self.model.predict(X_test)
        print("📊 Model Performance:\n", classification_report(y_test, preds))

        # Retrain on full dataset
        self.model.fit(X, y)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict probabilities and flags for provided engineered feature dataframe.
        Returns a DataFrame with columns: fraud_probability, fraud_flag_pred, anomaly_score (placeholder 0.0).
        """
        drop_cols = [c for c in ["fraud_flag", "claim_description", "cleaned_description", "user_history", "customer_id", "order_id", "claim_id", "product_category", "claim_reason", "order_timestamp", "delivery_timestamp", "claim_timestamp"] if c in df.columns]
        X = df.drop(columns=drop_cols)
        X = X.select_dtypes(include=["number"]).copy()
        # Ensure feature alignment
        X = X.reindex(columns=self.feature_cols, fill_value=0)
        proba = self.model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        out = pd.DataFrame({
            "fraud_probability": proba,
            "fraud_flag_pred": pred,
            "anomaly_score": 0.0,
        }, index=df.index)
        return out

    def predict_proba_text(self, claim_text: str, user_history: str = "") -> float:
        """
        Lightweight text+history heuristic score (0..1) for quick checks.
        Factors:
        - suspicious keywords in claim or history
        - repeated refund requests (numbers, phrases)
        - fuzzy matches for common typos (e.g., paneer/panner)
        """
        import re
        from difflib import SequenceMatcher

        score = 0.0
        claim = (claim_text or "").lower()
        history = (user_history or "").lower()

        # Suspicious keywords
        suspicious = [
            "never", "lost", "refund", "missing", "delivery", "didn't arrive",
            "did not arrive", "immediately", "demand", "chargeback",
        ]
        for kw in suspicious:
            if kw in claim or kw in history:
                score += 0.08

        # Repeated patterns / frequency hints
        if re.search(r"\b(again|another|multiple|repeated|repeat)\b", history):
            score += 0.15
        if re.search(r"\b(2|3|4|5|six|seven|eight|nine|ten)\b", history):
            score += 0.1
        if history.count("refund") >= 2:
            score += 0.15

        # Simple typo-tolerant keyword check (paneer/panner example)
        def similar(a: str, b: str) -> bool:
            return SequenceMatcher(None, a, b).ratio() >= 0.8

        words = set(re.findall(r"[a-z]+", claim + " " + history))
        for w in words:
            if similar(w, "paneer"):
                score += 0.05

        # Cap score in [0,1]
        return float(max(0.0, min(1.0, score)))
