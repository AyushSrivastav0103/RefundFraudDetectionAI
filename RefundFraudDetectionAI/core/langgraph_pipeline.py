from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI  # or swap with Ollama
from langchain.prompts import PromptTemplate
import os


class FraudDetectionGraph:
    def __init__(self, model, vector_store, llm_backend: str = "ollama"):
        self.model = model
        self.vector_store = vector_store

        # Choose LLM backend
        # Prefer Groq on cloud when GROQ_API_KEY is present
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            # Route OpenAI-compatible client to Groq
            os.environ["OPENAI_API_KEY"] = groq_key
            os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
            # Pick a Groq-supported model, e.g., llama3-8b-instant
            self.llm = ChatOpenAI(model="llama3-8b-instant", temperature=0)
        elif llm_backend == "openai":
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        else:
            # Local fallback via Ollama
            from langchain_ollama import OllamaLLM
            self.llm = OllamaLLM(model="llama3")

        # Build workflow graph
        self.graph = self._build_graph()

    # ----------------------------
    # Graph Node Functions
    # ----------------------------
    def ml_check(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Build a single-row DataFrame to use the trained model
        import pandas as pd
        from RefundFraudDetectionAI.core.features import engineer_features
        from datetime import datetime

        # Minimal synthetic row using text and default numeric values
        now = pd.Timestamp.utcnow()
        df = pd.DataFrame([
            {
                "customer_id": "temp_user",
                "order_id": "temp_order",
                "claim_id": "temp_claim",
                "product_category": "Groceries",
                "total_order_value": 500.0,
                "refund_amount": 200.0,
                "claim_reason": "Damaged",
                "order_timestamp": now - pd.Timedelta(days=10),
                "delivery_timestamp": now - pd.Timedelta(days=8),
                "claim_timestamp": now,
                "claim_description": state["claim_text"],
                "user_history": state.get("user_history", ""),
                "fraud_flag": 0,
            }
        ])

        df_feat = engineer_features(df)
        preds = self.model.predict(df_feat)
        model_prob = float(preds.iloc[0]["fraud_probability"])  # 0..1

        # Heuristic driven by text+history (captures repetition cues quickly)
        hist = state.get("user_history", "")
        heur = self.model.predict_proba_text(state["claim_text"], hist)  # 0..1

        # Blend: rely primarily on model, with heuristic as supportive signal
        blended = max(0.0, min(1.0, 0.7 * model_prob + 0.3 * heur))

        state["ml_score_model_pct"] = model_prob * 100.0
        state["ml_score_heuristic_pct"] = heur * 100.0
        state["ml_score_blended_pct"] = blended * 100.0
        # Backward-compat primary field
        state["ml_score"] = state["ml_score_blended_pct"]
        return state

    def faiss_check(self, state: Dict[str, Any]) -> Dict[str, Any]:
        similar_claims = self.vector_store.search(state["claim_text"], k=3)
        state["similar_claims"] = similar_claims
        return state

    def llm_reasoning(self, state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = PromptTemplate.from_template(
            """
            You are an AI fraud investigator. A customer submitted a refund claim.

            Claim: {claim}
            Past History: {history}
            ML Fraud Probability: {ml_score}
            Similar Past Fraudulent Claims: {similar}

            Task: Decide if this claim is suspicious. Return:
            1. A probability of fraud (0–100%)
            2. A short reasoning (why)
            """
        )

        chain = prompt | self.llm
        response = chain.invoke(
            {
                "claim": state["claim_text"],
                "history": state.get("user_history", ""),
                "ml_score": state["ml_score"],
                "similar": state["similar_claims"],
            }
        )
        state["llm_reasoning"] = getattr(response, "content", response)
        return state

    def final_decision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "claim_text": state["claim_text"],
            "ml_score": state.get("ml_score", 0.0),
            "ml_score_model": state.get("ml_score_model_pct", 0.0),
            "ml_score_heuristic": state.get("ml_score_heuristic_pct", 0.0),
            "ml_score_blended": state.get("ml_score_blended_pct", 0.0),
            "similar_claims": state["similar_claims"],
            "llm_reasoning": state["llm_reasoning"],
        }

    # ----------------------------
    # Graph Assembly
    # ----------------------------
    def _build_graph(self):
        graph = StateGraph(dict)

        graph.add_node("ml_check", self.ml_check)
        graph.add_node("faiss_check", self.faiss_check)
        graph.add_node("llm_reasoning", self.llm_reasoning)
        graph.add_node("final", self.final_decision)

        # Flow
        graph.set_entry_point("ml_check")
        graph.add_edge("ml_check", "faiss_check")
        graph.add_edge("faiss_check", "llm_reasoning")
        graph.add_edge("llm_reasoning", "final")
        graph.add_edge("final", END)

        return graph.compile()

    # ----------------------------
    # Run Graph
    # ----------------------------
    def run(self, claim_text: str, user_history: str = "") -> Dict[str, Any]:
        state = {"claim_text": claim_text, "user_history": user_history}
        return self.graph.invoke(state)
