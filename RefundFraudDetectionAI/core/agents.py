from typing import Dict, Any
from langgraph.graph import StateGraph, END
from openai import OpenAI
import os

from RefundFraudDetectionAI.core.vectorstore import ClaimVectorStore
from RefundFraudDetectionAI.core.model import FraudModel


# ---------------------------
# Judge Agent (LLM)
# ---------------------------

def llm_judge(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use LLM to judge claim risk based on retriever + analyzer results.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    retriever_out = context.get("retriever", [])
    analyzer_out = context.get("analyzer", {})

    prompt = f"""
    You are a fraud detection AI. A new refund claim has been submitted.

    Analyzer Results:
    Fraud Probability: {analyzer_out.get('fraud_probability')}
    Anomaly Score: {analyzer_out.get('anomaly_score')}

    Similar Past Claims (from retriever):
    {retriever_out}

    Task:
    - Decide risk strictly using these bands:
      - Low if prob < 0.20
      - Medium if 0.20 <= prob < 0.60
      - High if prob >= 0.60
      Do not contradict these thresholds.
    - Explain your reasoning in 3–4 clear sentences.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    decision = response.choices[0].message.content.strip()
    return {"judge": decision}


# ---------------------------
# Build LangGraph
# ---------------------------

def build_graph(model: FraudModel, store: ClaimVectorStore):
    """
    Build LangGraph pipeline: Retriever -> Analyzer -> Judge
    """
    workflow = StateGraph(dict)

    # Retriever Node
    def retriever(state: Dict[str, Any]) -> Dict[str, Any]:
        query = state["claim_description"]
        results = store.search(query, k=3)
        return {"retriever": results}

    # Analyzer Node
    def analyzer(state: Dict[str, Any]) -> Dict[str, Any]:
        df = state["df"]
        preds = model.predict(df)
        row = preds.iloc[0]
        return {
            "analyzer": {
                "fraud_probability": row["fraud_probability"],
                "anomaly_score": row["anomaly_score"],
                "fraud_flag_pred": row["fraud_flag_pred"],
            }
        }

    # Judge Node
    def judge(state: Dict[str, Any]) -> Dict[str, Any]:
        return llm_judge(state)

    # Build edges
    workflow.add_node("retriever", retriever)
    workflow.add_node("analyzer", analyzer)
    workflow.add_node("judge", judge)

    workflow.set_entry_point("retriever")
    workflow.add_edge("retriever", "analyzer")
    workflow.add_edge("analyzer", "judge")
    workflow.add_edge("judge", END)

    return workflow.compile()


if __name__ == "__main__":
    from RefundFraudDetectionAI.data.generate_synthetic_data import generate_synthetic_data
    from RefundFraudDetectionAI.core.features import engineer_features

    df = generate_synthetic_data(5, 0.2)
    df = engineer_features(df)

    model = FraudModel()
    model.train(df)

    store = ClaimVectorStore()
    store.build_index(df)

    # Take one sample claim
    sample = df.iloc[[0]]

    graph = build_graph(model, store)
    result = graph.invoke({"df": sample, "claim_description": sample["claim_description"].iloc[0]})

    print("🚀 Final Decision:", result["judge"])
