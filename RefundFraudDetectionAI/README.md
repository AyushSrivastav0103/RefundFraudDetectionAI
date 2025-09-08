# RefundFraudDetectionAI 🚀

RefundFraudDetectionAI is a next-gen **AI-powered refund fraud detection system**.  
It combines **structured ML models (XGBoost/RandomForest)**, **Vector Search (FAISS)**, and **Agentic Workflows (LangGraph)** with **LLMs (Ollama/OpenAI)**.  

## ✨ Features
- ML (RandomForest/XGBoost) on engineered features (temporal, history, text-derived).  
- Vector DB (FAISS) for similar-claim retrieval.  
- LangGraph pipeline: ML check → FAISS retrieval → LLM rationale.  
- Streamlit UI for quick demo with blended risk and explanations.  
- Works local; can use OpenAI for LLM or stay local with Ollama.

## 🔧 Installation
```bash
git clone https://github.com/AyushSrivastav0103/RefundFraudDetectionAI.git
cd RefundFraudDetectionAI
pip install -r requirements.txt
```

## 🧭 How it works (high level)
- Input a claim (and optional user history). The system computes a model probability, retrieves similar cases, and produces an LLM rationale.

## 🚀 Quickstart
1) Install
```bash
pip install -r requirements.txt
```

2) Train (creates `models/` artifacts)
```bash
python -m RefundFraudDetectionAI.pipeline.train
```

3) Run demo UI
```bash
streamlit run RefundFraudDetectionAI/ui/app.py
```

See detailed diagrams in `docs/ARCHITECTURE.md`.

## 🧠 Example
- Input claim: "I never received my package, but tracking shows it was delivered."
- Output: Model risk%, similar claims, and an LLM rationale.

## 🗣️ Layman’s explanation
- We analyze your claim and history, compare it with similar past cases, score the risk, and explain why.
