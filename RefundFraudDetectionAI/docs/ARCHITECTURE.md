# Architecture

## Overview
- Training builds a supervised model from engineered features and a FAISS vector index from claim text.
- Inference runs a LangGraph pipeline that computes a model probability, retrieves similar claims, and asks an LLM to explain the result.

## Components
- data/generate_synthetic_data.py: synthetic claims + user_history; labels correlate with repeated-refund patterns.
- core/features.py: temporal features, aggregates, text cleaning, and history-derived features (length, refund mentions, repeat cues).
- core/model.py: RandomForest classifier (drop-in: XGBoost). Exposes train/predict.
- core/vectorstore.py: SentenceTransformers encoder + FAISS IP index (cosine-like similarity).
- core/langgraph_pipeline.py: ml_check → faiss_check → llm_reasoning → final_decision; blended risk exposed.
- core/agents.py: alternative graph with explicit Judge bands tied to model probability.
- pipeline/train.py: orchestrates training and saves artifacts in models/.
- pipeline/predict.py: loads artifacts and runs the graph.
- ui/app.py: Streamlit UI to input claim/history and visualize outputs.

## Data Flow (Mermaid)
```mermaid
flowchart LR
  subgraph Training
    A[Generate Synthetic Data\n data/generate_synthetic_data.py]
    B[Engineer Features\n core/features.py]
    C[Train Model\n core/model.py]
    D[Build FAISS Index\n core/vectorstore.py]
    A --> B --> C
    B --> D
    C -->|save| M[(models/fraud_model.pkl)]
    D -->|save| I[(models/faiss_index.*)]
  end

  subgraph Inference (LangGraph)
    X[Input Claim + History]
    G1[ml_check\n model.predict]
    G2[faiss_check\n similar claims]
    G3[llm_reasoning\n explanation]
    X --> G1 --> G2 --> G3 --> Y[Final Decision\n scores + neighbors + rationale]
  end

  M -.load.-> G1
  I -.load.-> G2
```

## Risk Calculation
- Model probability is computed from engineered features.
- Heuristic (text+history) can be blended for responsiveness: blended = 0.7×model + 0.3×heuristic.
- UI displays risk band: Low (<20%), Medium (20–60%), High (≥60%).

## Notes
- For cloud deployment, prefer OpenAI for LLM or disable LLM step for fully local operation.
- Consider calibrating model probabilities and upgrading to XGBoost for stronger performance.
