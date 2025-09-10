import os
import sys
import streamlit as st

# Ensure the project root is on sys.path so package imports work when run via streamlit
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from RefundFraudDetectionAI.pipeline.predict import FraudPipeline

st.set_page_config(page_title="Refund Fraud Detection AI", page_icon="🤖", layout="centered")

st.title("Refund Fraud Detection AI")
st.markdown("Enter a claim description to analyze risk. Model + FAISS + LLM rationale.")

@st.cache_resource
def get_pipeline():
    return FraudPipeline()

pipeline = get_pipeline()

claim_text = st.text_area("Claim description", height=150, placeholder="I never received my package, but tracking shows it was delivered.")
user_history = st.text_area("User history (optional)", height=100, placeholder="Past 3 refund requests in last 2 months for similar reasons.")

if 'history' not in st.session_state:
    st.session_state['history'] = []

if st.button("Analyze"):
    if not claim_text.strip():
        st.warning("Please enter a claim description.")
    else:
        with st.spinner("Analyzing claim..."):
            try:
                result = pipeline.run(claim_text, user_history)
                st.success("Done!")
                # Save to history
                st.session_state['history'].append({
                    'claim_text': claim_text,
                    'user_history': user_history,
                    'result': result
                })
            except Exception as e:
                st.error(f"Error: {e}")
                result = None
        if result:
            # --- LLM Rationale First ---
            st.subheader("AI Investigator Rationale")
            rationale = result.get("llm_reasoning", "(no rationale)")
            st.markdown(
                f"""
                <div style=\"border:2px solid #ddd; border-radius:10px; padding:15px; background-color:#f9f9f9;\">
                    <span style=\"font-size:1.1em; font-weight:bold; color:#333;\">{rationale}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            # --- Fraud Risk Score as Support ---
            st.subheader("Fraud Risk Score")
            blended = result.get('ml_score_blended', result.get('ml_score', 0.0))
            model_pct = result.get('ml_score_model', 0.0)
            heur_pct = result.get('ml_score_heuristic', 0.0)

            if blended < 20:
                band, color = "Low", "#4caf50"
            elif blended < 60:
                band, color = "Medium", "#ff9800"
            else:
                band, color = "High", "#f44336"

            st.markdown(f'<div style="font-size:1.3em;font-weight:bold;color:{color};">Risk: {band} ({blended:.1f}%)</div>', unsafe_allow_html=True)
            st.progress(min(int(blended), 100), text=f"{blended:.1f}%")

            # --- Extra Details Hidden ---
            with st.expander("Details: Scores"):
                st.write(f"Model probability: {model_pct:.1f}%")
                st.write(f"Heuristic score: {heur_pct:.1f}%")

            with st.expander("Similar Claims"):
                for r in result.get("similar_claims", []) or []:
                    st.write(f"- {r['claim_text']} (sim={r.get('similarity', r.get('similarity_score', 0.0)):.3f})")

# Show history of past runs
if st.session_state['history']:
    st.sidebar.header("Past Analyses")
    for i, entry in enumerate(reversed(st.session_state['history'])):
        st.sidebar.markdown(f"**Claim {len(st.session_state['history'])-i}:** {entry['claim_text'][:40]}...")
        st.sidebar.markdown(f"Risk: {entry['result'].get('ml_score_blended', entry['result'].get('ml_score', 0.0)):.1f}%")
        st.sidebar.markdown(f"LLM: {entry['result'].get('llm_reasoning', '(no rationale)')[:60]}...")
        st.sidebar.markdown("---")
