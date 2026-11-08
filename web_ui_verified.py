"""Streamlit front-end for the Enterprise Knowledge Assistant.

Requires the existing FastAPI RAG server on 127.0.0.1:8001 and the
vLLM-Metal server on 127.0.0.1:8000. Does not directly load models.

Run: streamlit run web_ui_verified.py --server.port 8503
"""

from __future__ import annotations

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8001"
MODEL_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Verified Enterprise Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    .stApp { background: #f6f8fc; color: #152340; }
    .block-container { max-width: 1060px; padding-top: 1.6rem; padding-bottom: 5rem; }
    html, body, [class*="css"] { font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif; }
    [data-testid="stSidebar"] { background: #eef2f9; border-right: 1px solid #dce4f1; }
    [data-testid="stSidebar"] .block-container { padding-top: 2rem; }
    .hero {
        padding: 1.8rem 2rem; border-radius: 20px;
        background: linear-gradient(118deg, #172d61 0%, #264f97 60%, #3a6ca9 100%);
        color: #fff; box-shadow: 0 12px 32px rgba(19,48,103,.12);
        margin-bottom: 1.25rem;
    }
    .hero .eyebrow { font-size: .77rem; font-weight: 700; letter-spacing: .15em;
        text-transform: uppercase; color: #a9dfff; margin-bottom: .65rem; }
    .hero h1 { font-size: 2rem; line-height: 1.17; font-weight: 750;
        color: #fff; padding: 0; margin: 0 0 .45rem; }
    .hero p { margin: 0; max-width: 42rem; color: #dceafa;
        font-size: .98rem; line-height: 1.6; }
    .mini-label { color: #536682; text-transform: uppercase; letter-spacing: .08em;
        font-weight: 700; font-size: .75rem; margin: .65rem 0 .3rem; }
    .small-note { font-size: .85rem; color: #64758e; line-height: 1.5; }
    .status-line { padding: .6rem .8rem; margin-bottom: .5rem; background: white;
        border: 1px solid #dfe7f3; border-radius: 12px; font-size: .85rem; }
    .tag-row { display:flex; flex-wrap:wrap; gap:.45rem; margin-top:1rem; }
    .tag { font-size:.76rem; color:#e6f2ff; padding:.29rem .62rem;
        border:1px solid #7197c7; background:rgba(255,255,255,.10); border-radius:999px; }
    div.stButton > button { border-radius: 12px !important; border-color: #d5e0ef !important;
        background: #fff !important; color: #234071 !important; font-weight: 600 !important;
        min-height: 3.2rem; text-align: left; }
    div.stButton > button:hover { border-color: #6b90c9 !important;
        color: #163e77 !important; background: #f2f7fe !important; }
    [data-testid="stChatMessage"] { background: #fff; border: 1px solid #e2e8f2;
        border-radius: 15px; margin-bottom: .8rem; padding: .7rem .9rem; }
    [data-testid="stChatInput"] { background:#fff; border:1px solid #ccd9e9;
        border-radius: 13px; }
    div[data-testid="stExpander"] { border: 1px solid #e0e7f0; border-radius: 10px;
        background: #fafcff; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []


def check_status(url: str) -> bool:
    try:
        resp = requests.get(url, timeout=2)
        return resp.ok
    except requests.RequestException:
        return False


with st.sidebar:
    st.markdown("### 📚 Knowledge Assistant")
    st.caption("Local AI · RAG demonstration")
    st.divider()
    st.markdown('<div class="mini-label">System status</div>', unsafe_allow_html=True)
    rag_ready = check_status(f"{API_BASE}/health")
    model_ready = check_status(f"{MODEL_BASE}/v1/models")
    st.markdown(
        f'<div class="status-line">{"🟢" if rag_ready else "🔴"} '
        f'<strong>RAG API</strong> · {"Online" if rag_ready else "Offline"}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="status-line">{"🟢" if model_ready else "🔴"} '
        f'<strong>vLLM-Metal</strong> · {"Online" if model_ready else "Offline"}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="mini-label">Knowledge base</div>', unsafe_allow_html=True)
    st.markdown("**Enterprise Data & AI Operations Handbook**")
    st.caption("10-page fictional English handbook · 48 text chunks (initial indexing)")
    st.markdown('<div class="mini-label">Pipeline</div>', unsafe_allow_html=True)
    st.caption("PDF → Hybrid Retrieval → Adaptive Rerank → Evidence Verification → Answer/Abstain")
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    st.markdown(
        '<p class="small-note">This is a local portfolio demo. '
        'The current knowledge base is fixed; the page does not upload new files. '
        'Answers may be incomplete and should be checked against citations.</p>',
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">AI-powered document intelligence</div>
      <h1>Enterprise Knowledge Assistant</h1>
      <p>Ask questions about company data governance and AI operations.
      Get source-checked answers or an explicit refusal when evidence is insufficient.</p>
      <div class="tag-row">
        <span class="tag">RAG</span><span class="tag">Hybrid Retrieval</span>
        <span class="tag">Adaptive Reranking</span><span class="tag">vLLM-Metal</span>
        <span class="tag">FastAPI</span><span class="tag">Verified Evidence</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("#### Ask the handbook")

suggestions = [
    "What are common data quality checks?",
    "What are the rules for accessing sensitive data?",
    "How should AI model performance be evaluated?",
]

if not st.session_state.history:
    st.caption("Try one of these sample questions, or enter your own below.")
    cols = st.columns(3, gap="small")
    selected = None
    for col, prompt in zip(cols, suggestions):
        with col:
            if st.button(prompt, use_container_width=True):
                selected = prompt
else:
    selected = None

for turn in st.session_state.history:
    if turn["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(turn["content"])
    else:
        with st.chat_message("assistant", avatar="📚"):
            st.markdown(turn["content"])
            if turn.get("answerability_decision") == "abstain":
                st.caption("Answer withheld · " + str(turn.get("refusal_reason", "insufficient evidence")))
            sources = turn.get("sources", [])
            if turn.get("retrieval_method"):
                st.caption("Retrieval: " + turn["retrieval_method"])
            if sources:
                st.caption("📄 Verified text match · inspect the evidence (not a guarantee of entailment)")
                for n, source in enumerate(sources, 1):
                    page = source.get("page", "?")
                    score = source.get("rerank_score")
                    label = f"Source {n} · PDF page {page}"
                    if isinstance(score, (int, float)):
                        label += f" · rerank score {score:.2f}"
                    with st.expander(label):
                        st.write(source.get("excerpt", "No excerpt available."))

question = st.chat_input("Ask a question about the PDF…")
if question is None:
    question = selected

if question and question.strip():
    question = question.strip()
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="📚"):
        with st.spinner("Searching the PDF, selectively reranking and generating an answer…"):
            try:
                r = requests.post(f"{API_BASE}/ask", json={"question": question}, timeout=240)
                r.raise_for_status()
                payload = r.json()
                answer = payload.get("answer") or "No answer was returned."
                sources = payload.get("sources") or []
                st.markdown(answer)
                method = payload.get("retrieval_method", "Unknown")
                decision = payload.get("answerability_decision")
                if decision == "abstain":
                    st.warning("Answer withheld: " + str(payload.get("refusal_reason", "Insufficient verified evidence")))
                elif decision == "answer":
                    st.caption("Source quote verified against retrieved text · not an entailment guarantee")
                latency = payload.get("retrieval_latency_ms")
                st.caption(f"Retrieval: {method}" + (f" · {latency:.1f} ms (retrieval only)" if isinstance(latency, (int, float)) else ""))
                if sources:
                    st.caption("📄 Text-matched evidence (check factual support manually)")
                    for n, source in enumerate(sources, 1):
                        label = f"Source {n} · PDF page {source.get('page', '?')}"
                        with st.expander(label):
                            st.write(source.get("excerpt", "No excerpt available."))
                st.session_state.history.append(
                    {"role": "assistant", "content": answer, "sources": sources, "retrieval_method": method, "answerability_decision": decision, "refusal_reason": payload.get("refusal_reason")}
                )
            except requests.HTTPError as exc:
                msg = f"The RAG API returned an error: {exc}."
                st.error(msg)
                st.session_state.history.append({"role": "assistant", "content": msg})
            except requests.RequestException:
                msg = (
                    "Cannot reach the RAG service. Keep FastAPI running on port 8001 "
                    "and vLLM running on port 8000, then try again."
                )
                st.error(msg)
                st.session_state.history.append({"role": "assistant", "content": msg})
