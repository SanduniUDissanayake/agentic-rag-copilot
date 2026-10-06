import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8001")
st.set_page_config(page_title="Annual Report Copilot", layout="wide")


def esc(text):
    return text.replace("$", "\\$")


st.title("Annual Report Copilot")
st.caption("An AI agent that searches real company reports and calculates with a tool instead of guessing.")

with st.sidebar:
    st.header("Settings")
    preferred = st.radio("Preferred LLM provider", ["Groq", "Gemini"])
    st.markdown("If the preferred provider fails, the API falls back to the other automatically.")
    st.header("Knowledge base")
    st.markdown("- BHP 2026 Annual Report\n- CBA 2026 Annual Report\n- Telstra 2026 Annual Report\n- Woolworths 2026 Annual Report")
    st.header("Try one")
    examples = [
        "What was BHP's copper capital expenditure in FY2026?",
        "BHP's copper capex was $4.6B in FY2026 and is planned at $1.4B in FY2027. What is the percentage decrease?",
        "What is Woolworths Group's ABN?",
    ]
    clicked = None
    for ex in examples:
        if st.button(esc(ex), use_container_width=True):
            clicked = ex
    try:
        ok = requests.get(f"{API_URL}/health", timeout=3).ok
    except Exception:
        ok = False
    st.markdown("API status: " + ("online" if ok else "offline"))

if "history" not in st.session_state:
    st.session_state.history = []

for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(esc(item["q"]))
    with st.chat_message("assistant"):
        r = item["r"]
        st.write(esc(r["answer"]))
        st.caption(f"Answered by {r['provider']}" + (" (fallback used)" if r["fallback_used"] else ""))
        with st.expander("Agent trace: tools called"):
            for c in r["tool_calls"]:
                st.code(f"{c['name']}  {c['args']}")
        with st.expander("Source passages retrieved"):
            for s in r["sources"]:
                st.markdown(f"**{s['file']}** (similarity {s['score']:.2f})")
                st.write(esc(s["text"]))

question = st.chat_input("Ask about the annual reports...") or clicked
if question:
    with st.spinner("Agent is searching and reasoning..."):
        try:
            resp = requests.post(
                f"{API_URL}/query",
                json={"question": question, "provider": preferred},
                timeout=180,
            )
            resp.raise_for_status()
            st.session_state.history.append({"q": question, "r": resp.json()})
            st.rerun()
        except Exception as e:
            st.error(f"Request failed: {e}")