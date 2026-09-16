import streamlit as st
import os
import tempfile
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

PERSIST_DIR = "faiss_store"

st.set_page_config(
    page_title="DocMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global theme — dark, card-based, no default Streamlit chrome
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    #MainMenu, footer, header { visibility: hidden; }

    .stApp {
        background: radial-gradient(circle at 15% 0%, #1a1f3a 0%, #0d0f1a 45%, #08090f 100%);
        color: #e8e9f3;
    }

    .block-container { padding-top: 2rem; max-width: 1200px; }

    /* Brand header */
    .dm-header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 6px;
    }
    .dm-logo {
        width: 42px; height: 42px;
        border-radius: 12px;
        background: linear-gradient(135deg, #7c5cff, #22d3ee);
        display: flex; align-items: center; justify-content: center;
        font-size: 22px;
        box-shadow: 0 0 24px rgba(124,92,255,0.45);
    }
    .dm-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.7rem; font-weight: 700; color: #f4f4fb;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .dm-subtitle { color: #8b8fa8; font-size: 0.92rem; margin-top: -4px; }

    /* Panel card */
    .dm-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 20px 22px;
        backdrop-filter: blur(6px);
        margin-bottom: 16px;
    }
    .dm-card h4 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.95rem;
        color: #a5a9c9;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0 0 14px 0;
    }

    .dm-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(124,92,255,0.15);
        border: 1px solid rgba(124,92,255,0.35);
        color: #c9bfff;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        margin: 3px 4px 3px 0;
    }

    .dm-stat {
        display: flex; justify-content: space-between; align-items: baseline;
        padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .dm-stat:last-child { border-bottom: none; }
    .dm-stat-label { color: #8b8fa8; font-size: 0.85rem; }
    .dm-stat-value { font-family: 'Space Grotesk', sans-serif; font-weight: 600; color: #f4f4fb; }

    /* Chat bubbles (fully custom, not st.chat_message) */
    .dm-msg-row { display: flex; margin-bottom: 14px; }
    .dm-msg-row.user { justify-content: flex-end; }
    .dm-bubble {
        max-width: 74%;
        padding: 12px 16px;
        border-radius: 16px;
        font-size: 0.94rem;
        line-height: 1.5;
    }
    .dm-bubble.user {
        background: linear-gradient(135deg, #7c5cff, #5b3fd6);
        color: white;
        border-bottom-right-radius: 4px;
    }
    .dm-bubble.assistant {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        color: #e8e9f3;
        border-bottom-left-radius: 4px;
    }
    .dm-avatar {
        width: 30px; height: 30px; border-radius: 9px;
        display: flex; align-items: center; justify-content: center;
        font-size: 15px; flex-shrink: 0;
    }
    .dm-avatar.user { background: rgba(124,92,255,0.25); margin-left: 10px; order: 2; }
    .dm-avatar.assistant { background: rgba(34,211,238,0.18); margin-right: 10px; }

    /* Thinking indicator */
    .dm-bubble.dm-thinking {
        display: flex;
        align-items: center;
        gap: 5px;
        padding: 14px 18px;
    }
    .dm-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        background: #8b8fa8;
        animation: dm-bounce 1.2s infinite ease-in-out;
    }
    .dm-dot:nth-child(2) { animation-delay: 0.15s; }
    .dm-dot:nth-child(3) { animation-delay: 0.3s; }
    @keyframes dm-bounce {
        0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
        30% { transform: translateY(-5px); opacity: 1; }
    }

    .dm-empty {
        text-align: center;
        color: #6b6f8a;
        padding: 60px 20px;
        font-size: 0.95rem;
    }
    .dm-empty-icon { font-size: 2.4rem; margin-bottom: 10px; }

    /* Buttons + inputs */
    .stButton > button {
        background: linear-gradient(135deg, #7c5cff, #22d3ee);
        color: #08090f;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1rem;
    }
    .stButton > button:hover { opacity: 0.9; }

    [data-testid="stFileUploader"] {
        border-radius: 14px;
        border: 1.5px dashed rgba(124,92,255,0.4);
        padding: 10px;
        background: rgba(124,92,255,0.05);
    }

    .stChatInput textarea, .stChatInput { border-radius: 14px !important; }

    /* Floating background particles */
    .dm-bg {
        position: fixed;
        inset: 0;
        overflow: hidden;
        z-index: 0;
        pointer-events: none;
    }
    .dm-bg span {
        position: absolute;
        bottom: -120px;
        border-radius: 50%;
        background: radial-gradient(circle at 30% 30%, rgba(124,92,255,0.55), rgba(34,211,238,0.05));
        box-shadow: 0 0 12px rgba(124,92,255,0.35);
        animation: dm-float linear infinite;
    }
    @keyframes dm-float {
        0%   { transform: translateY(0) translateX(0) scale(1); opacity: 0; }
        10%  { opacity: 0.55; }
        50%  { transform: translateY(-55vh) translateX(20px) scale(1.05); }
        90%  { opacity: 0.35; }
        100% { transform: translateY(-115vh) translateX(-15px) scale(1.15); opacity: 0; }
    }
    .dm-bg span:nth-child(1)  { left: 4%;  width: 18px; height: 18px; animation-duration: 14s; animation-delay: 0s; }
    .dm-bg span:nth-child(2)  { left: 14%; width: 10px; height: 10px; animation-duration: 10s; animation-delay: 2s; }
    .dm-bg span:nth-child(3)  { left: 24%; width: 24px; height: 24px; animation-duration: 18s; animation-delay: 1s; }
    .dm-bg span:nth-child(4)  { left: 36%; width: 8px;  height: 8px;  animation-duration: 9s;  animation-delay: 4s; }
    .dm-bg span:nth-child(5)  { left: 48%; width: 16px; height: 16px; animation-duration: 16s; animation-delay: 0.5s; }
    .dm-bg span:nth-child(6)  { left: 58%; width: 12px; height: 12px; animation-duration: 11s; animation-delay: 3s; }
    .dm-bg span:nth-child(7)  { left: 68%; width: 22px; height: 22px; animation-duration: 20s; animation-delay: 5s; }
    .dm-bg span:nth-child(8)  { left: 76%; width: 9px;  height: 9px;  animation-duration: 8s;  animation-delay: 1.5s; }
    .dm-bg span:nth-child(9)  { left: 85%; width: 15px; height: 15px; animation-duration: 15s; animation-delay: 2.5s; }
    .dm-bg span:nth-child(10) { left: 92%; width: 11px; height: 11px; animation-duration: 12s; animation-delay: 6s; }
    .dm-bg span:nth-child(11) { left: 20%; width: 6px;  height: 6px;  animation-duration: 7s;  animation-delay: 3.5s; }
    .dm-bg span:nth-child(12) { left: 64%; width: 7px;  height: 7px;  animation-duration: 9s;  animation-delay: 4.5s; }

    /* Keep real content above the particle layer */
    .block-container { position: relative; z-index: 1; }

    /* Pinned bottom bar (st.bottom) */
    [data-testid="stBottom"] {
        background: rgba(13,15,26,0.85);
        backdrop-filter: blur(10px);
        border-top: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stBottomBlockContainer"] {
        max-width: 1200px;
        padding-top: 0.6rem;
        padding-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Floating background particles (purely decorative)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dm-bg">
    <span></span><span></span><span></span><span></span><span></span><span></span>
    <span></span><span></span><span></span><span></span><span></span><span></span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dm-header">
    <div class="dm-logo">🧠</div>
    <div>
        <p class="dm-title">DocMind</p>
        <p class="dm-subtitle">Ask questions, get answers grounded in your own documents</p>
    </div>
</div>
""", unsafe_allow_html=True)
st.write("")

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "indexed" not in st.session_state:
    st.session_state["indexed"] = False
if "doc_names" not in st.session_state:
    st.session_state["doc_names"] = []
if "chunk_count" not in st.session_state:
    st.session_state["chunk_count"] = 0

# ─────────────────────────────────────────────────────────────────────────────
# Pinned input bar at the very bottom of the app
# ─────────────────────────────────────────────────────────────────────────────
with st.bottom:
    query = st.chat_input(
        "Ask something about your documents…" if st.session_state["indexed"] else "Build a knowledge base first…",
        disabled=not st.session_state["indexed"],
    )

if query:
    st.session_state["messages"].append({"role": "user", "content": query})

# ─────────────────────────────────────────────────────────────────────────────
# Layout: left = knowledge base controls, right = chat
# ─────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 2], gap="large")

with left:
    st.markdown('<div class="dm-card"><h4>Knowledge Base</h4>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Add source files",
        type=["pdf", "txt", "csv", "xlsx", "docx", "json"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    build_clicked = st.button("⚡ Build knowledge base", use_container_width=True, disabled=not uploaded_files)

    if build_clicked and uploaded_files:
        with st.spinner("Reading and embedding your files…"):
            tmp_dir = tempfile.mkdtemp()
            for uf in uploaded_files:
                path = os.path.join(tmp_dir, uf.name)
                with open(path, "wb") as f:
                    f.write(uf.read())

            docs = load_all_documents(tmp_dir)
            if not docs:
                st.error("Couldn't extract any text from those files.")
            else:
                store = FaissVectorStore(PERSIST_DIR)
                store.build_from_documents(docs)
                st.session_state["indexed"] = True
                st.session_state["doc_names"] = [uf.name for uf in uploaded_files]
                st.session_state["chunk_count"] = len(docs)
                st.success(f"Ready — {len(docs)} document(s) processed.")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state["indexed"]:
        st.markdown('<div class="dm-card"><h4>Indexed Sources</h4>', unsafe_allow_html=True)
        pills = "".join(f'<span class="dm-pill">📄 {name}</span>' for name in st.session_state["doc_names"])
        st.markdown(pills, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="dm-card"><h4>Retrieval Settings</h4>', unsafe_allow_html=True)
    top_k = st.slider("Chunks per answer", 1, 10, 5, label_visibility="visible")
    st.markdown(f"""
        <div class="dm-stat"><span class="dm-stat-label">Status</span>
            <span class="dm-stat-value">{"🟢 Ready" if st.session_state["indexed"] else "⚪ Not indexed"}</span></div>
        <div class="dm-stat"><span class="dm-stat-label">Documents processed</span>
            <span class="dm-stat-value">{st.session_state["chunk_count"]}</span></div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    chat_container = st.container(height=560, border=False)

    with chat_container:
        if not st.session_state["messages"]:
            st.markdown("""
            <div class="dm-empty">
                <div class="dm-empty-icon">💬</div>
                Build your knowledge base on the left, then ask a question here.
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state["messages"]:
                role = msg["role"]
                avatar = "🧑" if role == "user" else "🧠"
                st.markdown(f"""
                <div class="dm-msg-row {role}">
                    <div class="dm-avatar {role}">{avatar}</div>
                    <div class="dm-bubble {role}">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)

        if query:
            st.markdown("""
            <div class="dm-msg-row assistant">
                <div class="dm-avatar assistant">🧠</div>
                <div class="dm-bubble assistant dm-thinking">
                    <span class="dm-dot"></span><span class="dm-dot"></span><span class="dm-dot"></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    if query:
        try:
            rag = RAGSearch(persist_dir=PERSIST_DIR)
            rag.vectorstore.load()
            answer = rag.search_and_summarize(query, top_k=top_k)
        except EnvironmentError as e:
            answer = f"⚠️ Setup issue: {e}"
        except Exception as e:
            answer = f"⚠️ Something went wrong: {e}"

        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.rerun()