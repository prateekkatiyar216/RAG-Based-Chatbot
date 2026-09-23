import streamlit as st
import streamlit.components.v1 as components
import os
import html as _html
import tempfile
import markdown as _markdown
from html.parser import HTMLParser
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch
from src.excel_analyzer import (
    load_excel_file,
    ExcelLoadError,
    get_file_metadata,
    classify_question,
    analyze_excel_question,
    PREVIEW_ROWS,
)

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
        transition: border-color 260ms ease, box-shadow 260ms ease, background 260ms ease;
        max-width: 92%;
        min-width: 0;   /* let the table wrapper below shrink instead of overflowing */
    }
    /* Soft blue glow on hover — answers only, not the thinking bubble */
    .dm-bubble.assistant:not(.dm-thinking):hover {
        border-color: rgba(50,150,255,0.6);
        box-shadow: 0 0 12px rgba(50,150,255,0.18), 0 0 25px rgba(50,150,255,0.08);
        background: rgba(255,255,255,0.075);
    }

    /* Entrance animations — applied once, only to a message the first time it
       is rendered (see `animated_count` in the render loop). */
    .dm-msg-row.dm-enter-user {
        animation: dm-rise-user 520ms cubic-bezier(0.22, 1, 0.36, 1) both;
    }
    .dm-msg-row.dm-enter-assistant {
        animation: dm-rise-assistant 560ms cubic-bezier(0.22, 1, 0.36, 1) both;
    }
    @keyframes dm-rise-user {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes dm-rise-assistant {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .dm-msg-row.dm-enter-assistant .dm-bubble.assistant {
        animation: dm-glow-settle 900ms ease-out 1;
    }
    @keyframes dm-glow-settle {
        0%   { box-shadow: 0 0 18px rgba(50,150,255,0.16); }
        100% { box-shadow: 0 0 0 rgba(50,150,255,0); }
    }
    /* The thinking bubble fades in too, so the hand-off reads smoothly */
    .dm-msg-row.dm-enter-thinking {
        animation: dm-rise-assistant 380ms cubic-bezier(0.22, 1, 0.36, 1) both;
    }
    @media (prefers-reduced-motion: reduce) {
        .dm-msg-row.dm-enter-user,
        .dm-msg-row.dm-enter-assistant,
        .dm-msg-row.dm-enter-thinking,
        .dm-msg-row.dm-enter-assistant .dm-bubble.assistant { animation: none; }
    }
    .dm-avatar {
        width: 30px; height: 30px; border-radius: 9px;
        display: flex; align-items: center; justify-content: center;
        font-size: 15px; flex-shrink: 0;
    }
    .dm-avatar.user { background: rgba(124,92,255,0.25); margin-left: 10px; order: 2; }
    .dm-avatar.assistant { background: rgba(34,211,238,0.18); margin-right: 10px; }

    .dm-bubble p { margin: 0 0 8px 0; }
    .dm-bubble p:last-child { margin-bottom: 0; }
    .dm-bubble code {
        background: rgba(0,0,0,0.35);
        padding: 1px 5px; border-radius: 5px;
        font-size: 0.86em;
    }

    /* ── Rendered Markdown inside assistant answers ──────────────────── */
    .dm-bubble.assistant :first-child { margin-top: 0; }
    .dm-bubble.assistant :last-child { margin-bottom: 0; }
    .dm-bubble.assistant strong { color: #f4f4fb; font-weight: 600; }
    .dm-bubble.assistant em { color: inherit; }
    .dm-bubble.assistant h1, .dm-bubble.assistant h2, .dm-bubble.assistant h3,
    .dm-bubble.assistant h4, .dm-bubble.assistant h5, .dm-bubble.assistant h6 {
        font-family: 'Space Grotesk', sans-serif;
        color: #f4f4fb;
        font-weight: 600;
        line-height: 1.3;
        margin: 14px 0 6px 0;
    }
    .dm-bubble.assistant h1 { font-size: 1.25rem; }
    .dm-bubble.assistant h2 { font-size: 1.12rem; }
    .dm-bubble.assistant h3 { font-size: 1.02rem; }
    .dm-bubble.assistant h4, .dm-bubble.assistant h5, .dm-bubble.assistant h6 { font-size: 0.94rem; }
    .dm-bubble.assistant ul, .dm-bubble.assistant ol {
        margin: 6px 0 10px 0;
        padding-left: 1.35em;
    }
    .dm-bubble.assistant li { margin: 3px 0; }
    .dm-bubble.assistant li > ul, .dm-bubble.assistant li > ol { margin: 3px 0 3px 0; }
    .dm-bubble.assistant blockquote {
        margin: 8px 0;
        padding: 4px 12px;
        border-left: 3px solid rgba(124,92,255,0.5);
        color: #b7bad4;
    }
    .dm-bubble.assistant hr {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.1);
        margin: 12px 0;
    }
    .dm-bubble.assistant pre {
        background: rgba(0,0,0,0.35);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 10px;
        padding: 12px 14px;
        margin: 10px 0;
        overflow-x: auto;
        max-width: 100%;
    }
    .dm-bubble.assistant pre code {
        background: none;
        padding: 0;
        font-size: 0.85em;
        white-space: pre;
        color: #d8dcf5;
    }

    /* Markdown tables — kept inside a horizontally scrollable wrapper so a
       wide table never pushes the bubble or the page sideways. */
    .dm-table-wrapper {
        width: 100%;
        max-width: 100%;
        overflow-x: auto;
        margin: 12px 0;
        border-radius: 12px;
        overscroll-behavior-x: contain;
    }
    .dm-table-wrapper::-webkit-scrollbar { height: 7px; }
    .dm-table-wrapper::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.14); border-radius: 8px;
    }
    .dm-table-wrapper::-webkit-scrollbar-track { background: transparent; }

    .dm-markdown-table {
        width: 100%;
        min-width: 320px;
        border-collapse: separate;
        border-spacing: 0;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 12px;
        overflow: hidden;
        font-size: 0.88rem;
    }
    .dm-markdown-table th, .dm-markdown-table td {
        padding: 9px 13px;
        text-align: left;
        vertical-align: top;
        border-bottom: 1px solid rgba(255,255,255,0.07);
        white-space: normal;
        word-break: break-word;
    }
    .dm-markdown-table th {
        background: rgba(124,92,255,0.14);
        color: #d9d5ff;
        font-weight: 600;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.82rem;
        letter-spacing: 0.2px;
        white-space: nowrap;
    }
    .dm-markdown-table tbody tr:last-child td { border-bottom: none; }
    .dm-markdown-table tbody tr:hover td { background: rgba(255,255,255,0.03); }

    /* ── Chat transcript area ────────────────────────────────────────────
       The container is created with a fixed pixel height by Streamlit; these
       rules relax it so it grows with the conversation and only starts
       scrolling once it would outgrow the viewport. `.dm-chat-marker` is an
       empty element rendered as the first child of the chat container, so
       :has() lets us target that one container and nothing else. */
    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-marker) {
        height: auto !important;
        min-height: 260px;
        max-height: calc(100vh - 250px) !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-marker) > div {
        max-height: calc(100vh - 250px);
        overflow-y: auto;
        overflow-x: hidden;
        overscroll-behavior: contain;   /* don't hand scrolling to the page */
        padding-right: 6px;
    }
    /* Fallback for browsers without :has() — keeps the old fixed-height behaviour */
    @supports not selector(:has(*)) {
        [data-testid="stVerticalBlockBorderWrapper"] > div { overscroll-behavior: contain; }
    }
    .dm-chat-marker { height: 0; margin: 0; }
    #dm-chat-end { height: 1px; margin: 0; }

    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-marker) > div::-webkit-scrollbar { width: 8px; }
    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-marker) > div::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.12); border-radius: 8px;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:has(.dm-chat-marker) > div::-webkit-scrollbar-track {
        background: transparent;
    }

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
# Excel workbooks: {filename: {sheet_name: DataFrame}}, kept as structured
# data (never flattened into the PDF/RAG text pipeline).
if "excel_data" not in st.session_state:
    st.session_state["excel_data"] = {}
# Excel metadata for display: {filename: {"filename":..., "sheets": [...]}}
if "excel_meta" not in st.session_state:
    st.session_state["excel_meta"] = {}
# Question waiting to be answered. Set on submit, cleared once the answer lands.
if "pending" not in st.session_state:
    st.session_state["pending"] = None
# UI-only: how many messages have already played their entrance animation.
if "animated_count" not in st.session_state:
    st.session_state["animated_count"] = 0


# ─────────────────────────────────────────────────────────────────────────────
# Safe Markdown rendering for assistant answers.
#
# The RAG answer is plain text that may contain Markdown (tables, bold,
# headings, lists, code). We convert it to HTML with python-markdown, then
# run the result through an allowlist sanitizer before it ever reaches
# `unsafe_allow_html=True` — so any raw HTML/script the model's output
# happens to contain (accidentally or via a crafted document) is stripped,
# not executed. User messages are never Markdown-rendered; they stay
# plain-escaped text, as before.
# ─────────────────────────────────────────────────────────────────────────────
_MD_ALLOWED_TAGS = {
    "p", "br", "strong", "b", "em", "i", "u",
    "ul", "ol", "li",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "code", "pre", "blockquote", "hr",
    "table", "thead", "tbody", "tr", "th", "td",
}
_MD_VOID_TAGS = {"br", "hr"}
# Tags whose entire content (not just the tag) must never reach the page.
_MD_DROP_CONTENT_TAGS = {
    "script", "style", "iframe", "object", "embed", "form",
    "input", "button", "select", "textarea", "link", "meta", "svg", "noscript",
}


class _SafeMarkdownHTML(HTMLParser):
    """Allowlist HTML sanitizer for python-markdown output.

    Any tag not in `_MD_ALLOWED_TAGS` is dropped (its text content is kept,
    re-escaped); any tag in `_MD_DROP_CONTENT_TAGS` is dropped along with
    everything inside it. No attributes are ever passed through — the one
    exception is that a `<table>` is given a fixed `class` and wrapped in a
    `.dm-table-wrapper` div so wide tables scroll instead of breaking the
    chat layout.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self._drop_depth = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in _MD_DROP_CONTENT_TAGS:
            self._drop_depth += 1
            return
        if self._drop_depth:
            return
        if tag not in _MD_ALLOWED_TAGS:
            return
        if tag == "table":
            self.out.append('<div class="dm-table-wrapper"><table class="dm-markdown-table">')
            return
        self.out.append(f"<{tag}>")

    def handle_startendtag(self, tag, attrs):
        tag = tag.lower()
        if self._drop_depth or tag in _MD_DROP_CONTENT_TAGS:
            return
        if tag in _MD_ALLOWED_TAGS:
            self.out.append(f"<{tag}/>")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in _MD_DROP_CONTENT_TAGS:
            if self._drop_depth:
                self._drop_depth -= 1
            return
        if self._drop_depth:
            return
        if tag not in _MD_ALLOWED_TAGS or tag in _MD_VOID_TAGS:
            return
        if tag == "table":
            self.out.append("</table></div>")
            return
        self.out.append(f"</{tag}>")

    def handle_data(self, data):
        if self._drop_depth:
            return
        self.out.append(_html.escape(data))

    def get_html(self) -> str:
        return "".join(self.out)


def markdown_to_safe_html(text: str) -> str:
    """Render assistant Markdown (tables, bold, headings, lists, code) to
    sanitized HTML. Never raises — falls back to plain escaped text."""
    try:
        raw_html = _markdown.markdown(
            str(text),
            extensions=["tables", "fenced_code", "sane_lists"],
            output_format="html5",
        )
        parser = _SafeMarkdownHTML()
        parser.feed(raw_html)
        parser.close()
        return parser.get_html()
    except Exception:
        return _html.escape(str(text)).replace("\n", "<br>")


def render_message(role: str, content: str, animate: bool = False) -> str:
    """Render a chat bubble. Assistant text is rendered as sanitized
    Markdown; user text stays plain-escaped (never Markdown-rendered)."""
    avatar = "🧑" if role == "user" else "🧠"
    enter = f" dm-enter-{role}" if animate else ""
    if role == "assistant":
        body = markdown_to_safe_html(content)
    else:
        body = _html.escape(str(content)).replace("\n", "<br>")
    return f"""
    <div class="dm-msg-row {role}{enter}">
        <div class="dm-avatar {role}">{avatar}</div>
        <div class="dm-bubble {role}">{body}</div>
    </div>
    """

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
        # Excel files go through the structured DataFrame pipeline; every
        # other supported type keeps going through the existing PDF/RAG
        # pipeline untouched.
        excel_files = [uf for uf in uploaded_files if uf.name.lower().endswith(".xlsx")]
        other_files = [uf for uf in uploaded_files if not uf.name.lower().endswith(".xlsx")]

        # ── PDF / document RAG pipeline (unchanged) ────────────────────
        if other_files:
            with st.spinner("Reading and embedding your files…"):
                tmp_dir = tempfile.mkdtemp()
                for uf in other_files:
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
                    st.session_state["doc_names"] = [uf.name for uf in other_files]
                    st.session_state["chunk_count"] = len(docs)
                    st.success(f"Ready — {len(docs)} document(s) processed.")

        # ── Excel structured-data pipeline ──────────────────────────────
        if excel_files:
            with st.spinner("Reading your spreadsheets…"):
                loaded_any = False
                for uf in excel_files:
                    tmp_dir = tempfile.mkdtemp()
                    path = os.path.join(tmp_dir, uf.name)
                    with open(path, "wb") as f:
                        f.write(uf.read())
                    try:
                        sheets = load_excel_file(path)
                        st.session_state["excel_data"][uf.name] = sheets
                        st.session_state["excel_meta"][uf.name] = get_file_metadata(uf.name, sheets)
                        loaded_any = True
                    except ExcelLoadError as e:
                        st.warning(f"⚠️ Skipped **{uf.name}** — {e}")
                if loaded_any:
                    total_sheets = sum(len(m["sheets"]) for m in st.session_state["excel_meta"].values())
                    st.success(f"Ready — {len(st.session_state['excel_data'])} workbook(s), {total_sheets} sheet(s) processed.")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state["indexed"] or st.session_state["excel_data"]:
        st.markdown('<div class="dm-card"><h4>Indexed Sources</h4>', unsafe_allow_html=True)
        pills = "".join(f'<span class="dm-pill">📄 {name}</span>' for name in st.session_state["doc_names"])
        pills += "".join(f'<span class="dm-pill">📊 {name}</span>' for name in st.session_state["excel_data"])
        st.markdown(pills, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state["excel_meta"]:
        st.markdown('<div class="dm-card"><h4>Excel Sheets</h4>', unsafe_allow_html=True)
        for meta in st.session_state["excel_meta"].values():
            for s in meta["sheets"]:
                st.markdown(f"""
                    <div class="dm-stat"><span class="dm-stat-label">{meta['filename']} — {s['name']}</span>
                        <span class="dm-stat-value">{s['rows']:,} × {s['cols']}</span></div>
                """, unsafe_allow_html=True)
        with st.expander("Preview sheet data"):
            for filename, sheets in st.session_state["excel_data"].items():
                for sheet_name, df in sheets.items():
                    st.caption(f"{filename} — {sheet_name}")
                    st.dataframe(df.head(PREVIEW_ROWS), use_container_width=True)
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

# ─────────────────────────────────────────────────────────────────────────────
# Pinned input bar — captured AFTER the left column so "indexed" is fresh,
# and laid out with the same [1, 2] ratio so it aligns under the right column
# ─────────────────────────────────────────────────────────────────────────────
is_busy = st.session_state["pending"] is not None
has_kb = st.session_state["indexed"] or bool(st.session_state["excel_data"])

with st.bottom:
    _, input_col = st.columns([1, 2], gap="large")
    with input_col:
        if not has_kb:
            placeholder = "Build a knowledge base first…"
        elif is_busy:
            placeholder = "Thinking…"
        else:
            placeholder = "Ask something about your documents or spreadsheets…"

        query = st.chat_input(
            placeholder,
            disabled=not has_kb or is_busy,
        )

# Step 1 — a question was just submitted: store it and rerun immediately so the
# bubble + thinking dots are painted BEFORE the slow retrieval call starts.
if query and not is_busy:
    st.session_state["messages"].append({"role": "user", "content": query})
    st.session_state["pending"] = query
    st.rerun()

# Step 2 — paint the transcript (this run either shows history, or history +
# the pending question with the animated dots underneath it).
with right:
    # Height is a ceiling, not a fixed box: the CSS above turns it into
    # "grow with the content, cap at the viewport, then scroll".
    chat_container = st.container(height=700, border=False)

    with chat_container:
        # Unique marker so the CSS/JS can identify THIS container only.
        st.markdown('<div class="dm-chat-marker"></div>', unsafe_allow_html=True)

        if not st.session_state["messages"]:
            st.markdown("""
            <div class="dm-empty">
                <div class="dm-empty-icon">💬</div>
                Build your knowledge base on the left, then ask a question here.
            </div>
            """, unsafe_allow_html=True)
        else:
            # Only messages that haven't been painted before get the entrance
            # animation; everything older renders static on every rerun.
            first_new = st.session_state["animated_count"]
            for i, msg in enumerate(st.session_state["messages"]):
                st.markdown(
                    render_message(msg["role"], msg["content"], animate=i >= first_new),
                    unsafe_allow_html=True,
                )
            st.session_state["animated_count"] = len(st.session_state["messages"])

        if is_busy:
            st.markdown("""
            <div class="dm-msg-row assistant dm-enter-thinking">
                <div class="dm-avatar assistant">🧠</div>
                <div class="dm-bubble assistant dm-thinking">
                    <span class="dm-dot"></span><span class="dm-dot"></span><span class="dm-dot"></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # End-of-transcript anchor. The scroll script starts here and walks up
        # to find the real scrollable element, so nothing else on the page moves.
        st.markdown('<div id="dm-chat-end"></div>', unsafe_allow_html=True)

        # Token changes only when a message is added or the thinking state
        # flips, so unrelated reruns (slider, upload) never trigger a scroll.
        _scroll_token = f"{len(st.session_state['messages'])}-{int(is_busy)}"
        components.html(
            f"""
            <script>
            (function () {{
                const token = "{_scroll_token}";
                const doc = window.parent.document;
                const win = window.parent;
                if (win.__dmScrollToken === token) return;

                function findScroller(el) {{
                    let node = el.parentElement;
                    while (node && node !== doc.body) {{
                        const oy = win.getComputedStyle(node).overflowY;
                        if ((oy === "auto" || oy === "scroll") &&
                            node.scrollHeight > node.clientHeight + 4) return node;
                        node = node.parentElement;
                    }}
                    return null;
                }}

                function scrollOnce() {{
                    const anchor = doc.getElementById("dm-chat-end");
                    if (!anchor) return false;
                    const scroller = findScroller(anchor);
                    if (!scroller) return true;   // everything already fits

                    const rows = scroller.querySelectorAll(".dm-msg-row");
                    const last = rows[rows.length - 1];
                    let top = scroller.scrollHeight;

                    // A very tall answer: line its TOP up instead of jumping to
                    // the end, so the beginning of the answer is what you see.
                    if (last && last.offsetHeight > scroller.clientHeight * 0.85) {{
                        const delta = last.getBoundingClientRect().top
                                    - scroller.getBoundingClientRect().top;
                        top = scroller.scrollTop + delta - 12;
                    }}
                    scroller.scrollTo({{ top: top, behavior: "smooth" }});
                    return true;
                }}

                let tries = 0;
                function run() {{
                    if (scrollOnce()) {{
                        win.__dmScrollToken = token;
                        // Re-settle after the slide-in animation changes height.
                        setTimeout(scrollOnce, 320);
                        setTimeout(scrollOnce, 700);
                    }} else if (tries++ < 30) {{
                        requestAnimationFrame(run);
                    }}
                }}
                requestAnimationFrame(run);
            }})();
            </script>
            """,
            height=0,
        )

# Step 3 — now that the UI is on screen, run retrieval, then rerun once more so
# the answer replaces the dots.
if is_busy:
    pending_query = st.session_state["pending"]
    has_pdf = st.session_state["indexed"]
    has_excel = bool(st.session_state["excel_data"])

    def _pdf_answer(q: str) -> str:
        rag = RAGSearch(persist_dir=PERSIST_DIR)
        rag.vectorstore.load()
        return rag.search_and_summarize(q, top_k=top_k)

    def _excel_answer(q: str) -> str:
        return analyze_excel_question(q, st.session_state["excel_data"])

    try:
        if has_pdf and not has_excel:
            answer = _pdf_answer(pending_query)
        elif has_excel and not has_pdf:
            answer = _excel_answer(pending_query)
        else:
            # Both a document knowledge base and Excel data are available —
            # classify the question before deciding which path(s) to run.
            route = classify_question(pending_query)
            if route == "pdf":
                answer = _pdf_answer(pending_query)
            elif route == "excel":
                answer = _excel_answer(pending_query)
            else:  # mixed
                doc_answer = _pdf_answer(pending_query)
                excel_answer = _excel_answer(pending_query)
                answer = (
                    f"**From the document:**\n\n{doc_answer}\n\n"
                    f"**From the Excel data:**\n\n{excel_answer}"
                )
    except EnvironmentError as e:
        answer = f"⚠️ Setup issue: {e}"
    except Exception as e:
        answer = f"⚠️ Something went wrong: {e}"

    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.session_state["pending"] = None
    st.rerun()