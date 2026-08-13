"""
app.py
------
TailorTalk — AI Saree Similarity Search
Streamlit chat application.

Run locally:
    streamlit run app.py

Deploy:
    Push to GitHub → Streamlit Community Cloud → set GOOGLE_API_KEY in secrets.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Page config — MUST be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TailorTalk · Saree Search",
    page_icon="🪡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "AI-powered fine-grained saree visual similarity search — Byrappa Silks.",
    },
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark fashion aesthetic
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,600;1,400&display=swap');

:root {
    --bg-base:      #0D0D1A;
    --bg-elevated:  #13132A;
    --bg-card:      rgba(255,255,255,0.04);
    --border:       rgba(255,255,255,0.08);
    --gold:         #C9A96E;
    --gold-light:   #E8C98A;
    --purple:       #7B61FF;
    --purple-light: #A594FF;
    --text-primary: #EEEEF5;
    --text-muted:   #8888AA;
    --success:      #4CAF82;
    --shadow:       0 8px 32px rgba(0,0,0,0.4);
    --radius-lg:    16px;
    --radius-md:    10px;
    --radius-sm:    6px;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-base) !important;
    font-family: 'Outfit', sans-serif;
    color: var(--text-primary);
}
[data-testid="stSidebar"] {
    background: var(--bg-elevated) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stHeader"] { background: transparent !important; }
footer { visibility: hidden; }

.brand-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    background: linear-gradient(135deg, var(--gold), var(--purple-light));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0; line-height: 1.2;
}
.brand-subtitle {
    font-size: 0.8rem; color: var(--text-muted);
    letter-spacing: 0.12em; text-transform: uppercase; margin-top: 4px;
}
.sidebar-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 16px 0;
}
.stat-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 20px; padding: 4px 12px;
    font-size: 0.78rem; color: var(--gold); margin: 4px 2px;
}
.section-label {
    font-size: 0.7rem; letter-spacing: 0.15em; text-transform: uppercase;
    color: var(--text-muted); margin-bottom: 8px; margin-top: 16px;
}

[data-testid="stFileUploader"] {
    background: var(--bg-card); border: 1px dashed var(--border);
    border-radius: var(--radius-md); padding: 4px; transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--gold); }

[data-testid="stTextInput"] input {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important; color: var(--text-primary) !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(201,169,110,0.15) !important;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #7B61FF, #C9A96E) !important;
    border: none !important; border-radius: var(--radius-md) !important;
    color: #fff !important; font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important; font-size: 0.9rem !important;
    padding: 10px 20px !important; transition: opacity 0.2s, transform 0.1s !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

.chat-header { display: flex; align-items: center; gap: 12px; padding: 20px 24px 12px; border-bottom: 1px solid var(--border); margin-bottom: 12px; }
.chat-header-title { font-family: 'Playfair Display', serif; font-size: 1.4rem; color: var(--text-primary); margin: 0; }
.chat-header-status { display: flex; align-items: center; gap: 6px; font-size: 0.75rem; color: var(--success); margin: 0; }
.status-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--success); animation: pulse 2s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

[data-testid="stChatInput"] {
    border: 1px solid var(--border) !important; border-radius: var(--radius-lg) !important;
    background: var(--bg-elevated) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--gold) !important; box-shadow: 0 0 0 3px rgba(201,169,110,0.12) !important;
}

.result-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(168px, 1fr));
    gap: 14px; margin-top: 14px;
}
.result-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-md); overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
}
.result-card:hover { transform: translateY(-4px); box-shadow: var(--shadow); border-color: var(--gold); }
.result-card img { width: 100%; height: 180px; object-fit: cover; display: block; }
.result-card-body { padding: 8px 10px; }
.result-filename { font-size: 0.72rem; color: var(--text-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 3px; }
.result-name { font-size: 0.78rem; color: var(--text-primary); line-height: 1.3; margin-bottom: 5px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; min-height: 2rem; }
.result-price-row { margin-bottom: 6px; font-size: 0.82rem; }
.result-footer { display: flex; align-items: center; justify-content: space-between; gap: 6px; }
.result-score { display: inline-flex; align-items: center; gap: 4px; font-size: 0.78rem; font-weight: 600; padding: 2px 8px; border-radius: 20px; }
.view-btn { font-size: 0.72rem; font-weight: 600; color: var(--gold); text-decoration: none; padding: 2px 8px; border: 1px solid rgba(201,169,110,0.35); border-radius: 20px; transition: background 0.15s; white-space: nowrap; }
.view-btn:hover { background: rgba(201,169,110,0.12); }
.score-high   { background: rgba(76,175,130,0.18);  color: #4CAF82; }
.score-medium { background: rgba(201,169,110,0.18); color: var(--gold); }
.score-low    { background: rgba(136,136,170,0.18); color: var(--text-muted); }

.query-preview { border-radius: var(--radius-md); overflow: hidden; border: 2px solid var(--gold); max-width: 220px; margin: 8px 0; }
.query-preview img { width: 100%; height: auto; display: block; }

.welcome-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 32px; text-align: center; max-width: 500px; margin: 60px auto; }
.welcome-icon { font-size: 3rem; margin-bottom: 12px; }
.welcome-title { font-family: 'Playfair Display', serif; font-size: 1.6rem; color: var(--text-primary); margin-bottom: 10px; }
.welcome-body { font-size: 0.9rem; color: var(--text-muted); line-height: 1.6; }
.welcome-hint { margin-top: 20px; font-size: 0.8rem; color: var(--gold); background: rgba(201,169,110,0.08); border-radius: var(--radius-sm); padding: 8px 14px; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
.stMarkdown p { color: var(--text-primary); }
hr { border-color: var(--border) !important; }
</style>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pil_to_b64(image: Image.Image, max_size: int = 300, quality: int = 75) -> str:
    image = image.copy()
    image.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def score_class(sim: float) -> str:
    if sim >= 0.80:
        return "score-high"
    elif sim >= 0.65:
        return "score-medium"
    return "score-low"


# ---------------------------------------------------------------------------
# Cached resource loaders (once per Streamlit session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI model …")
def load_embedder():
    from embedder import FashionEmbedder
    return FashionEmbedder()


@st.cache_resource(show_spinner="Loading saree index …")
def load_index():
    from search import SareeIndex
    return SareeIndex.load()


def get_agent(embedder, index, results_store):
    from agent import build_agent
    api_key = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
    return build_agent(embedder, index, results_store, google_api_key=api_key)


# ---------------------------------------------------------------------------
# Render result cards
# ---------------------------------------------------------------------------
def render_results(results: List[Dict[str, Any]]) -> None:
    """Render a responsive grid of rich product result cards."""
    if not results:
        return

    cards_html = '<div class="result-grid">'
    for r in results:
        thumb    = r.get("thumbnail_b64", "")
        img_url  = r.get("image_url", "")
        sim      = r.get("similarity", 0.0)
        rank     = r.get("rank", "?")
        name     = r.get("name", r.get("filename", "Saree"))
        sku      = r.get("sku", "")
        disc_p   = r.get("discounted_price", "")
        ret_p    = r.get("retail_price", "")
        link     = r.get("website_link", "")
        css_cls  = score_class(sim)

        img_src = thumb if thumb else img_url
        img_tag = (
            f'<img src="{img_src}" alt="{name}" loading="lazy" />'
            if img_src
            else '<div style="height:180px;background:#1A1A30;display:flex;align-items:center;justify-content:center;color:#555;font-size:0.8rem">No Preview</div>'
        )

        if disc_p and ret_p:
            price_html = (
                f'<span style="color:var(--gold);font-weight:600">&#8377;{disc_p}</span> '
                f'<span style="color:var(--text-muted);font-size:0.7rem;text-decoration:line-through">&#8377;{ret_p}</span>'
            )
        elif disc_p:
            price_html = f'<span style="color:var(--gold);font-weight:600">&#8377;{disc_p}</span>'
        else:
            price_html = ""

        link_btn = (
            f'<a href="{link}" target="_blank" class="view-btn">View &#8599;</a>'
            if link else ""
        )
        short_name = (name[:52] + "…") if len(name) > 55 else name
        star = "&#9733;" if sim >= 0.80 else "&#9678;"

        cards_html += f"""
        <div class="result-card">
            {img_tag}
            <div class="result-card-body">
                <div class="result-filename" title="{name}">#{rank} · {sku}</div>
                <div class="result-name">{short_name}</div>
                <div class="result-price-row">{price_html}</div>
                <div class="result-footer">
                    <span class="result-score {css_cls}">{star}&nbsp;{sim:.0%}</span>
                    {link_btn}
                </div>
            </div>
        </div>
        """
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Session state ─────────────────────────────────────────────────────
    defaults = {
        "messages": [],
        "results_store": [],
        "result_history": [],
        "pending_image_path": None,
        "pending_image_b64": None,
        "agent": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Load resources ────────────────────────────────────────────────────
    try:
        embedder = load_embedder()
        saree_index = load_index()
    except FileNotFoundError as exc:
        st.error(
            f"**Index not found:** {exc}\n\n"
            "Build the index first:\n"
            "```\npython scripts/build_index.py --csv data/images/byrappa_tejas_31july.csv\n```"
        )
        st.stop()
    except Exception as exc:
        st.error(f"**Startup error:** {exc}")
        st.stop()

    if st.session_state.agent is None:
        try:
            st.session_state.agent = get_agent(
                embedder, saree_index, st.session_state.results_store
            )
        except ValueError as exc:
            st.error(str(exc))
            st.info("Add `GOOGLE_API_KEY = \"your-key\"` to `.streamlit/secrets.toml`.")
            st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<p class="brand-title">TailorTalk</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="brand-subtitle">AI Saree Similarity Search</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        st.markdown(
            f'<span class="stat-pill">🪡 {saree_index.total_vectors:,} sarees indexed</span>'
            f'<span class="stat-pill">✦ Marqo FashionSigLIP</span>'
            f'<span class="stat-pill">🏷️ Byrappa Silks</span>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        st.markdown('<p class="section-label">Upload a saree image</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "upload_image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            key="file_uploader",
        )

        st.markdown(
            '<p class="section-label" style="margin-top:12px">… or paste an image URL</p>',
            unsafe_allow_html=True,
        )
        url_input = st.text_input(
            "url_input",
            placeholder="https://example.com/saree.webp",
            label_visibility="collapsed",
            key="url_box",
        )

        search_clicked = st.button("🔍  Find Similar Sarees", key="search_btn")

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        with st.expander("ℹ️ How it works", expanded=False):
            st.markdown(
                """
**Model**: Marqo-FashionSigLIP — fine-tuned for fashion with Generalised
Contrastive Learning across 7 fashion aspects.

**Extra colour signal**: HSV colour histogram fused with visual embedding
(85% visual + 15% colour) for better fine-grained discrimination.

**Search**: Exact cosine similarity via FAISS `IndexFlatIP`.

**Agent**: Gemini 1.5 Flash orchestrates tool calls and writes the response.
                """
            )

        if st.button("🗑️  Clear chat", key="clear_btn"):
            for k in ["messages", "result_history", "pending_image_path", "pending_image_b64"]:
                st.session_state[k] = [] if k in ("messages", "result_history") else None
            st.session_state.results_store.clear()
            st.rerun()

    # ── Handle search click ───────────────────────────────────────────────
    if search_clicked:
        image_source: Optional[str] = None
        if uploaded is not None:
            suffix = os.path.splitext(uploaded.name)[-1] or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                image_source = tmp.name
            pil_img = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
            st.session_state.pending_image_b64 = pil_to_b64(pil_img)
        elif url_input.strip():
            image_source = url_input.strip()
            st.session_state.pending_image_b64 = None

        if image_source:
            st.session_state.pending_image_path = image_source
            st.session_state.messages.append({
                "role": "user",
                "content": "Find sarees visually similar to the image I've provided.",
            })
            st.rerun()

    # ── Chat header ───────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="chat-header">
            <div>
                <p class="chat-header-title">Aria — Saree Stylist</p>
                <p class="chat-header-status">
                    <span class="status-dot"></span> Online · Gemini 1.5 Flash
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Welcome screen ────────────────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="welcome-card">
                <div class="welcome-icon">🪡</div>
                <h2 class="welcome-title">Welcome to TailorTalk</h2>
                <p class="welcome-body">
                    Upload a saree image from the sidebar and I'll find the most
                    visually similar designs from the Byrappa Silks catalogue —
                    over 1,000 unique sarees. I can spot similarities in fabric,
                    weave, colour, border and pallu work.
                </p>
                <div class="welcome-hint">
                    ← Upload or paste a URL, then click <strong>Find Similar Sarees</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Render existing conversation ──────────────────────────────────────
    result_idx = 0
    for i, msg in enumerate(st.session_state.messages):
        role = msg["role"]
        with st.chat_message(role, avatar="👤" if role == "user" else "🪡"):
            if (
                role == "user"
                and msg.get("show_image")
                and msg.get("image_b64")
            ):
                st.markdown(
                    f'<div class="query-preview"><img src="{msg["image_b64"]}" /></div>',
                    unsafe_allow_html=True,
                )
            st.write(msg["content"])
            if role == "assistant" and msg.get("has_results"):
                if result_idx < len(st.session_state.result_history):
                    render_results(st.session_state.result_history[result_idx])
                    result_idx += 1

    # ── Run agent when image is pending ───────────────────────────────────
    last_role = st.session_state.messages[-1]["role"] if st.session_state.messages else None

    if last_role == "user" and st.session_state.pending_image_path:
        if st.session_state.pending_image_b64:
            st.session_state.messages[-1]["show_image"] = True
            st.session_state.messages[-1]["image_b64"] = st.session_state.pending_image_b64

            with st.chat_message("user", avatar="👤"):
                st.markdown(
                    f'<div class="query-preview"><img src="{st.session_state.pending_image_b64}" /></div>',
                    unsafe_allow_html=True,
                )
                st.write(st.session_state.messages[-1]["content"])

        with st.chat_message("assistant", avatar="🪡"):
            with st.spinner("Searching through 1,000+ sarees …"):
                from langchain_core.messages import AIMessage, HumanMessage
                lc_history = [
                    HumanMessage(content=m["content"]) if m["role"] == "user"
                    else AIMessage(content=m["content"])
                    for m in st.session_state.messages[:-1]
                ]
                user_input = (
                    f"The user has provided a saree image at path: "
                    f"'{st.session_state.pending_image_path}'. "
                    f"{st.session_state.messages[-1]['content']}"
                )
                try:
                    reply = st.session_state.agent.chat(
                        user_input=user_input,
                        history=lc_history,
                    )
                except Exception as exc:
                    logger.error("Agent error: %s", exc, exc_info=True)
                    reply = "I'm sorry, something went wrong. Please check your API key and try again."

            st.write(reply)
            current_results = list(st.session_state.results_store)
            has_results = bool(current_results)
            if has_results:
                st.session_state.result_history.append(current_results)
                render_results(current_results)

        st.session_state.messages.append({
            "role": "assistant",
            "content": reply,
            "has_results": has_results,
        })
        st.session_state.pending_image_path = None
        st.session_state.pending_image_b64 = None
        st.rerun()

    # ── Free-text chat ────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask me about sarees, or upload an image to search …"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.write(prompt)

        with st.chat_message("assistant", avatar="🪡"):
            with st.spinner("Thinking …"):
                from langchain_core.messages import AIMessage, HumanMessage
                lc_history = [
                    HumanMessage(content=m["content"]) if m["role"] == "user"
                    else AIMessage(content=m["content"])
                    for m in st.session_state.messages[:-1]
                ]
                try:
                    reply = st.session_state.agent.chat(
                        user_input=prompt,
                        history=lc_history,
                    )
                except Exception as exc:
                    logger.error("Agent error: %s", exc, exc_info=True)
                    reply = "Something went wrong. Please try again."
            st.write(reply)

        st.session_state.messages.append({
            "role": "assistant",
            "content": reply,
            "has_results": False,
        })
        st.rerun()


if __name__ == "__main__":
    main()
