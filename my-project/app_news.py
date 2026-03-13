"""
app_news.py
-----------
Fake News Detector powered by Endee Vector DB + Groq LLaMA3.

Paste any news article or headline → Endee retrieves similar
verified/fake articles → Groq gives a verdict with evidence.

Usage:
    streamlit run app_news.py
"""

import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from endee import Endee
from groq import Groq

# ── Config ─────────────────────────────────────────────────────────────────────
INDEX_NAME  = "news_articles"
ENDEE_HOST  = "http://localhost:8080"
GROQ_MODEL  = "llama-3.1-8b-instant"
TOP_K       = 10
# ───────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VerifyAI — Fake News Detector",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Unbounded:wght@400;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace;
    background: #050508;
    color: #e8e8e0;
}

.hero { text-align: center; padding: 2rem 0 1rem 0; }

.hero-title {
    font-family: 'Unbounded', sans-serif;
    font-size: 2.8rem;
    font-weight: 900;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #ff4444, #ff8800, #ffdd00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    color: #555;
    font-size: 0.85rem;
    margin-top: 0.5rem;
}

.verdict-fake {
    background: linear-gradient(135deg, #1a0505, #2d0a0a);
    border: 2px solid #ff4444;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1rem 0;
}

.verdict-real {
    background: linear-gradient(135deg, #051a05, #0a2d0a);
    border: 2px solid #44ff88;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1rem 0;
}

.verdict-uncertain {
    background: linear-gradient(135deg, #1a1505, #2d2505);
    border: 2px solid #ffaa00;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1rem 0;
}

.verdict-label-fake {
    font-family: 'Unbounded', sans-serif;
    font-size: 2.5rem;
    font-weight: 900;
    color: #ff4444;
}

.verdict-label-real {
    font-family: 'Unbounded', sans-serif;
    font-size: 2.5rem;
    font-weight: 900;
    color: #44ff88;
}

.verdict-label-uncertain {
    font-family: 'Unbounded', sans-serif;
    font-size: 2.5rem;
    font-weight: 900;
    color: #ffaa00;
}

.evidence-box {
    background: #0d0d12;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 0.8rem;
    border-left: 4px solid #333;
}

.evidence-fake { border-left-color: #ff4444 !important; }
.evidence-real { border-left-color: #44ff88 !important; }

.evidence-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #ccc;
    margin-bottom: 4px;
}

.evidence-meta {
    font-size: 0.75rem;
    color: #555;
    margin-bottom: 6px;
}

.evidence-text {
    font-size: 0.78rem;
    color: #777;
    line-height: 1.5;
}

.label-badge-fake {
    display: inline-block;
    background: #ff444420;
    border: 1px solid #ff4444;
    color: #ff4444;
    border-radius: 4px;
    padding: 1px 8px;
    font-size: 0.7rem;
    font-weight: 600;
}

.label-badge-real {
    display: inline-block;
    background: #44ff8820;
    border: 1px solid #44ff88;
    color: #44ff88;
    border-radius: 4px;
    padding: 1px 8px;
    font-size: 0.7rem;
    font-weight: 600;
}

.similarity-bar-container {
    background: #1a1a2e;
    border-radius: 4px;
    height: 6px;
    margin-top: 4px;
}

.analysis-box {
    background: #0d0d12;
    border: 1px solid #2a2a3a;
    border-left: 4px solid #ffaa00;
    border-radius: 12px;
    padding: 1.5rem;
    line-height: 1.8;
    font-size: 0.9rem;
    color: #ccc;
    margin: 1rem 0;
}

.stat-card {
    background: #0d0d12;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.stTextArea textarea {
    background: #0d0d12 !important;
    color: #e8e8e0 !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
}

.stButton > button {
    background: linear-gradient(135deg, #ff444420, #ff880020) !important;
    color: #ff8800 !important;
    border: 1px solid #ff8800 !important;
    border-radius: 8px !important;
    font-family: 'Unbounded', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    width: 100% !important;
    padding: 0.6rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🔍 VerifyAI</div>
    <div class="hero-sub">Fake News Detector · Endee Vector Search · Groq LLaMA3 · ISOT Dataset (44K articles)</div>
</div>
""", unsafe_allow_html=True)
st.divider()

# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    groq_key = st.text_input("Groq API Key", type="password",
                              value=os.environ.get("GROQ_API_KEY", ""),
                              help="Free key at https://console.groq.com")
    top_k = st.slider("Similar articles to retrieve", 5, 20, TOP_K)
    endee_host = st.text_input("Endee Host", value=ENDEE_HOST)

    st.divider()
    st.markdown("### 🔗 How it works")
    st.markdown("""
```
News Article / Headline
        ↓
SentenceTransformer
(384-dim embedding)
        ↓
Endee Vector Search
(cosine similarity)
        ↓
Top Similar Articles
(with FAKE/REAL labels)
        ↓
Groq LLaMA3 Analysis
        ↓
Verdict + Evidence
```
    """)

    st.divider()
    st.markdown("### 📊 Dataset")
    st.markdown("**ISOT Fake News Dataset**\n\n44,000+ labeled articles\nSource: University of Victoria\n\nFAKE: fabricated political news\nREAL: verified Reuters articles")

    st.divider()
    st.markdown("### ⚠️ Disclaimer")
    st.markdown("This tool is for educational purposes. Always verify news through multiple trusted sources.")


# ── Cached resources ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_index(host):
    client = Endee()
    client.set_base_url(f"{host}/api/v1")
    return client.get_index(name=INDEX_NAME)


# ── Sample headlines for testing ──────────────────────────────────────────────────
SAMPLES = {
    "🔴 Suspicious headline": "BREAKING: Scientists confirm that drinking bleach cures all diseases, government hiding the truth",
    "🟢 Likely real news": "Federal Reserve raises interest rates by 25 basis points amid inflation concerns",
    "🟡 Ambiguous claim": "New study shows coffee consumption linked to reduced risk of heart disease",
}

# ── Input Section ─────────────────────────────────────────────────────────────────
st.markdown("### 📰 Paste News Article or Headline")

col_samples = st.columns(3)
selected_sample = ""
for i, (label, text) in enumerate(SAMPLES.items()):
    if col_samples[i].button(label, key=f"sample_{i}"):
        selected_sample = text

article_input = st.text_area(
    "Article or headline",
    value=selected_sample,
    placeholder="Paste any news headline or article text here...\n\nExample: 'Government announces new economic policy to boost GDP growth by 5%'",
    height=150,
    label_visibility="collapsed",
)

check_btn = st.button("🔍 VERIFY THIS NEWS", use_container_width=True)


# ── Verification Pipeline ──────────────────────────────────────────────────────────
def compute_verdict(results):
    """Compute fake/real ratio from retrieved similar articles."""
    if not results:
        return "UNCERTAIN", 0, 0, 0

    fake_count = sum(1 for r in results if r.get("meta", {}).get("label") == "FAKE")
    real_count = sum(1 for r in results if r.get("meta", {}).get("label") == "REAL")
    total = fake_count + real_count

    fake_pct = (fake_count / total * 100) if total > 0 else 50
    real_pct = (real_count / total * 100) if total > 0 else 50

    # Weight by similarity scores too
    fake_score = sum(r.get("similarity", 0) for r in results if r.get("meta", {}).get("label") == "FAKE")
    real_score = sum(r.get("similarity", 0) for r in results if r.get("meta", {}).get("label") == "REAL")
    total_score = fake_score + real_score

    weighted_fake = (fake_score / total_score * 100) if total_score > 0 else 50

    if weighted_fake >= 65:
        return "FAKE", weighted_fake, fake_count, real_count
    elif weighted_fake <= 35:
        return "REAL", 100 - weighted_fake, fake_count, real_count
    else:
        return "UNCERTAIN", weighted_fake, fake_count, real_count


if check_btn:
    if not article_input.strip():
        st.error("⚠️ Please paste a news article or headline first.")
        st.stop()
    if not groq_key:
        st.error("⚠️ Please enter your Groq API Key in the sidebar.")
        st.stop()

    # ── Step 1: Vector Search ──────────────────────────────────────────────────
    with st.spinner("🔍 Searching 6,000+ articles in Endee vector DB …"):
        try:
            embedder = load_embedder()
            index = get_index(endee_host)
            article_vec = embedder.encode(article_input).tolist()
            results = index.query(vector=article_vec, top_k=top_k)
        except Exception as e:
            st.error(f"❌ Endee error: {e}\n\nMake sure:\n1. Docker is running\n2. You ran `python ingest_news.py`")
            st.stop()

    if not results:
        st.warning("No similar articles found. Have you run `python ingest_news.py`?")
        st.stop()

    # ── Step 2: Compute Verdict ────────────────────────────────────────────────
    verdict, confidence, fake_count, real_count = compute_verdict(results)

    # ── Step 3: Groq Analysis ──────────────────────────────────────────────────
    with st.spinner("🤖 Groq LLaMA3 is analyzing patterns …"):
        try:
            similar_articles = "\n\n".join(
                f"Article {i+1} [Label: {r.get('meta',{}).get('label','?')}] "
                f"[Similarity: {r.get('similarity',0):.3f}]\n"
                f"Title: {r.get('meta',{}).get('title','')}\n"
                f"Excerpt: {r.get('meta',{}).get('text','')[:200]}"
                for i, r in enumerate(results[:6])
            )

            prompt = f"""You are an expert fact-checker and misinformation analyst.

A user submitted this news article/headline for verification:
"{article_input[:600]}"

Based on semantic similarity search, these are the most similar articles found in our database of 44,000 labeled news articles:

{similar_articles}

Statistical analysis:
- Similar FAKE articles found: {fake_count} out of {top_k}
- Similar REAL articles found: {real_count} out of {top_k}
- Weighted verdict: {verdict} (confidence: {confidence:.1f}%)

Please provide:
1. **Verdict** (FAKE / REAL / UNCERTAIN) with one sentence explanation
2. **Key red flags** or **credibility signals** you notice (2-3 bullet points)
3. **What to check** — specific verification steps for this claim (2 bullet points)
4. **Overall confidence** in your assessment (Low / Medium / High)

Be direct, specific, and educational. Keep response under 200 words."""

            groq_client = Groq(api_key=groq_key)
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            analysis = response.choices[0].message.content.strip()
        except Exception as e:
            analysis = f"Could not generate analysis: {e}"

    # ── Display Verdict ────────────────────────────────────────────────────────
    st.divider()

    if verdict == "FAKE":
        css_class = "verdict-fake"
        label_class = "verdict-label-fake"
        emoji = "🚨"
    elif verdict == "REAL":
        css_class = "verdict-real"
        label_class = "verdict-label-real"
        emoji = "✅"
    else:
        css_class = "verdict-uncertain"
        label_class = "verdict-label-uncertain"
        emoji = "⚠️"

    st.markdown(f"""
<div class="{css_class}">
    <div class="{label_class}">{emoji} {verdict}</div>
    <div style="color:#888; font-size:0.85rem; margin-top:0.5rem;">
        Confidence: {confidence:.1f}% &nbsp;|&nbsp;
        Similar FAKE articles: {fake_count}/{top_k} &nbsp;|&nbsp;
        Similar REAL articles: {real_count}/{top_k}
    </div>
</div>
""", unsafe_allow_html=True)

    # ── Stats Row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Articles Searched", "6,000+")
    c2.metric("Similar Found", top_k)
    c3.metric("FAKE Matches", fake_count)
    c4.metric("REAL Matches", real_count)

    # ── AI Analysis ────────────────────────────────────────────────────────────
    st.markdown("### 🤖 AI Fact-Check Analysis")
    st.markdown(f'<div class="analysis-box">{analysis}</div>', unsafe_allow_html=True)

    # ── Evidence Articles ──────────────────────────────────────────────────────
    st.markdown(f"### 📚 Evidence — Top {len(results)} Similar Articles from Database")
    st.caption("Retrieved by Endee vector similarity search · Sorted by cosine similarity score")

    for i, r in enumerate(results):
        meta = r.get("meta", {})
        label = meta.get("label", "?")
        sim   = r.get("similarity", 0)
        title = meta.get("title", "No title")
        text  = meta.get("text", "")[:200]
        subj  = meta.get("subject", "general")
        date  = meta.get("date", "unknown")

        card_class = "evidence-fake" if label == "FAKE" else "evidence-real"
        badge_class = "label-badge-fake" if label == "FAKE" else "label-badge-real"
        sim_pct = int(sim * 100)

        st.markdown(f"""
<div class="evidence-box {card_class}">
    <div class="evidence-title">
        <span class="{badge_class}">{label}</span>&nbsp;&nbsp;
        #{i+1} · {sim_pct}% similar &nbsp;·&nbsp; {subj} &nbsp;·&nbsp; {date}
    </div>
    <div class="evidence-title" style="margin-top:6px;">{title[:120]}</div>
    <div class="evidence-text" style="margin-top:6px;">{text}…</div>
    <div class="similarity-bar-container">
        <div style="background: {'#ff4444' if label=='FAKE' else '#44ff88'}; width:{sim_pct}%; height:6px; border-radius:4px;"></div>
    </div>
</div>
""", unsafe_allow_html=True)

    # ── Confidence Meter ───────────────────────────────────────────────────────
    st.markdown("### 📊 Confidence Breakdown")
    fake_pct_display = int(fake_count / top_k * 100)
    real_pct_display = int(real_count / top_k * 100)

    col_f, col_r = st.columns(2)
    with col_f:
        st.markdown(f"**🔴 FAKE signals: {fake_pct_display}%**")
        st.progress(fake_pct_display / 100)
    with col_r:
        st.markdown(f"**🟢 REAL signals: {real_pct_display}%**")
        st.progress(real_pct_display / 100)

elif not check_btn:
    st.markdown("""
<div style="text-align:center; padding:3rem; color:#333;">
    <div style="font-size:4rem;">🔍</div>
    <div style="font-size:1.1rem; margin-top:1rem; font-family:'Unbounded',sans-serif;">
        PASTE NEWS ABOVE AND CLICK VERIFY
    </div>
    <div style="font-size:0.8rem; margin-top:0.5rem;">
        Powered by Endee vector similarity search · ISOT Dataset
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()
st.markdown('<p style="text-align:center; color:#222; font-size:0.75rem;">Built with Endee Vector DB · sentence-transformers · Groq LLaMA3 · ISOT Fake News Dataset · Streamlit</p>', unsafe_allow_html=True)
