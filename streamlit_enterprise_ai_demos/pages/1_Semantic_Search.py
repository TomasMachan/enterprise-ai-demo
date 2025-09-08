import streamlit as st
import os, glob
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🔍 Semantic Search in Documents")
st.caption("Upload files or use the built-in samples. Ask questions in natural language.")

st.sidebar.header("Documents")
use_samples = st.sidebar.checkbox("Use sample documents", value=True)
uploaded = st.sidebar.file_uploader("Or upload .txt files", type=["txt"], accept_multiple_files=True)

corpus = []
meta = []
if use_samples:
    docs_dir = "sample_documents"
    for path in glob.glob(os.path.join(docs_dir, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        corpus.append(raw)
        meta.append(os.path.basename(path))

if uploaded:
    for up in uploaded:
        text = up.read().decode("utf-8", errors="ignore")
        corpus.append(text)
        meta.append(up.name)

if not corpus:
    st.info("Add some documents from the sidebar to begin.")
    st.stop()

def chunk_text(text, max_chars=600):
    parts = [p.strip() for p in text.splitlines() if p.strip()]
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) < max_chars:
            cur += (" " + p if cur else p)
        else:
            chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks

rows = []
for fname, doc in zip(meta, corpus):
    for c in chunk_text(doc):
        rows.append({"file": fname, "text": c})
df = pd.DataFrame(rows)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

@st.cache_data(show_spinner=False)
def build_index(texts):
    embs = model.encode(texts, normalize_embeddings=True)
    return embs

embs = build_index(df["text"].tolist())

q = st.text_input("Ask a question", value="What is the notice period?")
top_k = st.slider("Top-K results", 1, 10, 3)

if st.button("Search"):
    q_emb = model.encode([q], normalize_embeddings=True)
    sims = cosine_similarity(q_emb, embs)[0]
    idx = sims.argsort()[::-1][:top_k]
    st.subheader("Results")
    for i in idx:
        st.markdown(f"**{df.iloc[i]['file']}** — score: `{sims[i]:.3f}`")
        st.write(df.iloc[i]["text"])
        st.markdown("---")
