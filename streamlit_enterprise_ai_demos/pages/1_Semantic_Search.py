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
    # Try multiple possible paths for the sample documents
    possible_paths = [
        "sample_documents",  # Original relative path
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_documents"),  # From pages/ to root
        os.path.join(os.getcwd(), "sample_documents"),  # From current working directory
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sample_documents"),  # Alternative relative path
    ]
    
    docs_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            docs_dir = path
            break
    
    if docs_dir:
        txt_files = glob.glob(os.path.join(docs_dir, "*.txt"))
        
        for path in txt_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = f.read().strip()
                corpus.append(raw)
                meta.append(os.path.basename(path))
            except Exception as e:
                st.sidebar.error(f"Error reading {path}: {e}")
    else:
        # Fallback: create sample documents inline
        st.sidebar.info("Using fallback sample documents...")
        sample_docs = {
            "Employment_Contract.txt": """Employment Contract – Analyst Position (2025)
1) Role: Business Analyst reporting to Integration Manager.
2) Compensation: Annual gross salary of EUR 65,000, payable monthly.
3) Working Hours: 40 hours per week, flexible schedule allowed.
4) Vacation: 25 working days per year, in addition to public holidays.
5) Termination: Both parties may terminate with 60 days' written notice.""",
            
            "HR_policy.txt": """Human Resources Policy Manual
1) Equal Opportunity: We provide equal employment opportunities regardless of race, gender, age, religion, or disability.
2) Work-Life Balance: Flexible working hours and remote work options available.
3) Professional Development: Annual training budget of €2,000 per employee.
4) Code of Conduct: Professional behavior expected at all times.
5) Grievance Procedure: Formal process for addressing workplace concerns.""",
            
            "Data_Privacy_Policy.txt": """Data Privacy and Protection Policy
1) Data Collection: We collect only necessary personal information for business purposes.
2) Data Storage: Personal data is stored securely with encryption.
3) Data Sharing: We do not sell personal data to third parties.
4) User Rights: Users can request access, correction, or deletion of their data.
5) Compliance: We comply with GDPR and other applicable privacy regulations."""
        }
        
        for filename, content in sample_docs.items():
            corpus.append(content)
            meta.append(filename)

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
