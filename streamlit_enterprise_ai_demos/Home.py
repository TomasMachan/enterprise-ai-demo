import streamlit as st

st.set_page_config(page_title="Enterprise AI Demos", page_icon="🤖", layout="centered")

st.title("🤖 Enterprise AI Demos (2025)")
st.markdown("Welcome! This is a no-code set of demos your audience can try from a browser.")
st.markdown("Pages on the left:")
st.markdown("1. Semantic Search – Ask questions over business documents by meaning.")
st.markdown("2. Personalized Copy – Generate persona‑tailored outreach emails.")
st.markdown("3. Lead Scoring Agent – Score leads and see why some are hotter.")

st.subheader("How to run locally")
st.code("pip install -r requirements.txt\nstreamlit run Home.py", language="bash")

st.info("Tip for workshops: deploy on Streamlit Community Cloud or Hugging Face Spaces and share the URL.")
