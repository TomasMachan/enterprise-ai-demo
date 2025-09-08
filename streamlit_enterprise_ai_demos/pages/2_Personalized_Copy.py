import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import textwrap

st.title("✉️ Personalized Marketing Copy Generator")
st.caption("Generate persona-specific outreach emails with subject, body, and a single CTA.")

@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
    return tok, mdl

tokenizer, model = load_model()

product = st.text_input("Product/Service", "cloud backup solution")
persona = st.selectbox("Persona", ["CFO", "IT Manager", "Small Business Owner"])
tone = st.selectbox("Tone", ["professional", "friendly", "bold"])
cta = st.selectbox("CTA", ["Book a demo", "Start free trial", "Get pricing"])
variants = st.slider("Number of variants", 1, 3, 3)

def generate_copy(product, persona, tone, cta):
    prompt = (
        "You are a seasoned B2B marketer.\n"
        "Write an outreach email tailored to the persona below.\n"
        "Include: (1) a compelling subject line, (2) a crisp 120–180 word body,\n"
        "(3) a single clear CTA. Keep it " + tone + " and benefit-focused.\n\n"
        "Product: " + product + "\n"
        "Persona: " + persona + "\n"
        "CTA: " + cta
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=220, num_beams=4, do_sample=False)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

if st.button("Generate"):
    for i in range(variants):
        if i == 1:
            p = persona + " who is budget-conscious"
        elif i == 2:
            p = persona + " at a growing company"
        else:
            p = persona
        st.markdown(f"### Variant {i+1}")
        st.write(textwrap.fill(generate_copy(product, p, tone, cta), width=100))
        st.markdown("---")
