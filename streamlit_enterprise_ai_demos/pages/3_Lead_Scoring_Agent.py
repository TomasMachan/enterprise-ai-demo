import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

st.title("🤝 AI Lead Scoring Agent (Explainable)")
st.caption("Upload a leads CSV or use the sample. Get scores plus human-readable reasons.")

@st.cache_data
def load_sample():
    # Try multiple possible paths for the leads.csv file
    possible_paths = [
        "leads.csv",  # Original relative path
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "leads.csv"),  # From pages/ to root
        os.path.join(os.getcwd(), "leads.csv"),  # From current working directory
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "leads.csv"),  # Alternative relative path
    ]
    
    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            break
    
    if csv_path:
        return pd.read_csv(csv_path)
    else:
        # Fallback: create sample data inline
        st.info("Using fallback sample data...")
        return pd.DataFrame({
            "Name": ["Petr Novak", "Alex Smith", "Jana Horakova", "Marco Rossi", "Eva Kral"],
            "CompanySize": ["Enterprise", "Mid-Market", "SMB", "Enterprise", "SMB"],
            "Industry": ["Finance", "SaaS", "Retail", "Manufacturing", "Finance"],
            "EngagementLevel": [5, 3, 4, 2, 5],
            "PagesViewed": [12, 6, 8, 3, 15],
            "RequestedPricing": [1, 0, 1, 0, 1],
            "LeadSource": ["Webinar", "Website", "Referral", "Outbound", "Conference"],
            "Region": ["EU", "US", "EU", "EMEA", "EU"]
        })

use_sample = st.checkbox("Use sample leads.csv", value=True)
if use_sample:
    df = load_sample()
else:
    up = st.file_uploader("Upload leads.csv", type=["csv"])
    if up:
        df = pd.read_csv(up)
    else:
        st.info("Upload a CSV or use the sample to continue.")
        st.stop()

st.dataframe(df)

num_cols = ["EngagementLevel","PagesViewed","RequestedPricing"]
cat_cols = ["CompanySize","Industry","LeadSource","Region"]

pre = ColumnTransformer([
    ("num","passthrough", num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

clf = Pipeline([
    ("pre", pre),
    ("lr", LogisticRegression(max_iter=1000))
])

if "did_convert" not in df.columns:
    rng = np.random.default_rng(42)
    def heuristic(row):
        s = 0
        s += row.get("EngagementLevel",0) * 0.8
        s += min(row.get("PagesViewed",0), 15) * 0.2
        s += 2.0 if row.get("RequestedPricing",0) == 1 else 0
        s += 1.5 if row.get("CompanySize","") in ("Enterprise","Mid-Market") else 0.5
        s += 1.0 if row.get("LeadSource","") in ("Referral","Conference","Webinar") else 0
        return s
    base = df.apply(heuristic, axis=1).values
    prob = 1/(1+np.exp(-(base - base.mean())/2.0))
    did = (prob > rng.random(len(prob))).astype(int)
    df["did_convert"] = did

X = df.drop(columns=[c for c in ["did_convert","Name"] if c in df.columns])
y = df["did_convert"]
clf.fit(X, y)

st.subheader("Lead Scores")
def score_row(row):
    try:
        # Convert the Series to a DataFrame with proper column names
        row_df = pd.DataFrame([row.values], columns=row.index)
        proba = clf.predict_proba(row_df)[0,1]
        reasons = []
        if "RequestedPricing" in row and row["RequestedPricing"] == 1:
            reasons.append("Requested pricing → high intent")
        if "EngagementLevel" in row and row["EngagementLevel"] >= 4:
            reasons.append("High engagement level")
        if "PagesViewed" in row and row["PagesViewed"] >= 10:
            reasons.append("Multiple page views")
        if "LeadSource" in row and row["LeadSource"] in ("Referral","Conference","Webinar"):
            reasons.append("Strong source: " + str(row["LeadSource"]))
        if "CompanySize" in row and row["CompanySize"] in ("Enterprise","Mid-Market"):
            reasons.append("Company size: " + str(row["CompanySize"]))
        return float(proba), "; ".join(reasons)
    except Exception as e:
        st.error(f"Error scoring lead: {e}")
        return 0.0, "Error in scoring"

scored = []
for i, r in X.iterrows():
    p, why = score_row(r)
    name = df.loc[i, "Name"] if "Name" in df.columns else f"Lead {i+1}"
    scored.append({"Name": name, "Score": round(p,3), "Reasons": why})

out = pd.DataFrame(scored).sort_values("Score", ascending=False)
st.dataframe(out)

st.subheader("Add a Custom Lead")
with st.form("add_lead"):
    company_size = st.selectbox("CompanySize", ["SMB","Mid-Market","Enterprise"], index=1)
    industry = st.text_input("Industry", "Finance")
    engagement = st.slider("EngagementLevel", 1, 5, 4)
    pages = st.number_input("PagesViewed", 0, 100, 9)
    requested = st.selectbox("RequestedPricing (0/1)", [0,1], index=1)
    source = st.selectbox("LeadSource", ["Website","Webinar","Outbound","Referral","Conference"], index=0)
    region = st.text_input("Region", "EU")
    submitted = st.form_submit_button("Score lead")
    if submitted:
        new = pd.DataFrame([{
            "CompanySize": company_size,
            "Industry": industry,
            "EngagementLevel": engagement,
            "PagesViewed": pages,
            "RequestedPricing": requested,
            "LeadSource": source,
            "Region": region
        }])
        try:
            proba = float(clf.predict_proba(new)[0,1])
            st.success(f"Predicted conversion score: {proba:.3f}")
        except Exception as e:
            st.error(f"Error predicting score: {e}")
            proba = 0.0
        st.write("Reasons:")
        if requested == 1: st.write("- Requested pricing → high intent")
        if engagement >= 4: st.write("- High engagement level")
        if pages >= 10: st.write("- Multiple page views")
        if source in ("Referral","Conference","Webinar"): st.write(f"- Strong source: {source}")
        if company_size in ("Enterprise","Mid-Market"): st.write(f"- Company size: {company_size}")
