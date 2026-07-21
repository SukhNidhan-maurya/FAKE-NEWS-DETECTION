import re
import string

import joblib
import streamlit as st
import nltk
from nltk.corpus import stopwords

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

@st.cache_resource
def load_artifacts():
    nltk.download("stopwords", quiet=True)
    model = joblib.load("model/fake_news_model.pkl")
    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
    return model, vectorizer

model, tfidf = load_artifacts()
stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\w*\d\w*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

st.title("📰 Fake News Detector")
st.write("Paste a news headline or article below and the model will predict whether it looks Real or Fake.")

user_input = st.text_area("News text", height=220, placeholder="Paste the article or headline here...")

col1, col2 = st.columns([1, 3])
with col1:
    predict_clicked = st.button("Analyze", type="primary")

if predict_clicked:
    if not user_input.strip():
        st.warning("Please paste some text first.")
    else:
        cleaned = clean_text(user_input)
        vec = tfidf.transform([cleaned])
        pred = model.predict(vec)[0]

        if pred == 1:
            st.success("🟢 This looks like Real news.")
        else:
            st.error("🔴 This looks like Fake news.")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vec)[0]
            st.caption(f"Confidence — Fake: {proba[0]*100:.1f}% | Real: {proba[1]*100:.1f}%")

st.markdown("---")
st.caption("This tool is a machine-learning demo trained on a static dataset. It is not a substitute for fact-checking.")