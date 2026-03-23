import streamlit as st
import pandas as pd
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# --- DATA GENERATION ---
subjects = ["you", "ndi a", "that person", "those people"]
hate_words = ["stupid", "useless", "foolish", "wicked"]
non_hate_words = ["good", "kind", "nice", "amazing"]
verbs = ["are", "is"]
extras = ["very", "so", "really"]

data = []

for _ in range(2500):
    sentence = f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(extras)} {random.choice(hate_words)}"
    data.append((sentence, "hate"))

for _ in range(2500):
    sentence = f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(extras)} {random.choice(non_hate_words)}"
    data.append((sentence, "not_hate"))

df = pd.DataFrame(data, columns=["text", "label"])

# --- CLEAN ---
df["clean"] = df["text"].str.lower()

# --- VECTORIZE ---
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean"])
y = df["label"]

# --- TRAIN MODEL ---
svm = SVC()
svm.fit(X, y)

# --- GUI ---
st.title("Hate Speech Detection System")

text = st.text_input("Enter text")

if st.button("Check"):
    if text == "":
        st.write("Please enter text")
    else:
        vec = vectorizer.transform([text.lower()])
        result = svm.predict(vec)
        st.write("Prediction:", result[0])
