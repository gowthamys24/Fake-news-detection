import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# --------------------------
# 1. Load & Preprocess Data
# --------------------------
@st.cache_data
def load_data():
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")

    fake["label"] = 0
    true["label"] = 1

    data = pd.concat([fake, true], axis=0).reset_index(drop=True)

    # drop empty texts
    data = data[data["text"].notnull() & data["text"].str.strip().ne("")]

    # lowercase + remove punctuation
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = "".join([ch for ch in text if ch not in string.punctuation])
        return text

    data["text"] = data["text"].apply(clean_text)
    return data

# --------------------------
# 2. Train Model
# --------------------------
@st.cache_resource
def train_model():
    data = load_data()
    X = data["text"]
    y = data["label"]

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    # ANN
    model = Sequential([
        Dense(128, activation="relu", input_dim=X_train.shape[1]),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(X_train.toarray(), y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=0)

    # Test accuracy
    y_pred = (model.predict(X_test.toarray()) > 0.5).astype("int32")
    acc = accuracy_score(y_test, y_pred)

    return model, vectorizer, acc

# --------------------------
# 3. Streamlit UI
# --------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection with ANN")
st.write("Enter any news text below and find out if it's **Fake** or **Real**.")

# Train model (cached)
with st.spinner("Training model... please wait (~1 min)"):
    model, vectorizer, acc = train_model()

st.success(f"✅ Model trained successfully (Test Accuracy: {acc:.2f})")

# User Input
user_input = st.text_area("✍️ Paste a news article or headline:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned = user_input.lower()
        cleaned = "".join([c for c in cleaned if c not in string.punctuation])
        vec = vectorizer.transform([cleaned])

        pred = model.predict(vec.toarray())[0][0]
        if pred > 0.5:
            st.success("✅ This news is **REAL**")
        else:
            st.error("❌ This news is **FAKE**")
