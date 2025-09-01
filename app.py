import streamlit as st
import pickle
import tensorflow as tf


st.title("✅ App Loaded Successfully")

# --------------------------
# Load Model & Vectorizer
# --------------------------
model = tf.keras.models.load_model("ann_model.h5")
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection App (with ANN)")
st.write("Enter a news article below and the model will predict whether it is **Fake** or **True**.")

# User input
user_input = st.text_area("Paste the news article here:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # Transform input using TF-IDF
        input_vec = vectorizer.transform([user_input]).toarray()
        
        # ANN Prediction (probability)
        prediction = model.predict(input_vec)[0][0]
        label = "✅ True News" if prediction >= 0.5 else "❌ Fake News"
        
        st.subheader("Prediction Result:")
        st.success(label if prediction >= 0.5 else label)

st.markdown("---")
st.caption("Built with Streamlit, TensorFlow & Scikit-learn")
