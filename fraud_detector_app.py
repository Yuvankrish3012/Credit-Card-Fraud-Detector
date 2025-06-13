import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load trained model and scaler
model = pickle.load(open("D:/ML PROJECTS/Credit Card Fraud Detection/credit_fraud_model.pkl", "rb"))
scaler = pickle.load(open("D:/ML PROJECTS/Credit Card Fraud Detection/amount_scaler.pkl", "rb"))

# App configuration
st.set_page_config(page_title="💳 Credit Card Fraud Detection", layout="centered")

# Add a logo image from URL (top left)
st.markdown(
    """
    <div style='display: flex; align-items: center; gap: 10px;'>
        <img src='https://cdn-icons-png.flaticon.com/512/4228/4228703.png' width='60'>
        <h1 style='display:inline;'>Credit Card Fraud Detector</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("🔍 **Detect if a credit card transaction is fraudulent or legitimate**")

st.sidebar.header("🔁 Choose Mode")
mode = st.sidebar.radio("Select Input Mode:", ["🧍 Single Prediction", "📁 Bulk CSV Upload"])

# V1-V28 feature input
features = []
if mode == "🧍 Single Prediction":
    st.markdown("### 🧮 Enter Transaction Features")

    for i in range(1, 29):
        features.append(st.number_input(f"V{i}", value=0.0, format="%.4f"))

    amount = st.number_input("💰 Transaction Amount (₹)", value=100.0)
    scaled_amount = scaler.transform(np.array([[amount]]))[0][0]
    features.append(scaled_amount)

    if st.button("🚀 Predict Fraud"):
        input_data = np.array(features).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

        # Result
        if prediction == 1:
            st.error("⚠️ This transaction is likely **FRAUDULENT**")
        else:
            st.success("✅ This transaction is likely **LEGITIMATE**")

        # Pie chart
        st.markdown("### 🎯 Prediction Confidence")
        fig, ax = plt.subplots()
        ax.pie(proba, labels=["Legit", "Fraud"], autopct="%1.1f%%", startangle=90, colors=["green", "red"])
        ax.axis("equal")
        st.pyplot(fig)

elif mode == "📁 Bulk CSV Upload":
    st.markdown("### 📄 Upload a CSV file with V1–V28 and Amount columns")

    csv_file = st.file_uploader("Upload CSV", type=["csv"])
    if csv_file is not None:
        df = pd.read_csv(csv_file)

        if "Amount" in df.columns:
            df["Amount"] = scaler.transform(df[["Amount"]])
        else:
            st.error("CSV must include an 'Amount' column!")
            st.stop()

        input_data = df.values
        preds = model.predict(input_data)

        df["Prediction"] = preds
        df["Prediction"] = df["Prediction"].map({0: "Legit", 1: "Fraud"})

        st.dataframe(df)
        fraud_count = df["Prediction"].value_counts()
        st.markdown("### 📊 Prediction Distribution")

        fig, ax = plt.subplots()
        ax.bar(fraud_count.index, fraud_count.values, color=["green", "red"])
        st.pyplot(fig)
