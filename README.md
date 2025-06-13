# Credit-Card-Fraud-Detector

This project aims to build a robust model to detect fraudulent transactions using anonymized credit card data. We use data preprocessing, exploratory data analysis, and a machine learning classifier to predict fraudulent behavior.

---

## 📁 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records**: 284,807 transactions
- **Features**:
  - `Time`: Seconds elapsed between each transaction and the first transaction.
  - `V1` to `V28`: **PCA-transformed** features to **anonymize sensitive information**.
  - `Amount`: Transaction amount.
  - `Class`: Target variable — `0` for **Legit**, `1` for **Fraud**.

---

## 📊 What Are V1–V28?

> These are **principal components** from PCA (Principal Component Analysis).  
> They're transformed versions of original confidential features like merchant name, location, etc.  
> Their actual meanings are unknown but they capture essential variance for detecting fraud.

---

## 🔧 Tools & Technologies

- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn
- XGBoost
- Streamlit

---

## 📈 Exploratory Data Analysis

### ✅ Class Distribution

![image](https://github.com/user-attachments/assets/099a3e3f-db27-44ab-95c4-e7ab0983108d)


### ✅ Correlation Heatmap

![image](https://github.com/user-attachments/assets/81756c6f-ee07-47cd-82e3-86e2f47bb51c)


### ✅ Boxplot (Transaction Amount by Class)

![image](https://github.com/user-attachments/assets/1ba62ba4-9c1d-4808-9de5-c812d9db53dc)


### ✅ Violin Plots (V1–V28 by Class)
📌 Multiple violin plots were created to visualize how each PCA component varies by class.  

![image](https://github.com/user-attachments/assets/f1b0a6dd-6cc8-410d-a56b-0e0e28f3ff71)
![image](https://github.com/user-attachments/assets/b3d4411e-2a28-44b5-bbca-bd98d80622e8)
![image](https://github.com/user-attachments/assets/dfd136a3-3ad6-4e0a-b89b-c24adac2a101)
![image](https://github.com/user-attachments/assets/ae2e839e-a180-4a5b-a823-37f80414a456)
![image](https://github.com/user-attachments/assets/0a6a030f-07b2-40fd-8bd6-b0522eae1603)



---

## ⚙️ Model

We used an **XGBoost Classifier** for its performance on imbalanced datasets.

### ✅ Evaluation Metrics

Accuracy  : 0.96
Precision : 0.97
Recall    : 0.96
F1 Score  : 0.96

📋 Classification Report:

Label	Precision	Recall	F1-score	Support
Legit	0.96	0.97	0.96	99
Fraud	0.97	0.96	0.96	98

💾 Saved Artifacts

File	Description
credit_fraud_model.pkl	Trained XGBoost model
amount_scaler.pkl	Scaler used for Amount
fraud_detector_app.py	Streamlit frontend code```

🌐 Streamlit UI
▶️ How to Run
bash
Copy
Edit
streamlit run "D:/ML PROJECTS/Credit Card Fraud Detection/fraud_detector_app.py"
🧪 Sample Input for UI (JSON)
json
Copy
Edit
{
  "V1": -1.359807134,
  "V2": -0.072781173,
  ...
  "V28": -0.021053053,
  "Amount": 149.62
}
📤 Sample CSV Input (for bulk predictions)
python-repl
Copy
Edit
V1,V2,V3,...,V28,Amount
-1.35,...,-0.02,149.62
...
✅ Sample UI Output
This transaction is likely LEGITIMATE ✅
(or)
⚠️ This transaction is FRAUDULENT


![Screenshot 2025-06-13 110801](https://github.com/user-attachments/assets/d2e75a1f-c305-4778-b869-be8d139d40c5)
![image](https://github.com/user-attachments/assets/7e625493-226a-414a-a8b1-1977c61d3ca4)
![image](https://github.com/user-attachments/assets/cdf61432-652f-446f-8ce8-ea7cb814bbd4)
![image](https://github.com/user-attachments/assets/0c5f640a-115d-40e8-95ec-195eab8887a9)


🚀 Future Improvements
Add SHAP/LIME explainability

Use ensemble techniques or autoencoders

Integrate with real-time APIs or dashboards

Auto-alert system via email/SMS

📚 License
MIT License © 2025 V Yuvan Krishnan
