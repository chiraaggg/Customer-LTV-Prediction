# Customer-LTV-Prediction
Interactive Streamlit dashboard for predicting Customer Lifetime Value (LTV), segmenting users with RFM + KMeans, and analyzing revenue patterns using BG/NBD and Gamma-Gamma models. Built to drive retention and growth decisions at Origin.

# 🧠 Customer LTV Prediction & RFM Segmentation Dashboard

An interactive dashboard built using **Streamlit** to forecast **Customer Lifetime Value (LTV)**, segment users by behavior (RFM and KMeans), and help teams make smarter, data-driven retention and marketing decisions.

This project was built to support user growth at **Origin**, a fast-growing green commerce platform.

---

## 🚀 Features

### 🔮 LTV Prediction
- Predicts each user's future 6-month revenue using:
  - **BG/NBD model** for frequency of repeat purchases
  - **Gamma-Gamma model** for average spend per purchase
- Predicts:
  - Total revenue by cohort month (next 6 months)
  - Repeat purchase probability
  - Top 10% users by future value

### 📊 RFM Segmentation
- Rule-based segmentation (Recency, Frequency, Monetary):
  - Champions, Loyal, At Risk, Lost, etc.
- ML-based segmentation using **KMeans clustering**
- Upload past segmentation runs to track user movement across segments

### 💡 Insights Dashboard
- Total predicted revenue card
- Segment breakdowns and exportable user lists
- Visual cohort trends and segment distribution
- Segment-level reorder analysis post-campaign

---

###  Upload your order data
Make sure your CSV includes at least these columns:

user_id

order_id

order_date (YYYY-MM-DD format)

order_value

### 📊 Sample Output
LTV forecast over next 6 months

Predicted repeat purchase rate

Top 10% contributing users

Interactive RFM segments and cohort views

### 🤖 Tech Stack
Python

Streamlit

Lifetimes (BG/NBD, Gamma-Gamma)

Scikit-learn (KMeans)

Pandas, NumPy, Matplotlib, Seaborn

### 🧠 Motivation
Growth isn't just about how many users you acquire.
It's about how well you understand and grow the value of the users you already have.

This dashboard was built to help teams:

Predict future revenue, not just analyze the past

Target campaigns by segment, not assumption

Build retention engines instead of churn patches

### 📬 Contact
Made with ❤️ by Chirag
📫 Reach out on LinkedIn if you’d like to collaborate or learn more! https://www.linkedin.com/in/chiiraagg/


