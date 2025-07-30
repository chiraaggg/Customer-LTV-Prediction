import pandas as pd
import numpy as np
from lifetimes import BetaGeoFitter, GammaGammaFitter
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="📈 LTV Prediction Dashboard", layout="wide")
st.title("📈 Customer Lifetime Value (LTV) Prediction")

st.markdown("""
Upload orders.csv with the following columns:  
user_id, first_name, phone, created_at, total
""")

# ------------------------------
# File Upload
# ------------------------------
uploaded_file = st.file_uploader("Upload Orders CSV", type=["csv"])
if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)
df['created_at'] = pd.to_datetime(df['created_at'])
df['cohort_month'] = df['created_at'].dt.to_period("M").astype(str)

# ------------------------------
# Prepare Data
# ------------------------------
snapshot_date = df['created_at'].max() + pd.Timedelta(days=1)

ltv_data = df.groupby(['user_id', 'first_name', 'phone']).agg({
    'created_at': [lambda x: (x.max() - x.min()).days,
                   lambda x: (snapshot_date - x.min()).days,
                   'count', 'min'],
    'total': 'mean'
}).reset_index()

ltv_data.columns = ['user_id', 'FIRSTNAME', 'MOBILE', 'recency', 'T', 'frequency', 'created_at', 'monetary_value']
ltv_data['recency'] /= 7
ltv_data['T'] /= 7
ltv_data = ltv_data[ltv_data['frequency'] > 1]

# ------------------------------
# BG/NBD + Gamma-Gamma Model
# ------------------------------
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(ltv_data['frequency'], ltv_data['recency'], ltv_data['T'])

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(ltv_data['frequency'], ltv_data['monetary_value'])

# Prediction Horizon
st.sidebar.header("📅 Prediction Settings")
days = st.sidebar.slider("Prediction Horizon (days)", 30, 365, 90, 30)
weeks = days / 7

ltv_data['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
    weeks, ltv_data['frequency'], ltv_data['recency'], ltv_data['T']
)
ltv_data['predicted_monetary'] = ggf.conditional_expected_average_profit(
    ltv_data['frequency'], ltv_data['monetary_value']
)
ltv_data['predicted_ltv'] = ltv_data['predicted_purchases'] * ltv_data['predicted_monetary']

# ------------------------------
# Advanced Model (Random Forest for boost)
# ------------------------------
X = ltv_data[['recency', 'T', 'frequency', 'monetary_value']]
y = ltv_data['predicted_ltv']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
ltv_data['predicted_ltv'] = model.predict(X)

# ------------------------------
# Segmentation
# ------------------------------
high_threshold = ltv_data['predicted_ltv'].quantile(0.66)
low_threshold = ltv_data['predicted_ltv'].quantile(0.33)

def segment_ltv(value):
    if value >= high_threshold:
        return 'High'
    elif value >= low_threshold:
        return 'Medium'
    else:
        return 'Low'

ltv_data['LTV_segment'] = ltv_data['predicted_ltv'].apply(segment_ltv)

# ------------------------------
# Metrics and Charts
# ------------------------------
st.subheader(f"📊 LTV Insights (Next {days} Days)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Users", len(ltv_data))
col2.metric("Avg LTV", f"{ltv_data['predicted_ltv'].mean():.2f}")
col3.metric("Max LTV", f"{ltv_data['predicted_ltv'].max():.2f}")
col4.metric("Total Predicted Revenue", f"{ltv_data['predicted_ltv'].sum():,.2f}")

segment_counts = ltv_data['LTV_segment'].value_counts().reset_index()
segment_counts.columns = ['Segment', 'User Count']
fig1 = px.pie(segment_counts, names='Segment', values='User Count', hole=0.5)
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.bar(segment_counts, x='User Count', y='Segment', orientation='h', text='User Count')
st.plotly_chart(fig2, use_container_width=True)

# Top High LTV Users
st.subheader("💎 Top High LTV Users")
top_users = ltv_data[ltv_data['LTV_segment'] == 'High'].sort_values(by='predicted_ltv', ascending=False)
fig3 = px.bar(top_users.head(10), x='predicted_ltv', y='FIRSTNAME', orientation='h', text='MOBILE')
st.plotly_chart(fig3, use_container_width=True)

# Trend Over Time
st.subheader("📈 LTV Trend Over Time")
ltv_trend = df[df['created_at'].dt.year == 2025].groupby(df['created_at'].dt.date)['total'].mean().reset_index()
ltv_trend.columns = ['Date', 'Avg LTV']
fig4 = px.line(ltv_trend, x='Date', y='Avg LTV', markers=True)
st.plotly_chart(fig4, use_container_width=True)

# Cohort Trend
st.subheader("📆 Cohort-Wise Weekly LTV Trend")
ltv_data['cohort_month'] = ltv_data['created_at'].dt.to_period("M").astype(str)
cohort_ltv_trend = ltv_data.groupby(['cohort_month', pd.Grouper(key='created_at', freq='W')])['predicted_ltv'].mean().reset_index()
fig_cohort = px.line(cohort_ltv_trend, x='created_at', y='predicted_ltv', color='cohort_month', markers=True)
st.plotly_chart(fig_cohort, use_container_width=True)

# Revenue Breakdown by Cohort
st.subheader("📊 Predicted Revenue by Cohort (Last 6 Months)")
recent_cohorts = ltv_data['cohort_month'].sort_values().unique()[-6:]
cohort_revenue = ltv_data[ltv_data['cohort_month'].isin(recent_cohorts)].groupby('cohort_month')['predicted_ltv'].sum().reset_index()
fig_rev = px.bar(cohort_revenue, x='cohort_month', y='predicted_ltv', text='predicted_ltv', labels={'predicted_ltv': 'Predicted Revenue'})
st.plotly_chart(fig_rev, use_container_width=True)



# Top 10% Contribution
st.subheader("🔝 Revenue Contribution by Top 10% Users")
ltv_data_sorted = ltv_data.sort_values(by='predicted_ltv', ascending=False)
top_10pct = int(0.10 * len(ltv_data_sorted))
top_users = ltv_data_sorted.head(top_10pct)
contribution_pct = (top_users['predicted_ltv'].sum() / ltv_data['predicted_ltv'].sum()) * 100
st.metric("Top 10% Revenue Contribution", f"{contribution_pct:.2f}%")

# Repeat Purchase Rate
st.subheader("🔁 Predicted Repeat Purchase Rate")
total_users = len(ltv_data)
repeat_users = (ltv_data['predicted_purchases'] >= 1).sum()
repeat_rate = (repeat_users / total_users) * 100
st.metric("Repeat Purchase Rate", f"{repeat_rate:.2f}%")

# Downloads
st.subheader("📥 Download Segmented Users")
export_df = ltv_data[['MOBILE', 'FIRSTNAME', 'LTV_segment']]

st.download_button("⬇️ Download All Users", export_df.to_csv(index=False), file_name=f"ltv_all_{days}d.csv")
st.download_button("⬇️ High LTV", export_df[export_df['LTV_segment'] == 'High'].to_csv(index=False), file_name=f"ltv_high_{days}d.csv")
st.download_button("⬇️ Medium LTV", export_df[export_df['LTV_segment'] == 'Medium'].to_csv(index=False), file_name=f"ltv_medium_{days}d.csv")
st.download_button("⬇️ Low LTV", export_df[export_df['LTV_segment'] == 'Low'].to_csv(index=False), file_name=f"ltv_low_{days}d.csv")
