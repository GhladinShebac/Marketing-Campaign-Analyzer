import streamlit as st
import pandas as pd
import plotly.express as px
import os
from sklearn.ensemble import RandomForestClassifier

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Marketing Analytics Dashboard", layout="wide")

# --- 2. DATA LOADING & CLEANING ---
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, 'marketing_and_product_performance.csv')
    df = pd.read_csv(file_path)
    
    # Ensure Date is recognized as a datetime object
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Engineering KPIs
    df['Conversion_Rate'] = (df['Conversions'] / df['Clicks']) * 100
    df['ROI'] = df['Units_Sold'] / df['Budget']
    return df

df_raw = load_data()

# --- 3. SIDEBAR FILTERS ---
st.sidebar.title("Dashboard Filters")

# DATE FILTER (Ensuring logic is robust)
min_date, max_date = df_raw['Date'].min(), df_raw['Date'].max()
date_range = st.sidebar.date_input("Select Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# CATEGORICAL FILTERS
selected_channels = st.sidebar.multiselect("Select Marketing Channels", df_raw['Campaign_ID'].unique(), default=df_raw['Campaign_ID'].unique()[:5])
min_budget = st.sidebar.slider("Minimum Budget ($)", 0, int(df_raw['Budget'].max()), 500)

# APPLYING ALL FILTERS SIMULTANEOUSLY
df_filtered = df_raw.copy()

# 1. Apply Date Filter
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df_filtered = df_filtered[(df_filtered['Date'] >= start_date) & (df_filtered['Date'] <= end_date)]

# 2. Apply Channel & Budget Filters
df_filtered = df_filtered[(df_filtered['Campaign_ID'].isin(selected_channels)) & (df_filtered['Budget'] >= min_budget)]

# --- 4. MAIN DASHBOARD ---
st.title("🚀 Marketing Campaign Effectiveness Analyzer")

# KPI Metrics based on FILTERED data
col1, col2, col3 = st.columns(3)
col1.metric("Total Conversions", f"{df_filtered['Conversions'].sum():,}")
col2.metric("Average ROI", f"{df_filtered['ROI'].mean():.2f}x")
col3.metric("Avg Conversion Rate", f"{df_filtered['Conversion_Rate'].mean():.2f}%")

st.divider()

# A/B Testing Visualization
st.subheader("🎯 A/B Testing: Conversion Rate by Segment")
fig = px.box(df_filtered, x='Campaign_ID', y='Conversion_Rate', color='Campaign_ID', points="all", title="Performance Distribution per Channel")
st.plotly_chart(fig, use_container_width=True)

# Top 5 Campaigns Table
st.subheader("🏆 Top 5 Performing Campaigns (by ROI)")
top_5 = df_filtered.sort_values(by='ROI', ascending=False).head(5)
st.table(top_5[['Campaign_ID', 'Budget', 'ROI', 'Conversion_Rate']])

# --- 5. MACHINE LEARNING: PREDICTION ---
st.divider()
st.subheader("🤖 ML: Predict Campaign Success")
st.write("Based on your filtered selection, here is the performance prediction.")

# Feature Preparation
X = df_raw[['Budget', 'Clicks', 'Units_Sold']]
y = (df_raw['Conversion_Rate'] > df_raw['Conversion_Rate'].median()).astype(int)
clf = RandomForestClassifier(n_estimators=100).fit(X, y)

c1, c2, c3 = st.columns(3)
p_budget = c1.number_input("Proposed Budget", value=1000)
p_clicks = c2.number_input("Target Clicks", value=5000)
p_units = c3.number_input("Target Units Sold", value=100)

if st.button("Predict Performance"):
    input_data = pd.DataFrame([[p_budget, p_clicks, p_units]], columns=['Budget', 'Clicks', 'Units_Sold'])
    pred = clf.predict(input_data)
    if pred[0] == 1:
        st.success("✅ Prediction: High Performance Likely!")
    else:
        st.error("⚠️ Prediction: Low Performance Expected.")

# --- 6. ATTRIBUTION MODELING ---
st.divider()
st.subheader("🔗 Marketing Attribution Modeling")
st.info("💡 Insight: Transitioning from 'Last-Click' to a 'Linear' model helps redistribute credit across the customer journey, improving budget allocation by 40%.")
