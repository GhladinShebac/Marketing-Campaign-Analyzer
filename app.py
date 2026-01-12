import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

# --- 1. SETTINGS & DATA LOADING ---
st.set_page_config(page_title="Marketing Effectiveness Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('marketing_and_product_performance.csv')
    df['Conversion_Rate'] = (df['Conversions'] / df['Clicks']) * 100
    df['ROI'] = df['Units_Sold'] / df['Budget']
    return df

df = load_data()

# --- 2. SIDEBAR FILTERS ---
st.sidebar.title("Dashboard Filters")
channel = st.sidebar.selectbox("Select Marketing Channel", df['Campaign_ID'].unique())
budget_filter = st.sidebar.slider("Minimum Budget ($)", 0, int(df['Budget'].max()), 500)

filtered_df = df[(df['Campaign_ID'] == channel) & (df['Budget'] >= budget_filter)]

# --- 3. KEY PERFORMANCE INDICATORS (KPIs) ---
st.title("🚀 Marketing Campaign Effectiveness Analyzer")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Conversions", f"{df['Conversions'].sum():,}")
with col2:
    st.metric("Average ROI", f"{df['ROI'].mean():.2f}x")
with col3:
    st.metric("Avg Conversion Rate", f"{df['Conversion_Rate'].mean():.2f}%")

st.divider()

# --- 4. A/B TESTING & ANALYSIS ---
st.subheader("🎯 A/B Testing: Conversion Rate by Segment")
# Comparing how different tiers or channels perform
fig = px.box(df, x='Campaign_ID', y='Conversion_Rate', color='Campaign_ID',
             title="Distribution of Conversion Rates per Channel")
st.plotly_chart(fig, use_container_width=True)

# --- 5. MACHINE LEARNING PREDICTION ---
st.subheader("🤖 ML: Predict Campaign Success")
st.write("Enter campaign details below to predict if it will be high-performing.")

# Simple User Input for ML
user_budget = st.number_input("Planned Budget", value=1000)
user_clicks = st.number_input("Estimated Clicks", value=5000)

# Training a quick model (In production, you'd load a pre-trained model)
X = df[['Budget', 'Clicks', 'Units_Sold']]
y = (df['Conversion_Rate'] > df['Conversion_Rate'].median()).astype(int)
model = RandomForestClassifier().fit(X, y)

prediction = model.predict([[user_budget, user_clicks, 100]]) # 100 is dummy Units_Sold

if st.button("Predict Performance"):
    if prediction[0] == 1:
        st.success("✅ Prediction: High Impact Campaign!")
    else:
        st.error("⚠️ Prediction: Low Impact. Re-evaluate Budget.")



# --- 7. ATTRIBUTION MODELING ---
st.divider()
st.subheader("🔗 Marketing Attribution Modeling")

st.write("""
In a multi-channel journey, attribution helps us decide which channel gets credit for the sale.
Below, we distribute total conversions across different 'touchpoint' weights.
""")

# Creating a simple attribution simulation
attr_data = {
    'Channel': ['Social Media', 'Email', 'Search Ads', 'Referral'],
    'Last-Click Credit': [40, 20, 30, 10],
    'Linear Credit (Equal)': [25, 25, 25, 25]
}
attr_df = pd.DataFrame(attr_data)

st.table(attr_df)
st.info("💡 By shifting to a Linear Model, we can see if Social Media is undervalued as a first-touch channel.")