import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
import plotly.express as px

# ------------------ Streamlit Config ------------------
st.set_page_config(page_title="FraudDetectX", layout="wide")

# ------------------ Custom CSS for modern fonts ------------------
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        font-size: 16px;
    }
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ Title ------------------
st.markdown("## 🔎 FraudDetectX – Advanced Subsidy Fraud Detection System")
st.markdown("Detects fake or suspicious subsidy claims using rule-based logic, statistical analysis, and ML.")

# ------------------ Helper Functions ------------------

def detect_duplicate_aadhaar(df):
    result = df[df.duplicated(['Aadhaar'], keep=False)].copy()
    result['FraudType'] = "Duplicate Aadhaar"
    return result

def detect_high_claims(df, threshold=10000):
    result = df[df['ClaimAmount'] > threshold].copy()
    result['FraudType'] = f"High Claim Amount > ₹{threshold}"
    return result

def detect_temporal_patterns(df, days_threshold=10):
    result = df.sort_values(by=['Aadhaar', 'Date']).groupby('Aadhaar').filter(
        lambda x: (x['Date'].max() - x['Date'].min()).days < days_threshold and len(x) > 1
    ).copy()
    result['FraudType'] = f"Rapid Resubmission (<{days_threshold} days)"
    return result

def detect_name_aadhaar_mismatch(df):
    mismatch_ids = df.groupby('Aadhaar')['Name'].nunique()
    mismatch_ids = mismatch_ids[mismatch_ids > 1].index
    result = df[df['Aadhaar'].isin(mismatch_ids)].copy()
    result['FraudType'] = "Multiple Names for Same Aadhaar"
    return result

def detect_iqr_outliers(df):
    results = []
    for subsidy_type, group in df.groupby('SubsidyType'):
        q1 = group['ClaimAmount'].quantile(0.25)
        q3 = group['ClaimAmount'].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        outliers = group[group['ClaimAmount'] > upper_bound].copy()
        outliers['FraudType'] = f"Outlier in {subsidy_type} (IQR)"
        results.append(outliers)
    return pd.concat(results) if results else pd.DataFrame()

def detect_ml_anomalies(df):
    try:
        ml_input = df[['ClaimAmount']].copy()
        model = IsolationForest(contamination=0.05, random_state=42)
        df['anomaly'] = model.fit_predict(ml_input)
        result = df[df['anomaly'] == -1].copy()
        result['FraudType'] = "ML-Based Anomaly (Isolation Forest)"
        return result
    except Exception as e:
        st.warning(f"⚠️ ML detection failed: {e}")
        return pd.DataFrame()

# ------------------ Main App ------------------

uploaded_file = st.file_uploader("📁 Upload Subsidy Claims CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    with st.expander("📄 Preview Uploaded Data", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown(f"✅ **{len(df)} total records** uploaded.")

    method = st.selectbox("🧠 Choose Detection Method", [
        "All Methods",
        "Duplicate Aadhaar Detection",
        "High Claim Amount Detection",
        "Temporal Pattern Detection",
        "Name-Aadhaar Mismatch",
        "Outlier Detection (IQR Method)",
        "ML-Based Detection (Isolation Forest)"
    ])

    if st.button("🚀 Run Fraud Detection", type="primary", use_container_width=True):
        st.markdown("### 🔍 Detection Results")
        all_results = pd.DataFrame()

        if method in ["All Methods", "Duplicate Aadhaar Detection"]:
            all_results = pd.concat([all_results, detect_duplicate_aadhaar(df)])

        if method in ["All Methods", "High Claim Amount Detection"]:
            all_results = pd.concat([all_results, detect_high_claims(df)])

        if method in ["All Methods", "Temporal Pattern Detection"]:
            all_results = pd.concat([all_results, detect_temporal_patterns(df)])

        if method in ["All Methods", "Name-Aadhaar Mismatch"]:
            all_results = pd.concat([all_results, detect_name_aadhaar_mismatch(df)])

        if method in ["All Methods", "Outlier Detection (IQR Method)"]:
            all_results = pd.concat([all_results, detect_iqr_outliers(df)])

        if method in ["All Methods", "ML-Based Detection (Isolation Forest)"]:
            all_results = pd.concat([all_results, detect_ml_anomalies(df)])

        all_results.drop_duplicates(inplace=True)

        if all_results.empty:
            st.success("✅ No suspicious claims detected.")
        else:
            st.error(f"⚠️ {len(all_results)} suspicious records found.")

            with st.expander("📄 View Detected Records", expanded=True):
                st.dataframe(all_results, use_container_width=True)

            # Fraud Summary Chart
            fraud_summary = all_results['FraudType'].value_counts().reset_index()
            fraud_summary.columns = ['FraudType', 'Count']
            fig = px.pie(fraud_summary, names='FraudType', values='Count', title='Fraud Type Distribution')
            fig.update_traces(textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

            # Metric Summary
            col1, col2, col3 = st.columns(3)
            col1.metric("🔁 Duplicate Aadhaar", fraud_summary[fraud_summary['FraudType'] == "Duplicate Aadhaar"]['Count'].sum() if "Duplicate Aadhaar" in fraud_summary['FraudType'].values else 0)
            col2.metric("💰 High Claims", fraud_summary[fraud_summary['FraudType'].str.contains("High Claim", na=False)]['Count'].sum())
            col3.metric("🤖 ML Anomalies", fraud_summary[fraud_summary['FraudType'].str.contains("ML", na=False)]['Count'].sum())

            # Download
            csv = all_results.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Flagged Records", csv, "frauds_detected.csv", "text/csv", use_container_width=True)

else:
    st.info("⬆️ Upload a valid CSV with these columns: `Name`, `Aadhaar`, `ClaimAmount`, `SubsidyType`, `Date`.")
