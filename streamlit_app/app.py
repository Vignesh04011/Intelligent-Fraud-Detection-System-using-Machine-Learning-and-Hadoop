import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest

# ------------------ Streamlit Config ------------------
st.set_page_config(page_title="FraudDetectX", layout="wide")

st.markdown("## 🔎 FraudDetectX – Advanced Subsidy Fraud Detection System")
st.markdown("Detects fake or suspicious subsidy claims using rule-based logic, statistical analysis, and ML.")

# ------------------ Upload File ------------------
uploaded_file = st.file_uploader("📁 Upload Subsidy Claims CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Preview Uploaded Data")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown(f"✅ **{len(df)} total records** uploaded.")

    # ------------------ Combine Fraud Flags ------------------
    def combine_flags(row):
        flags = []
        if 'AadhaarFlag' in df.columns and row['AadhaarFlag']:
            flags.append("Duplicate Aadhaar")
        if 'ClaimFlag' in df.columns and row['ClaimFlag']:
            flags.append("Duplicate Claim")
        if 'FraudFlag' in df.columns and row['FraudFlag']:
            flags.append("Other Fraud")
        return " | ".join(flags) if flags else None

    df['FraudType'] = df.apply(combine_flags, axis=1)

    # ------------------ ML Anomaly Detection ------------------
    try:
        if 'ClaimAmount' in df.columns:
            model = IsolationForest(contamination=0.05, random_state=42)
            df['ML_Anomaly'] = model.fit_predict(df[['ClaimAmount']])
            df.loc[df['ML_Anomaly'] == -1, 'FraudType'] = df['FraudType'].fillna('') + " | ML Anomaly"
            df['FraudType'] = df['FraudType'].str.strip(" |")
    except Exception as e:
        st.warning(f"⚠️ ML detection skipped: {e}")

    # ------------------ Keep Only Fraud Cases ------------------
    flagged = df[df['FraudType'].notnull()].copy()

    if flagged.empty:
        st.success("✅ No suspicious claims detected.")
    else:
        st.error(f"⚠️ {len(flagged)} suspicious records found.")

        # Fraud summary by type
        st.subheader("📊 Fraud Cases by Type")
        fraud_summary = flagged['FraudType'].value_counts().reset_index()
        fraud_summary.columns = ['FraudType', 'Count']
        fig = px.pie(fraud_summary, names='FraudType', values='Count', title='Fraud Type Distribution')
        st.plotly_chart(fig, use_container_width=True)

        # Fraud by person
        st.subheader("👤 Fraud by Beneficiary")
        fraud_by_person = flagged.groupby('Name')['FraudType'].apply(lambda x: ', '.join(set(x))).reset_index()
        st.dataframe(fraud_by_person, use_container_width=True)

        # Search functionality
        st.subheader("🔎 Search Fraud Records")
        search_query = st.text_input("Enter Name or Aadhaar to search")
        if search_query:
            results = flagged[
                flagged['Name'].str.contains(search_query, case=False, na=False) |
                flagged['Aadhaar'].astype(str).str.contains(search_query, na=False)
            ]
            if not results.empty:
                st.success(f"Found {len(results)} matching records")
                st.dataframe(results, use_container_width=True)
            else:
                st.warning("No matching records found.")

        # Detailed fraud cases
        st.subheader("📄 All Flagged Fraud Cases")
        st.dataframe(flagged[['Name', 'Aadhaar', 'ClaimAmount', 'SubsidyType', 'FraudType']], use_container_width=True)

        # Download
        csv = flagged.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Flagged Records", csv, "frauds_detected.csv", "text/csv", use_container_width=True)

else:
    st.info("⬆️ Upload a valid CSV with columns: `Name`, `Aadhaar`, `ClaimAmount`, `SubsidyType`, `Date`, and fraud flags.")
