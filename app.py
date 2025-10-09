# streamlit_app/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import subprocess
import os

# ------------------ Streamlit Config ------------------
st.set_page_config(page_title="FraudDetectX", layout="wide")
st.markdown("## 🔎 FraudDetectX – Advanced Subsidy Fraud Detection System")
st.markdown("Detects fake or suspicious subsidy claims using rule-based logic, statistical analysis, and ML.")

# ------------------ Session State ------------------
if "pipeline_ran" not in st.session_state:
    st.session_state.pipeline_ran = False

# ------------------ Upload File ------------------
uploaded_file = st.file_uploader("📁 Upload Subsidy Claims CSV", type=["csv"])

if uploaded_file:
    raw_path = "data/raw/uploaded_subsidy_claims.csv"
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.subheader("📄 Preview Uploaded Data")
    df_preview = pd.read_csv(raw_path)
    st.dataframe(df_preview.head(10), use_container_width=True)
    st.markdown(f"✅ **{len(df_preview)} total records** uploaded.")

    # ------------------ Run Pipeline ------------------
    if st.button("🚀 Detect Fraud"):
        with st.spinner("Running full pipeline..."):
            try:
                subprocess.run(["python", "-m", "scripts.run_pipeline"], check=True)
                st.session_state.pipeline_ran = True
                st.success("✅ Pipeline completed!")
            except subprocess.CalledProcessError as e:
                st.error(f"❌ Pipeline failed: {e}")

# ------------------ Display Flagged Results ------------------
if st.session_state.pipeline_ran:
    flagged_path = "data/results/fraud_flagged.csv"
    if os.path.exists(flagged_path):
        df_flagged = pd.read_csv(flagged_path)

        if df_flagged.empty or df_flagged['FraudType'].isnull().all():
            st.success("✅ No suspicious claims detected.")
        else:
            st.error(f"⚠️ {len(df_flagged)} suspicious records found.")

            # Fraud summary by type
            st.subheader("📊 Fraud Cases by Type")
            fraud_summary = df_flagged['FraudType'].value_counts().reset_index()
            fraud_summary.columns = ['FraudType', 'Count']
            fig = px.pie(fraud_summary, names='FraudType', values='Count', title='Fraud Type Distribution')
            st.plotly_chart(fig, use_container_width=True)

            # Fraud by Beneficiary
            st.subheader("👤 Fraud by Beneficiary")
            fraud_by_person = df_flagged.groupby('Name')['FraudType'].apply(lambda x: ', '.join(set(x))).reset_index()
            st.dataframe(fraud_by_person, use_container_width=True)

            # Search functionality
            st.subheader("🔎 Search Fraud Records")
            search_query = st.text_input("Enter Name or Aadhaar to search")
            if search_query:
                results = df_flagged[
                    df_flagged['Name'].str.contains(search_query, case=False, na=False) |
                    df_flagged['Aadhaar'].astype(str).str.contains(search_query, na=False)
                ]
                if not results.empty:
                    st.success(f"Found {len(results)} matching records")
                    st.dataframe(results, use_container_width=True)
                else:
                    st.warning("No matching records found.")

            # Detailed flagged records
            st.subheader("📄 All Flagged Fraud Cases")
            cols_to_show = ['Name', 'Aadhaar', 'ClaimAmount', 'SubsidyType', 'FraudType']
            available_cols = [c for c in cols_to_show if c in df_flagged.columns]
            st.dataframe(df_flagged[available_cols], use_container_width=True)

            # Download
            csv = df_flagged.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Flagged Records", csv, "frauds_detected.csv", "text/csv", use_container_width=True)

else:
    st.info("⬆️ Upload a valid CSV with columns: `Name`, `Aadhaar`, `ClaimAmount`, `SubsidyType`, `Date`.")
