import streamlit as st
import pandas as pd
import subprocess
import os
import plotly.express as px

st.set_page_config(page_title="FraudDetectX", layout="wide")
st.title("🔎 Fraud Detection System")

uploaded_file = st.file_uploader("📁 Upload Subsidy Claims CSV", type=["csv"])

if uploaded_file:
    os.makedirs("data/raw", exist_ok=True)
    file_path = "data/raw/uploaded_subsidy_claims.csv"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    df = pd.read_csv(file_path)
    st.success(f"✅ Uploaded {len(df)} records")
    st.dataframe(df.head(10))

    if st.button("🚀 Run Fraud Detection"):
        with st.spinner("Running hybrid pipeline..."):
            try:
                subprocess.run(
                    ["python", "scripts/run_pipeline.py"], 
                    check=True
                )
            except subprocess.CalledProcessError as e:
                st.error(f"❌ Pipeline failed with exit code {e.returncode}")
                st.error("Check the terminal logs for details.")
                st.stop()

        # Load results
        results_path = "data/results/fraud_results.csv"
        if os.path.exists(results_path):
            df_results = pd.read_csv(results_path)
            st.subheader("📊 Fraud Detection Results")
            st.dataframe(df_results)

            fraud_summary = df_results['FraudType'].value_counts().reset_index()
            fraud_summary.columns = ['FraudType', 'Count']
            fig = px.pie(fraud_summary, names='FraudType', values='Count', title='Fraud Type Distribution')
            st.plotly_chart(fig, use_container_width=True)

            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results", csv, "fraud_results.csv", "text/csv")
        else:
            st.error("❌ Results file not found")

else:
    st.info("⬆️ Upload a CSV with columns: Name, Aadhaar, ClaimAmount, SubsidyType, Date")
