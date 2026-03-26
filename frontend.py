import streamlit as st
import pandas as pd
import requests
import json

# =========================
# CONFIG
# =========================
API_URL = "https://your-api.onrender.com"  # 🔥 UPDATE THIS

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide"
)

st.title("💳 Fraud Detection Dashboard")
st.markdown("Upload transaction data and detect fraudulent activities in real time.")

# =========================
# LOAD SAMPLE DATA
# =========================
st.sidebar.header("Options")

if st.sidebar.button("Load Sample Data"):
    try:
        response = requests.get(f"{API_URL}/sample")
        sample_data = response.json()
        st.session_state["data"] = pd.DataFrame(sample_data)
        st.success("Sample data loaded!")
    except:
        st.error("Could not load sample data")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state["data"] = df

# =========================
# DISPLAY DATA
# =========================
if "data" in st.session_state:
    df = st.session_state["data"]

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # =========================
    # RUN PREDICTION
    # =========================
    if st.button("🚀 Run Fraud Detection"):

        try:
            with st.spinner("Processing..."):

                files = {"file": uploaded_file} if uploaded_file else None

                response = requests.post(
                    f"{API_URL}/predict_csv",
                    files=files
                )

                if response.status_code != 200:
                    st.error(f"API Error: {response.text}")
                else:
                    result = response.json()
                    result_df = pd.DataFrame(result)

                    st.session_state["result"] = result_df

        except Exception as e:
            st.error(f"Error: {e}")

# =========================
# DISPLAY RESULTS
# =========================
if "result" in st.session_state:
    result_df = st.session_state["result"]

    st.subheader("✅ Prediction Results")
    st.dataframe(result_df)

    # =========================
    # SUMMARY METRICS
    # =========================
    st.subheader("📊 Summary")

    total = len(result_df)
    fraud_cases = int(result_df["prediction"].sum())
    fraud_rate = result_df["prediction"].mean()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", total)
    col2.metric("Fraud Cases", fraud_cases)
    col3.metric("Fraud Rate", f"{fraud_rate:.2%}")

    # =========================
    # CHART
    # =========================
    st.subheader("📈 Fraud Distribution")
    st.bar_chart(result_df["prediction"].value_counts())

    # =========================
    # SHAP EXPLANATION (FIRST ROW)
    # =========================
    if "shap_top_features" in result_df.columns:
        st.subheader("🔍 SHAP Explanation (First Row)")
        try:
            shap_data = result_df.iloc[0]["shap_top_features"]
            if isinstance(shap_data, str):
                shap_data = json.loads(shap_data)

            st.json(shap_data)
        except:
            st.warning("Could not parse SHAP data")

    # =========================
    # DOWNLOAD REPORT
    # =========================
    st.subheader("📥 Download Report")

    csv = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Fraud Report",
        data=csv,
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("Built with ❤️ using FastAPI + Streamlit + RandomForest + SHAP")