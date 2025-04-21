import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from fpdf import FPDF
from transformers import pipeline  # ✅ NEW

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="FinTalk Risk Analyzer", layout="wide")

# -----------------------------
# Load Dataset
# -----------------------------
DATA_PATH = r"D:\FinTalk Pro\Risk detection\user_risk_features_dataset.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["transaction_time"])
    return df[df["user_id"] == 1017]  # Filter for single user

df = load_data()

st.title("🔍 FinTalk Pro: Risk Analyzer")
st.markdown("Personalized risk insights for user **#1017**")

# -----------------------------
# 1. Credit Utilization Risk
# -----------------------------
st.header("💳 Credit Utilization Risk")
avg_util = df["credit_utilization_percent"].mean()
st.progress(min(avg_util / 100, 1.0))
st.metric("Avg Credit Utilization", f"{avg_util:.2f} %")

if avg_util > 30:
    st.warning("⚠️ High credit utilization. Try to keep it under 30%.")
    st.markdown("💡 _Tip: Pay down your balance or request a credit limit increase._")
else:
    st.success("✅ Your credit utilization is within the healthy range.")

# -----------------------------
# 2. Loan Repayment Risk
# -----------------------------
st.header("📉 Loan Repayment Risk")
df["repayment_ratio"] = df["monthly_loan_payment"] / df["monthly_income"]
avg_ratio = df["repayment_ratio"].mean()

st.metric("Repayment-to-Income Ratio", f"{avg_ratio*100:.2f} %")

if avg_ratio > 0.4:
    st.error("🚨 High loan repayment burden!")
else:
    st.success("✅ Loan repayment burden is manageable.")

# -----------------------------
# 3. Over-Budget Spending Risk
# -----------------------------
st.header("🛒 Over-Budget Spending Risk")
df["over_budget"] = df["groceries_spent"] > df["groceries_budget"]
over_budget_count = df["over_budget"].sum()

if over_budget_count > 0:
    st.warning(f"⚠️ You exceeded your groceries budget in {over_budget_count} out of {len(df)} transactions.")
    st.bar_chart(df[["groceries_budget", "groceries_spent"]])
else:
    st.success("✅ You're staying within your grocery budget.")

# -----------------------------
# 4. Income Volatility Risk
# -----------------------------
st.header("📈 Income Volatility Risk")
income_std = df["monthly_income"].std()
income_mean = df["monthly_income"].mean()

volatility_ratio = income_std / income_mean
st.metric("Income Std Dev", f"{income_std:.2f}")
st.line_chart(df[["monthly_income"]])

if volatility_ratio > 0.25:
    st.warning("⚠️ Your income is highly variable.")
    st.markdown("💡 _Tip: Consider building a financial buffer._")
else:
    st.success("✅ Your income appears stable.")

# -----------------------------
# 5. Transaction Anomalies / Fraud Risk
# -----------------------------
st.header("🔍 Fraud & Anomaly Risk")
anomaly_df = df[df["is_large"] | df["is_foreign"] | df["is_unusual_time"]]

if not anomaly_df.empty:
    st.error(f"🚨 {len(anomaly_df)} suspicious transactions detected!")
    st.dataframe(anomaly_df[["transaction_time", "transaction_amount", "merchant", "is_large", "is_foreign", "is_unusual_time"]])
    st.markdown("🛡️ _Please review and mark transactions as safe or report._")
else:
    st.success("✅ No suspicious transactions detected.")

# -----------------------------
# 6. Account Balance Risk
# -----------------------------
st.header("🏦 Account Balance Risk")
avg_expense = df["monthly_expenses"].mean()
low_balance_df = df[df["account_balance"] < avg_expense]

if not low_balance_df.empty:
    st.warning(f"⚠️ Low balance detected in {len(low_balance_df)} records. You may not cover monthly expenses.")
else:
    st.success("✅ Account balance is generally sufficient.")

# -----------------------------
# 7. Security Alert Monitoring
# -----------------------------
st.header("🔐 Security Alerts")
alerts = df[df["security_alert_type"].notnull()]

if not alerts.empty:
    st.error(f"🚨 Security alerts found: {alerts['security_alert_type'].nunique()} types")
    st.dataframe(alerts[["transaction_time", "merchant", "security_alert_type"]])
    st.markdown("🔑 _Please follow up with appropriate security measures (e.g., password reset)._")
else:
    st.success("✅ No security alerts found.")

# -----------------------------
# 8. EMI Clustering Risk
# -----------------------------
st.header("📆 EMI Load Risk")
avg_emi_count = df["emi_count"].mean()

if avg_emi_count > 5:
    st.error(f"🚨 You have {avg_emi_count:.1f} EMIs on average – high risk of overload.")
else:
    st.success(f"✅ EMI load is within a healthy range: {avg_emi_count:.1f}")

# -----------------------------
# Option 6: High-Risk Transaction Days
# -----------------------------
st.header("📅 High-Risk Transaction Days")
df["date_only"] = df["transaction_time"].dt.date
daily_anomalies = df[df["is_large"] | df["is_foreign"] | df["is_unusual_time"]].groupby("date_only").size()
high_risk_days = daily_anomalies[daily_anomalies > 2]

if not high_risk_days.empty:
    st.error("🚨 Days with multiple suspicious transactions detected:")
    st.dataframe(high_risk_days.rename("Suspicious Txns").reset_index())
else:
    st.success("✅ No high-risk days found.")

# -----------------------------
# Option 7: Smart Bank Recommendations
# -----------------------------
st.header("🤖 Smart Bank Recommendations")

recommendations = []

if avg_util > 30:
    recommendations.append("🔄 Consider lowering your credit utilization below 30%.")
if avg_ratio > 0.4:
    recommendations.append("💡 Re-evaluate your loan structure or seek financial counseling.")
if volatility_ratio > 0.25:
    recommendations.append("📉 Set up an emergency fund due to income fluctuations.")
if len(low_balance_df) > 5:
    recommendations.append("🏦 Maintain a minimum balance to avoid penalties.")
if avg_emi_count > 5:
    recommendations.append("📆 Too many EMIs – consolidate loans if possible.")

if not recommendations:
    st.success("✅ Your financial profile looks healthy. Keep it up!")
else:
    for rec in recommendations:
        st.markdown(rec)

# -----------------------------
# Advisor Note Section
# -----------------------------
NOTE_FILE = "advisor_note.json"

# Load saved note if exists
if os.path.exists(NOTE_FILE):
    with open(NOTE_FILE, "r") as f:
        saved_note = json.load(f).get("note", "")
else:
    saved_note = ""

st.header("📝 Advisor Note")
note = st.text_area("Enter your personalized note/advice here:", value=saved_note, height=150)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("💾 Save Note"):
        with open(NOTE_FILE, "w") as f:
            json.dump({"note": note}, f)
        st.success("Note saved successfully!")

with col2:
    if st.button("🗑️ Clear Note"):
        if os.path.exists(NOTE_FILE):
            os.remove(NOTE_FILE)
        note = ""
        st.success("Note cleared!")

# -----------------------------
# PDF Generation Section
# -----------------------------
st.header("📄 Download Risk Report")

def generate_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="FinTalk Pro: Risk Report for User #1017", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, "Credit Utilization Risk", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Average Utilization: {avg_util:.2f}%", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, "Loan Repayment Risk", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Repayment-to-Income Ratio: {avg_ratio*100:.2f}%", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, "Over-Budget Spending Risk", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Over-budget Transactions: {over_budget_count}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, "Income Volatility Risk", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Income Std Dev: {income_std:.2f}", ln=True)
    pdf.cell(0, 10, f"Volatility Ratio: {volatility_ratio:.2f}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, "Fraud & Anomaly Risk", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Suspicious Transactions: {len(anomaly_df)}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, "Account Balance Risk", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Low Balance Records: {len(low_balance_df)}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, "Security Alert Monitoring", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Unique Alerts: {alerts['security_alert_type'].nunique() if not alerts.empty else 0}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, "EMI Load Risk", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Average EMI Count: {avg_emi_count:.1f}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, "Advisor Note", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, note if note else "No advisor note saved.")
    pdf.ln(5)

    return pdf.output(dest="S").encode("latin1")

st.download_button("📥 Download PDF Report", data=generate_pdf(), file_name="risk_report_user_1017.pdf", mime="application/pdf")

# -----------------------------
# Final Transformer-based Analysis Summary (UPDATED)
# -----------------------------
st.header("🔍 See AI Full Analysis")
st.write("Click the button below to generate an in-depth financial analysis based on your data.")

@st.cache_resource
def get_summary_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = get_summary_model()

summary_input = f"""
Credit Utilization: {avg_util:.2f}%. Repayment Ratio: {avg_ratio*100:.2f}%. Over-budget grocery txns: {over_budget_count}. 
Income Std Dev: {income_std:.2f}, Volatility Ratio: {volatility_ratio:.2f}. 
Suspicious Transactions: {len(anomaly_df)}. Low Balance Records: {len(low_balance_df)}. 
Security Alerts: {alerts['security_alert_type'].nunique() if not alerts.empty else 0}. Avg EMI Count: {avg_emi_count:.1f}.
"""

if st.button("Final Analysis"):
    with st.spinner("Generating AI summary..."):
        summary = summarizer(summary_input, max_length=250, min_length=80, do_sample=False)[0]['summary_text']
        st.markdown(
            f"<div style='background-color: rgba(255, 255, 255, 0); padding: 15px; border-radius: 10px;'>"
            f"<p style='color: white; font-size: 16px;'>{summary}</p></div>",
            unsafe_allow_html=True
        )

# -----------------------------
# Auto Refresh Every 600 Seconds
# -----------------------------
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=600000, key="data_refresh")  # 600 sec = 10 min

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("<div style='text-align:center; color: gray;'>Your Financial Story — Designed & Delivered by <b>FinTalk Pro</b>.</div>", unsafe_allow_html=True)
