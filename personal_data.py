import streamlit as st
import pandas as pd
import numpy as np
from fpdf import FPDF
from io import BytesIO
import re
import time
import os
from transformers import pipeline

# --------------------------------------
# ✅ Page Config (Must be first)
# --------------------------------------
st.set_page_config(page_title="FinTalk Pro Dashboard", layout="wide")

# --------------------------------------
# 🔄 Auto Refresh every 600 seconds
# --------------------------------------
st.query_params.update(run=str(time.time()))

# --------------------------------------
# 📊 Load CSV Data
# --------------------------------------
DATA_PATH = "D:\FinTalk Pro\personal data\data\\financial_dashboard_dataset.csv"
NOTE_PATH = 'advisor_note_dashboard.txt'

@st.cache_data(ttl=600)  # refresh every 600 seconds
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

df = load_data()

# --------------------------------------
# 💼 Financial Insights
# --------------------------------------
def financial_summary(df):
    income = df['monthly_income'].mean()
    expenses = df['monthly_expenses'].mean()
    net_savings = income - expenses

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Monthly Income", f"₹{income:.2f}")
    col2.metric("Avg Monthly Expenses", f"₹{expenses:.2f}")
    col3.metric("Net Savings", f"₹{net_savings:.2f}")

    if net_savings < 0:
        st.error("⚠️ You're spending more than you earn.")
    else:
        st.success("✅ You're saving money on average.")

def top_spending_categories(df):
    st.subheader("🛍️ Spending by Category")
    cat_sum = df.groupby("transaction_category")["recent_transaction_amount"].sum().sort_values()
    st.bar_chart(cat_sum.abs())
    return cat_sum

def credit_utilization_analysis(df):
    st.subheader("💳 Credit Utilization")
    avg_util = df["credit_utilization_(%)"].mean()
    st.metric("Average Utilization", f"{avg_util:.2f}%")
    if avg_util > 30:
        st.warning("⚠️ Try to keep credit utilization under 30%.")
    else:
        st.success("✅ Your credit utilization is healthy.")

# --------------------------------------
# 📄 PDF Report Generator
# --------------------------------------
def generate_report_text(df):
    income = df['monthly_income'].mean()
    expenses = df['monthly_expenses'].mean()
    net_savings = income - expenses
    avg_util = df["credit_utilization_(%)"].mean()
    top_cats = df.groupby("transaction_category")["recent_transaction_amount"].sum().sort_values()

    report = f"📊 Financial Report Summary\n\n"
    report += f"Avg Monthly Income: ₹{income:.2f}\n"
    report += f"Avg Monthly Expenses: ₹{expenses:.2f}\n"
    report += f"Net Savings: ₹{net_savings:.2f}\n\n"
    report += f"Top Spending Categories:\n"
    for cat, amt in top_cats.items():
        report += f" - {cat}: ₹{abs(amt):.2f}\n"
    report += f"\nAvg Credit Utilization: {avg_util:.2f}%\n"

    # Include advisor note if exists
    if os.path.exists(NOTE_PATH):
        with open(NOTE_PATH, "r") as f:
            note = f.read().strip()
            if note:
                report += f"\n---\n📌 Advisor Notes:\n{note}\n"

    return report

def clean_text_for_pdf(text):
    emoji_pattern = re.compile("[" 
        u"\U0001F600-\U0001F64F" 
        u"\U0001F300-\U0001F5FF" 
        u"\U0001F680-\U0001F6FF" 
        u"\U0001F1E0-\U0001F1FF" 
        u"\U00002700-\U000027BF" 
        u"\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    return text.replace("₹", "Rs.")

def generate_pdf_report(report_text):
    clean_text = clean_text_for_pdf(report_text)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in clean_text.split('\n'):
        pdf.multi_cell(0, 10, line)
    return BytesIO(pdf.output(dest='S').encode('latin1'))

# --------------------------------------
# 🧭 Sidebar Navigation
# --------------------------------------
st.sidebar.title("📁 FinTalk Pro")
st.sidebar.markdown("Your smart finance assistant 💼")
st.sidebar.write("🔢 Records loaded:", df.shape[0])

# --------------------------------------
# 🚀 Main Dashboard
# --------------------------------------
st.markdown("<h1 style='text-align:center;'>🏦 FinTalk Personal Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>Detailed investment insights for client ID: 1</h3>", unsafe_allow_html=True) # Add this line for details

# 🧾 Summary
st.header("💼 Monthly Summary")
financial_summary(df)

# 📈 Income vs Expenses
st.header("📈 Income vs Expenses Trend")
st.line_chart(df[['monthly_income', 'monthly_expenses']])

# 🛒 Spending Categories
top_spending_categories(df)

# 💳 Credit Analysis
credit_utilization_analysis(df)

# 📉 Loan Overview
st.subheader("📉 Loan Overview")
st.dataframe(df[["loan_balance", "monthly_loan_payment"]])

# 🥦 Groceries Budget
st.subheader("🥦 Groceries Budget Status")
df["groceries_status"] = np.where(
    df["groceries_spent"] > df["groceries_budget"], "Over Budget",
    np.where(df["groceries_spent"] > 0.8 * df["groceries_budget"], "Near Limit", "Under Budget")
)
st.dataframe(df[["groceries_budget", "groceries_spent", "groceries_status"]])

# 🚨 Security Alerts
st.subheader("🚨 Security Alerts")
st.dataframe(df[["security_alert_type", "recent_transaction_amount", "transaction_category"]])

# 🗒️ Advisor Notes
st.subheader("🗒️ Advisor Notes")

# Load existing note (for the first run)
if "note_loaded" not in st.session_state:
    if os.path.exists(NOTE_PATH):
        with open(NOTE_PATH, "r") as f:
            st.session_state["note_content"] = f.read()
    else:
        st.session_state["note_content"] = ""
    st.session_state["note_loaded"] = True

# Show the text area
note_input = st.text_area("Write advisor note here:", value=st.session_state["note_content"], height=150)

# Buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("💾 Save Note"):
        with open(NOTE_PATH, "w") as f:
            f.write(note_input)
        st.success("✅ Note saved!")
        st.session_state["note_content"] = note_input

with col2:
    if st.button("🧹 Clear Note"):
        if os.path.exists(NOTE_PATH):
            os.remove(NOTE_PATH)
        st.session_state["note_content"] = ""
        st.success("✅ Note cleared!")

# --------------------------------------
# 📄 Download PDF Report
# --------------------------------------
st.header("📥 Download Financial Report")
pdf_file = generate_pdf_report(generate_report_text(df))
st.download_button("📄 Download PDF Report", data=pdf_file, file_name="FinTalk_Pro_Report.pdf", mime="application/pdf")

# --------------------------------------
# 🤖 Final Analysis (Hugging Face Model)
# --------------------------------------

# Load Hugging Face transformer model for final analysis
@st.cache_data
def load_model():
    return pipeline("text-generation", model="gpt2")  # You can change this model to something else if needed

model = load_model()

# Button to generate final analysis
st.header("🔍 See AI Full Analysis")
st.write("Click the button below to generate an in-depth financial analysis based on your data.")

# Button to generate final analysis
if st.button("Final Analysis"):
    input_text = f"Financial Summary: Avg Monthly Income: ₹{df['monthly_income'].mean():.2f}, " \
                 f"Avg Monthly Expenses: ₹{df['monthly_expenses'].mean():.2f}, " \
                 f"Net Savings: ₹{df['monthly_income'].mean() - df['monthly_expenses'].mean():.2f}, " \
                 f"Avg Credit Utilization: {df['credit_utilization_(%)'].mean():.2f}%."
    analysis = model(input_text, max_length=300)[0]['generated_text']  # Increased max_length for more output
    st.subheader("📝 Final Analysis")
    st.write(analysis)

# --------------------------------------
# 📌 Footer
# --------------------------------------
st.markdown("---")
st.markdown("<div style='text-align:center; color: gray;'>Your Financial Story — Designed & Delivered by <b>FinTalk Pro</b>.</div>", unsafe_allow_html=True)
