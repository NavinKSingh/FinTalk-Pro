import streamlit as st
import pandas as pd
import plotly.express as px
import base64
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
import threading
import time
from transformers import pipeline

# --- MUST BE FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="FinTalk Portfolio", layout="wide")

# --- Config ---
USD_TO_INR = 83.0
NOTE_PATH = "advisor_note.txt"

# --- Load Portfolio Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("D:\FinTalk Pro\Investment portfolio\data2\\bank_investment_portfolio_dataset.csv")
    return df

df = load_data()

# Filter to one user (first user's portfolio)
user_id = df['user_id'].iloc[0]
df = df[df['user_id'] == user_id]

# Convert to INR and calculate return
df['investment_amount'] *= USD_TO_INR
df['current_value'] *= USD_TO_INR
df['return (%)'] = ((df['current_value'] - df['investment_amount']) / df['investment_amount'] * 100).round(2)

# --- Header ---
st.markdown(f"""
    <h1 style='text-align: center; color: #2E86AB;'>🏦 FinTalk Portfolio Dashboard</h1>
    <p style='text-align: center; font-size: 18px;'>Detailed investment insights for client ID: <b>{user_id}</b></p>
""", unsafe_allow_html=True)

# --- Summary Metrics ---
total_investment = df['investment_amount'].sum()
total_value = df['current_value'].sum()
total_return = round((total_value - total_investment) / total_investment * 100, 2)
top_sector = df.groupby('sector')['current_value'].sum().idxmax()
high_risk_ratio = df['risk_level'].value_counts(normalize=True).get('High', 0)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Investment (₹)", f"{total_investment:,.0f}")
col2.metric("Current Value (₹)", f"{total_value:,.0f}")
col3.metric("Net Return (%)", f"{total_return}%")
col4.metric("Top Sector", top_sector)

# --- Sector Investment ---
st.markdown("### 📊 Sector-wise Investment Value")
sector_chart_df = df.groupby('sector')['current_value'].sum().reset_index()
sector_bar_fig = px.bar(sector_chart_df, x='sector', y='current_value', 
                        title='Investment Value by Sector (₹)', 
                        text_auto=True, color='sector')
sector_bar_fig.update_layout(xaxis_title="Sector", yaxis_title="Investment Value (₹)", xaxis_tickangle=-45)
st.plotly_chart(sector_bar_fig, use_container_width=True)

# --- Risk Analysis ---
st.markdown("### ⚠️ Risk Level Distribution")
risk_chart_df = df['risk_level'].value_counts().reset_index()
risk_chart_df.columns = ['risk_level', 'count']
risk_bar_fig = px.bar(risk_chart_df, x='risk_level', y='count', 
                      title='Distribution of Risk Levels', 
                      text_auto=True, color='risk_level')
risk_bar_fig.update_layout(xaxis_title="Risk Level", yaxis_title="Number of Assets")
st.plotly_chart(risk_bar_fig, use_container_width=True)

# --- Asset Allocation Pie Chart ---
st.markdown("### 🧁 Portfolio Allocation by Sector (Pie Chart)")
sector_pie = df.groupby('sector')['current_value'].sum().reset_index()
fig = px.pie(sector_pie, values='current_value', names='sector',
             title='Current Portfolio Distribution by Sector (₹)',
             color_discrete_sequence=px.colors.sequential.RdBu)
st.plotly_chart(fig, use_container_width=True)

# --- Table View ---
with st.expander("📂 Full Portfolio Details"):
    st.dataframe(df.reset_index(drop=True), use_container_width=True)

# --- Strategic Note ---
st.markdown(f"""
<div style='margin-top: 40px; font-size: 16px;'>
<b>💡 Strategic Insight:</b><br>
Your current portfolio shows a high allocation to the <b>{top_sector}</b> sector and <b>{high_risk_ratio * 100:.1f}%</b> high-risk assets.
{"Consider diversifying into low-risk instruments for better stability." if high_risk_ratio > 0.5 else "Your portfolio is well balanced. Continue monitoring market trends."}
</div>
""", unsafe_allow_html=True)

# --- Option 5: Portfolio Health Score ---
risk_score = (1 - high_risk_ratio) * 100
health_score = round((total_return * 0.7 + risk_score * 0.3), 2)

st.markdown("### 🧮 Portfolio Health Score")
st.metric("📈 Health Score (0-100)", f"{health_score}/100")

# --- Option 6: Advisor Notes with Persistent Save/Clear ---
st.markdown("### 🗒️ Advisor Notes")

# Load existing note if present
if os.path.exists(NOTE_PATH):
    with open(NOTE_PATH, "r") as f:
        saved_note = f.read()
else:
    saved_note = ""

user_note = st.text_area("Add any insights, suggestions or comments below:", value=saved_note)

col_save, col_clear = st.columns([1, 1])

with col_save:
    if st.button("💾 Save Note"):
        with open(NOTE_PATH, "w") as f:
            f.write(user_note)
        st.success("✅ Note saved!")

with col_clear:
    if st.button("🧹 Clear Note"):
        if os.path.exists(NOTE_PATH):
            os.remove(NOTE_PATH)
        user_note = ""
        st.success("🗑️ Note cleared!")

# --- Final AI-Powered Analysis using Hugging Face Transformer ---
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

dashboard_text = f"""
Client ID: {user_id}
Total Investment: ₹{total_investment:,.0f}
Current Value: ₹{total_value:,.0f}
Net Return: {total_return}%
Top Sector: {top_sector}
High Risk Ratio: {high_risk_ratio:.2f}
Portfolio Health Score: {health_score}/100
"""

sector_data = ", ".join([f"{row['sector']} ({int(row['current_value'])} ₹)" for _, row in sector_chart_df.iterrows()])
dashboard_text += f"\nSector Allocation: {sector_data}"

risk_data = ", ".join([f"{row['risk_level']} ({row['count']})" for _, row in risk_chart_df.iterrows()])
dashboard_text += f"\nRisk Levels: {risk_data}"

# --- Generate PDF using ReportLab (₹ replaced with Rs.) ---
def generate_pdf():
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, height - 50, "FinTalk Portfolio Report")

    c.setFont("Helvetica", 12)
    y = height - 100
    c.drawString(50, y, f"Client ID: {user_id}")
    y -= 20
    c.drawString(50, y, f"Total Investment (Rs.): {total_investment:,.0f}")
    y -= 20
    c.drawString(50, y, f"Current Value (Rs.): {total_value:,.0f}")
    y -= 20
    c.drawString(50, y, f"Net Return (%): {total_return}%")
    y -= 20
    c.drawString(50, y, f"Top Sector: {top_sector}")
    y -= 20
    c.drawString(50, y, f"Health Score: {health_score}/100")
    y -= 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Portfolio Details")
    y -= 20
    c.setFont("Helvetica", 9)

    for i, row in df.iterrows():
        if y < 60:
            c.showPage()
            y = height - 50
        text = f"{row['asset_type']} | {row['asset_name']} ({row['ticker']}) | Rs.{int(row['investment_amount'])} ➡ Rs.{int(row['current_value'])} | {row['sector']} | {row['risk_level']}"
        c.drawString(50, y, text)
        y -= 15

    # Add advisor note if it exists
    if os.path.exists(NOTE_PATH):
        with open(NOTE_PATH, "r") as f:
            note = f.read()

        c.setFont("Helvetica-Bold", 12)
        y -= 30
        c.drawString(50, y, "Advisor Note:")
        y -= 15
        c.setFont("Helvetica", 10)
        for line in note.splitlines():
            if y < 60:
                c.showPage()
                y = height - 50
            c.drawString(60, y, line)
            y -= 15

    c.save()
    buffer.seek(0)
    return buffer

# --- PDF Download Button ---
st.markdown("### 📥 Download Portfolio Report")
pdf_buffer = generate_pdf()
st.download_button(
    label="📄 Download Portfolio Report (PDF)",
    data=pdf_buffer,
    file_name="FinTalk_Portfolio_Report.pdf",
    mime="application/pdf"
)

# --- See Full AI Analysis and Generate Final Analysis (Moved below Download) ---
st.markdown("### 🔍 See Full AI Analysis")

if "final_summary" not in st.session_state:
    st.session_state.final_summary = None

if st.button("Final Analysis"):
    combined_text = f"{dashboard_text}\n\nAdvisor Note: {user_note}"
    summarized = summarizer(combined_text, max_length=150, min_length=50, do_sample=False)
    st.session_state.final_summary = summarized[0]['summary_text']

if st.session_state.final_summary:
    st.subheader("📑 Final Analysis Summary")
    st.write(st.session_state.final_summary)

# --- Auto-refresh every 600 seconds (10 minutes) ---
def auto_refresh(interval=600):
    def rerun():
        time.sleep(interval)
        st.experimental_rerun()
    threading.Thread(target=rerun, daemon=True).start()

auto_refresh(600)

st.markdown("---")
st.markdown("<div style='text-align:center; color: gray;'>Your Financial Story — Designed & Delivered by <b>FinTalk Pro</b>.</div>", unsafe_allow_html=True)