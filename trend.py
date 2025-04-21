import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from streamlit_autorefresh import st_autorefresh  # ← ADDED for auto-refresh
from transformers import pipeline  # ← ADDED for Transformer-based final analysis

# Page Config
st.set_page_config(page_title="FinTalk Trend Analysis", layout="wide")

# Auto-refresh every 600 seconds (600000 milliseconds) ← CHANGED from 60 to 600 seconds
st_autorefresh(interval=600000, key="datarefresh")

st.title("📊 FinTalk Pro: Financial Trend Analysis")

# Load Dataset
DATA_PATH = "D:\FinTalk Pro\Trend analysis\generated_transactions_with_dates.csv"
df = pd.read_csv(DATA_PATH)

# Data Preprocessing
df['transaction_datetime'] = pd.to_datetime(df['transaction_date'] + ' ' + df['transaction_time'])
df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
df['balance'] = pd.to_numeric(df['balance'], errors='coerce')
df['type'] = df['amount'].apply(lambda x: 'Income' if x > 0 else 'Expense')

# ------------------- ADVANCED FEATURES SECTION --------------------

st.markdown("## 🛠️ Advanced Analytics Features")

# Date Range Filter
st.subheader("📅 Filter by Date Range (Transaction Time)")
min_date = df['transaction_datetime'].dt.date.min()
max_date = pd.to_datetime("2050-12-31").date()
start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
end_date = st.date_input("End Date", min_value=min_date, max_value=max_date, value=df['transaction_datetime'].dt.date.max())
filtered_df = df[(df['transaction_datetime'].dt.date >= start_date) & (df['transaction_datetime'].dt.date <= end_date)]

# Category Filter
categories = st.multiselect("📂 Filter by Transaction Categories", options=sorted(df['transaction_category'].unique()), default=list(df['transaction_category'].unique()))
filtered_df = filtered_df[filtered_df['transaction_category'].isin(categories)]

# Time-Based Aggregation
st.subheader("⏱️ Time-based Aggregation")
agg_level = st.radio("Group by", ["Minute", "Hour"], horizontal=True)
if agg_level == "Hour":
    filtered_df['time_bucket'] = filtered_df['transaction_datetime'].dt.hour
else:
    filtered_df['time_bucket'] = filtered_df['transaction_datetime'].dt.strftime("%H:%M")

agg_data = filtered_df.groupby(['time_bucket', 'type'])['amount'].sum().unstack().fillna(0)
st.bar_chart(agg_data)

# Cumulative Balance Trend
st.subheader("📈 Cumulative Balance Trend")
cumulative_df = filtered_df.sort_values("transaction_datetime").copy()
cumulative_df["cumulative_balance"] = cumulative_df["amount"].cumsum()
fig, ax = plt.subplots()
ax.plot(cumulative_df["transaction_datetime"], cumulative_df["cumulative_balance"], marker='o')
ax.set_xlabel("Time")
ax.set_ylabel("Cumulative Balance")
ax.set_title("Cumulative Balance Over Time")
plt.xticks(rotation=45)
st.pyplot(fig)

# Chart Type Toggle
st.subheader("📊 Toggle Chart Type")
chart_option = st.selectbox("Choose Chart Type", ["Line Chart", "Bar Chart"])
selected_type = st.radio("Select Type", ["Income", "Expense"])
chart_df = filtered_df[filtered_df["type"] == selected_type].copy()
chart_df = chart_df.groupby(chart_df["transaction_datetime"].dt.strftime("%H:%M"))["amount"].sum()

if chart_option == "Line Chart":
    st.line_chart(chart_df)
else:
    st.bar_chart(chart_df)

# Anomaly Detection
st.subheader("🚨 Anomaly Detection")
threshold = st.slider("Set Outlier Threshold (Absolute Amount)", 10000, 100000, 25000)
outliers = filtered_df[filtered_df['amount'].abs() > threshold]
if not outliers.empty:
    st.warning(f"⚠️ Detected {len(outliers)} potential anomalies:")
    st.dataframe(outliers[['transaction_datetime', 'merchant', 'amount', 'transaction_category']])
else:
    st.success("No anomalies detected.")

# ------------------- ADVISOR NOTE SECTION --------------------
st.subheader("📝 Advisor Note")
NOTE_FILE = "advisor_note.txt"

# Load existing note
if os.path.exists(NOTE_FILE):
    with open(NOTE_FILE, "r", encoding="utf-8") as f:
        saved_note = f.read()
else:
    saved_note = ""

note_input = st.text_area("Write your note here:", value=saved_note, height=150)

# PDF generation function
def create_pdf(note_text):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 40

    p.setFont("Helvetica-Bold", 14)
    p.drawString(40, y, "FinTalk Pro: Financial Trend Analysis")
    y -= 30

    p.setFont("Helvetica", 11)
    p.drawString(40, y, f"Selected Date Range: {start_date} to {end_date}")
    y -= 20
    p.drawString(40, y, f"Selected Categories: {', '.join(categories)}")
    y -= 30

    if note_text.strip():
        p.setFont("Helvetica-Bold", 12)
        p.drawString(40, y, "Advisor Note:")
        y -= 20
        p.setFont("Helvetica", 11)
        for line in note_text.splitlines():
            p.drawString(50, y, line)
            y -= 15
            if y < 50:
                p.showPage()
                y = height - 40
        y -= 10

    p.setFont("Helvetica-Bold", 12)
    p.drawString(40, y, f"Anomaly Threshold: {threshold}")
    y -= 20
    p.setFont("Helvetica", 11)
    p.drawString(40, y, f"Anomalies Detected: {len(outliers)}")
    y -= 30

    if not outliers.empty:
        for _, row in outliers.head(20).iterrows():  # limit to 20 entries
            txt = f"{row['transaction_datetime']} | {row['merchant']} | Rs.{row['amount']} | {row['transaction_category']}"
            p.drawString(50, y, txt)
            y -= 15
            if y < 50:
                p.showPage()
                y = height - 40

    p.save()
    buffer.seek(0)
    return buffer

col1, col2 = st.columns(2)

with col1:
    if st.button("✅ Save Note"):
        with open(NOTE_FILE, "w", encoding="utf-8") as f:
            f.write(note_input)
        st.success("Note saved!")

with col2:
    if st.button("🗑️ Clear Note"):
        note_input = ""  # Clear note from current session
        if os.path.exists(NOTE_FILE):
            os.remove(NOTE_FILE)
        st.success("Note cleared!")

# Regenerate PDF using current note_input (which may be blank if cleared)
pdf_data = create_pdf(note_input)
st.download_button(
    label="⬇️ Download PDF Report",
    data=pdf_data,
    file_name="fintalk_trend_analysis.pdf",
    mime="application/pdf"
)

# ------------------- TRANSFORMER FINAL ANALYSIS --------------------
st.header("🔍 See AI Full Analysis")
st.write("Click the button below to generate an in-depth financial analysis based on your data.")
if st.button("Final Analysis"):
    with st.spinner("Analyzing trends and generating summary..."):
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

        summary_text = f"""
        Financial Trend Analysis Summary:
        - Date Range: {start_date} to {end_date}
        - Selected Categories: {', '.join(categories)}
        - Total Transactions: {len(filtered_df)}
        - Total Income: Rs.{filtered_df[filtered_df['type'] == 'Income']['amount'].sum():,.2f}
        - Total Expense: Rs.{filtered_df[filtered_df['type'] == 'Expense']['amount'].sum():,.2f}
        - Anomalies Detected: {len(outliers)}
        """

        # Split long content if needed
        output = summarizer(summary_text, max_length=130, min_length=30, do_sample=False)
        st.write(output[0]['summary_text'])

# ------------------- FOOTER --------------------
st.markdown("---")
st.markdown("<div style='text-align:center; color: gray;'>Your Financial Story — Designed & Delivered by <b>FinTalk Pro</b>.</div>", unsafe_allow_html=True)
