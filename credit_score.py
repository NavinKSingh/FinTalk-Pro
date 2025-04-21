import pandas as pd
import streamlit as st
import re
from fpdf import FPDF
from io import BytesIO
import matplotlib.pyplot as plt
import os
from streamlit_autorefresh import st_autorefresh
from transformers import pipeline  # Hugging Face transformer

# ------------------------------------------
# Credit Score Storytelling Function
# ------------------------------------------
def credit_score_insights(row):
    utilization = (row['Total Credit Limit'] - row['Available Credit']) / row['Total Credit Limit'] * 100

    story = f"""### 🔍 Credit Summary

📊 Credit Utilization: {utilization:.2f}%  
{"⚠️ High credit utilization. Try to keep it below 30%." if utilization > 30 else "✅ Good credit utilization."}

💳 Loan Balance: ₹{row['Loan Balance']}  
📆 Monthly Payment: ₹{row['Monthly Loan Payment']}  
{"🚨 Missed Payments: " + str(row['Number of Missed Payments']) if row['Number of Missed Payments'] > 0 else "✅ No missed payments."}

📈 Credit Age: {row['Credit Age (in months)']} months  
{"⚠️ Short credit history." if row['Credit Age (in months)'] < 24 else "✅ Healthy credit age."}

📁 Credit Types: {row['Number of Credit Types (loan, card)']}  
🔍 New Credit Inquiries (6 months): {row['New Credit Inquiries (last 6 months)']}  
{"⚠️ Too many inquiries. Space out new applications." if row['New Credit Inquiries (last 6 months)'] > 2 else ""}

✅ On-Time Payment %: {row['On-Time Payment Percentage']}%  
{"⚠️ Improve on-time payments." if row['On-Time Payment Percentage'] < 85 else "🎯 Great on-time record!"}
"""
    return story

# ------------------------------------------
# PDF Export Utility
# ------------------------------------------
def clean_text_for_pdf(text):
    emoji_pattern = re.compile(
        "[" 
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
        u"\U00002700-\U000027BF"  
        u"\U000024C2-\U0001F251"  
        u"\U0001F900-\U0001F9FF"  
        "]", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text).replace("₹", "Rs.").replace("–", "-")

def generate_pdf_report(insight_text, advisor_note):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    full_text = insight_text.strip()
    if advisor_note.strip():
        full_text += f"\n\n---\n\n📌 Advisor Notes:\n{advisor_note.strip()}"

    for line in clean_text_for_pdf(full_text).split('\n'):
        pdf.multi_cell(0, 10, line)
    return BytesIO(pdf.output(dest='S').encode('latin-1', errors='replace'))

# ------------------------------------------
# Chart Utilities (Updated Styles Below)
# ------------------------------------------
def plot_credit_utilization(row):
    fig, ax = plt.subplots()
    used = row['Total Credit Limit'] - row['Available Credit']
    labels = ['Used Credit', 'Available Credit']
    sizes = [used, row['Available Credit']]
    colors = ['#EF553B', '#00CC96']
    explode = (0.05, 0.05)

    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                      autopct='%1.1f%%', startangle=140, pctdistance=0.85)

    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)
    ax.axis('equal')
    plt.setp(autotexts, size=12, weight="bold", color="white")
    plt.setp(texts, size=12)
    fig.patch.set_facecolor('#F9F9F9')
    ax.set_title("Credit Utilization", fontsize=16, color='#333333')
    return fig

def plot_payment_history_bar(row):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    percentage = [max(min(row['On-Time Payment Percentage'] - i * 1.5, 100), 0) for i in range(len(months))][::-1]

    colors = ['#00CC96', '#1FA4A8', '#3C8DAD', '#4C78A8', '#A05195', '#EF553B']

    fig, ax = plt.subplots()
    bars = ax.bar(months, percentage, color=colors)

    ax.set_title('On-Time Payment History (Last 6 Months)', fontsize=16, color='#333333')
    ax.set_ylim(0, 100)
    ax.set_ylabel('% On-Time', fontsize=12)
    ax.set_facecolor('#F9F9F9')
    fig.patch.set_facecolor('#F9F9F9')

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, weight='bold')
    return fig

def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

# ------------------------------------------
# Streamlit App
# ------------------------------------------
st.set_page_config(layout="centered")
st.title("🏦 FinTalk Pro – Credit Score Storytelling")

# 🔄 Auto refresh every 60 seconds
st_autorefresh(interval=600 * 1000, key="credit_autorefresh")

# Load transformer model from Hugging Face
summarizer = pipeline("summarization")

try:
    credit_df = pd.read_csv("D:\FinTalk Pro\Credit_score data\data1\credit_data.csv")
    selected = credit_df.iloc[0]  # Automatically select the first row

    st.write("### Credit Profile Summary")
    st.dataframe(selected.to_frame())

    st.subheader("📖 AI-Powered Credit Insights")
    insights = credit_score_insights(selected)
    st.markdown(insights)

    st.divider()

    # 📊 Credit Utilization
    st.subheader("📊 Credit Utilization")
    fig1 = plot_credit_utilization(selected)
    st.pyplot(fig1, use_container_width=True)
    chart1_bytes = fig_to_bytes(fig1)
    st.download_button("⬇️ Download Utilization Chart", data=chart1_bytes, file_name="credit_utilization.png", mime="image/png")

    # 📈 Payment History (Bar chart)
    st.subheader("📈 Payment History")
    fig2 = plot_payment_history_bar(selected)
    st.pyplot(fig2, use_container_width=True)
    chart2_bytes = fig_to_bytes(fig2)
    st.download_button("⬇️ Download Payment History Chart", data=chart2_bytes, file_name="payment_history.png", mime="image/png")

    # 🗒️ Advisor Notes (Persistent + Real-Time)
    st.markdown("### 🗒️ Advisor Notes")

    NOTE_PATH = "advisor_note_credit.txt"

    if os.path.exists(NOTE_PATH):
        with open(NOTE_PATH, "r") as f:
            saved_note = f.read()
    else:
        saved_note = ""

    user_note = st.text_area("Add your custom advisory note for this profile:", value=saved_note)

    col_save, col_clear = st.columns(2)

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

    # 📥 PDF Download
    st.markdown("### 📥 Download Full PDF Report")
    pdf = generate_pdf_report(insights, user_note)
    st.download_button("📥 Download PDF Report", data=pdf, file_name="Credit_Report.pdf", mime="application/pdf")

    # 📊 Final Analysis Section (persistent)
    st.markdown("### 🔍 See Full AI Analysis")

    if "final_summary" not in st.session_state:
        st.session_state.final_summary = None

    if st.button("Final Analysis"):
        combined_text = f"{insights}\n\nAdvisor Note: {user_note}"
        summarized = summarizer(combined_text, max_length=150, min_length=50, do_sample=False)
        st.session_state.final_summary = summarized[0]['summary_text']

    if st.session_state.final_summary:
        st.subheader("📑 Final Analysis Summary")
        st.write(st.session_state.final_summary)

except Exception as e:
    st.error(f"❌ Error loading credit data: {e}")

st.markdown("---")
st.markdown("<div style='text-align:center; color: gray;'>Your Financial Story — Designed & Delivered by <b>FinTalk Pro</b>.</div>", unsafe_allow_html=True)
