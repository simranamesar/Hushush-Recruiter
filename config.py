import streamlit as st

EMAIL_CONFIG = st.secrets["EMAIL_CONFIG"]

EMAIL_SENDER = EMAIL_CONFIG.get("sender_email")
EMAIL_PASSWORD = EMAIL_CONFIG.get("app_password")
SMTP_SERVER = EMAIL_CONFIG.get("smtp_server")
SMTP_PORT = EMAIL_CONFIG.get("smtp_port")

EMAIL_SUBJECT = "You have been Shortlisted!"
DEADLINE = st.secrets.get("DEADLINE", "20 February, 2026 17:00")
CODECHEF_LINK = st.secrets.get(
    "CODECHEF_LINK",
    "https://www.codechef.com/practice/course/python/LPPYAS01/problems/LPYAS01"
)
