import smtplib
from email.mime.text import MIMEText
import streamlit as st
from config import EMAIL_CONFIG, EMAIL_SUBJECT, DEADLINE, CODECHEF_LINK

def send_shortlist_email(to_email, dataframe):
    """Send shortlist email to candidate"""
    sender_email = EMAIL_CONFIG["sender_email"]
    app_password = EMAIL_CONFIG["app_password"]
    
    body = f"""You have been Shortlisted!

Please complete the coding challenge by {DEADLINE}

{CODECHEF_LINK}
"""

    msg = MIMEText(body)
    msg["Subject"] = EMAIL_SUBJECT
    msg["From"] = sender_email
    msg["To"] = to_email

    try:
        server = smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"])
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
        st.success(" Email sent successfully!")
        return True
    except Exception as e:
        st.error(f" Email failed: {str(e)}")
        return False
