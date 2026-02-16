import smtplib
from email.mime.text import MIMEText
import streamlit as st
import pandas as pd
from Models.GradientBoosting_CodeForces import get_gradientboosting_codeforces_table
from Models.RandomForest_Github import get_randomforest_github_table
from Models.DecisionTree_Github import get_decisiontree_github_table
from Models.RandomForest_CodeForces import get_randomforest_codeforces_table



def send_email_to_hr(to_email, dataframe):
    sender_email = "simran.amesar@yahoo.in"
    app_password = "sehgqqiblmirfbhk"
    subject = "You have been Shortlisted!"
    body = """You have been Shortlisted!
    
    Please complete the coding challenge by 20 February, 2026 17:00
    
    https://www.codechef.com/practice/course/python/LPPYAS01/problems/LPYAS01
    """

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = to_email

    try:
        server = smtplib.SMTP("smtp.mail.yahoo.com", 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print("Email failed:", e)

# Layout
st.set_page_config(page_title="Hiring Intelligence Dashboard", layout="wide")

st.title("Hiring Intelligence Dashboard")

persona = st.sidebar.selectbox(
    "Select persona",
    ["Manager view", "HR view", "Candidate view"],
)

st.sidebar.markdown("---")
st.sidebar.write("Built by Simran, Ishwari & Sakshi")

if persona == "Manager view":
    st.header("Manager View – Model Insights")

    tab1, tab2, tab3, tab4 = st.tabs([
        "RandomForest (GitHub)",
        "GradientBoosting (Codeforces)",
        "DecisionTree (GitHub)",
        "RandomForest (Codeforces)"
    ])

    with tab1:
        rf_table = get_randomforest_github_table()
        st.dataframe(rf_table, use_container_width=True)

    with tab2:
        gb_table = get_gradientboosting_codeforces_table()
        st.dataframe(gb_table, use_container_width=True)

    with tab3:
        dt_table = get_decisiontree_github_table()
        st.dataframe(dt_table, use_container_width=True)

    with tab4:
        cf_table = get_randomforest_codeforces_table()
        st.dataframe(cf_table, use_container_width=True)


elif persona == "HR view":
    st.header("HR View – Final Shortlist")

    # Load tables
    rf_table = get_randomforest_github_table()
    gb_table = get_gradientboosting_codeforces_table()
    dt_table = get_decisiontree_github_table()
    cf_table = get_randomforest_codeforces_table()

    # Combine
    combined = pd.concat(
        [rf_table, gb_table, dt_table, cf_table],
        ignore_index=True
    )

    combined["Candidate"] = (
        combined["user_name"]
        .fillna(combined["username"])
        .fillna(combined["github_username"])
    )

    combined["Stage"] = "Round 2 – Technical Interview"

    final_shortlist = combined[["Candidate", "Stage"]].drop_duplicates()

    st.subheader("Shortlisted Candidates")
    st.dataframe(final_shortlist, use_container_width=True)


    default_email = "Simran.Amesar@stud.srh-university.de"
    hr_email = st.text_input(
        "Enter Candidate Email Address",
        value=default_email
    )

    if st.button("Send shortlist email to Candidate"):
        if hr_email:
            send_email_to_hr(hr_email, final_shortlist)
            st.success("Email sent successfully.")
        else:
            st.error("Please enter an email address.")


elif persona == "Candidate view":

    st.header("Candidate Dashboard")

    col1, col2 = st.columns(2)
    with col1:
        gh_username = st.text_input("Username :")

    if st.button("Check my status"):

        if not gh_username:
            st.error("Please enter a username to check status.")
        else:
            rf_table = get_randomforest_github_table()
            gb_table = get_gradientboosting_codeforces_table()
            dt_table = get_decisiontree_github_table()
            cf_table = get_randomforest_codeforces_table()

            all_tables = [rf_table, gb_table, dt_table, cf_table]
            combined_df = pd.concat(all_tables, ignore_index=True)

            combined_df["name"] = combined_df[["user_name", "username", "github_username"]].bfill(axis=1).iloc[:, 0]
            combined_df["name"] = combined_df["name"].astype(str)

            candidate_row = combined_df[combined_df["name"].str.lower() == gh_username.lower()]

            if candidate_row.empty:
                st.error("No candidate found in any table.")
            else:
                candidate_row = candidate_row.iloc[0]

                assessment_1_status = "Done"
                assessment_2_status = "Submitted"
                shortlisted_status = True  # change to False if needed

                tab1, tab2, tab3 = st.tabs(["User Information", "Assessment", "Status"])

                with tab1:
                    st.subheader("User Information")
                    candidate_df = candidate_row.to_frame().T
                    candidate_df = candidate_df.dropna(axis=1, how="all")

                    for col in candidate_df.columns:
                        if pd.api.types.is_float_dtype(candidate_df[col]):
                            if (candidate_df[col] % 1 == 0).all():
                                candidate_df[col] = candidate_df[col].astype(int)

                    st.dataframe(candidate_df, use_container_width=True)

                with tab2:
                    st.subheader("Assessment Status")
                    st.write("Assessment 1:", assessment_1_status)
                    st.write("Assessment 2:", assessment_2_status)

                    if assessment_1_status == "Done" and assessment_2_status == "Done":
                        st.success("All assessments completed ")
                    elif assessment_1_status == "Done" or assessment_2_status == "Done":
                        st.info("One assessment completed.")
                    else:
                        st.warning("Assessments not submitted.")

                with tab3:
                    st.subheader("Application Status")

                    if shortlisted_status:
                        st.success("You have been shortlisted!")
                    else:
                        st.error("Not shortlisted yet.")

                    st.caption("Final decision may include manual review and interviews.")

