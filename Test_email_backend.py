import streamlit as st
from email_backend import get_email_service
import os

# Initialize the backend
email_service = get_email_service()

# -------------------------------
# Session State Initialization
# -------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Login"

if "user" not in st.session_state:
    st.session_state.user = None

if "pending_email" not in st.session_state:
    st.session_state.pending_email = None

def switch_page(page):
    st.session_state.page = page

# -------------------------------
# Page: Register
# -------------------------------
def register_page():
    st.title("🔐 Register")

    with st.form("register_form"):
        first_name = st.text_input("First Name")
        last_name = st.text_input("Last Name")
        email = st.text_input("Email")
        phone = st.text_input("Phone Number (Optional)")
        password = st.text_input("Password", type="password")
        confirm = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Register")

        if submit:
            data = {
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "phone_number": phone,
                "password": password,
                "confirm_password": confirm
            }
            result = email_service.register_user(data)
            st.toast(result["popup_toast"], icon="🔔")
            if result.get("success"):
                st.session_state.pending_email = email
                switch_page("Verify OTP")
            elif result.get("errors"):
                st.error(result["errors"])
            if result.get("warnings"):
                st.warning(result["warnings"])

# -------------------------------
# Page: OTP Verification
# -------------------------------
def otp_page():
    st.title("📧 Verify Email")
    email = st.session_state.pending_email or st.text_input("Email")
    otp_code = st.text_input("Enter OTP Code")
    verify = st.button("Verify")

    if verify:
        result = email_service.verify_otp(email, otp_code)
        st.toast(result["popup_toast"])
        if result.get("success"):
            switch_page("Login")
        else:
            st.error(result["error"])

    if st.button("Resend OTP"):
        resend = email_service.resend_otp(email)
        st.toast(resend["popup_toast"])

# -------------------------------
# Page: Login
# -------------------------------
def login_page():
    st.title("🔓 Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    login = st.button("Login")

    if login:
        result = email_service.login_user(email, password)
        st.toast(result["popup_toast"])
        if result.get("success"):
            st.session_state.user = result["user"]
            switch_page("Home")
        elif result.get("needs_verification"):
            st.session_state.pending_email = email
            switch_page("Verify OTP")
        else:
            st.error(result["error"])

# -------------------------------
# Page: Home + Email Sender
# -------------------------------
def home_page():
    st.title("🏠 Welcome to SecureMail Pro")

    user = st.session_state.user
    st.success(f"Hello {user['first_name']} {user['last_name']} ({user['email']})")

    with st.expander("📧 Send Email"):
        with st.form("send_email_form"):
            to = st.text_input("To (comma-separated)").split(",")
            cc = st.text_input("CC (comma-separated)").split(",")
            bcc = st.text_input("BCC (comma-separated)").split(",")
            subject = st.text_input("Subject")
            body = st.text_area("Email Body (HTML allowed)")
            attachments = st.file_uploader(
                "Attachments (Optional)",
                type=["pdf", "txt", "docx", "png", "jpg"],
                accept_multiple_files=True
            )
            submit = st.form_submit_button("Send Email")

            if submit:
                login_creds = {
                    "email": os.getenv("SYSTEM_EMAIL"),
                    "password": os.getenv("SYSTEM_EMAIL_PASSWORD"),
                    "smtp_server": os.getenv("SYSTEM_SMTP_SERVER", "smtp.gmail.com"),
                    "smtp_port": os.getenv("SYSTEM_SMTP_PORT", "587")
                }

                attachment_paths = []
                if attachments:
                    for file in attachments:
                        save_path = file.name
                        with open(save_path, "wb") as f:
                            f.write(file.read())
                        attachment_paths.append(save_path)

                result = email_service.send_email(
                    login_credentials=login_creds,
                    to=[t.strip() for t in to if t.strip()],
                    cc=[c.strip() for c in cc if c.strip()],
                    bcc=[b.strip() for b in bcc if b.strip()],
                    subject=subject,
                    body=body,
                    attachments=attachment_paths if attachment_paths else None
                )
                st.toast(result["popup_toast"])
                if not result.get("success"):
                    st.error(result.get("error"))

    if st.button("🚪 Logout"):
        st.session_state.user = None
        switch_page("Login")

# -------------------------------
# Sidebar + Navigation
# -------------------------------
st.sidebar.title("🔄 Navigation")

if st.session_state.user:
    st.sidebar.success(f"Logged in as {st.session_state.user['first_name']}")
    nav = st.sidebar.radio("Go to", ["Home", "Logout"])
    if nav == "Home":
        switch_page("Home")
    elif nav == "Logout":
        st.session_state.user = None
        switch_page("Login")
else:
    nav = st.sidebar.radio("Go to", ["Login", "Register", "Verify OTP"])
    switch_page(nav)

# -------------------------------
# Router
# -------------------------------
if st.session_state.page == "Login":
    login_page()
elif st.session_state.page == "Register":
    register_page()
elif st.session_state.page == "Verify OTP":
    otp_page()
elif st.session_state.page == "Home":
    home_page()
