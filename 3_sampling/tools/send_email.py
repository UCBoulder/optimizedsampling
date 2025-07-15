import os
import sys
import yagmail

def send_email(subject, body, to_email):
    gmail_user = "liviabetti1@gmail.com"
    gmail_password = os.getenv("YAGMAIL_PASSWORD")
    if not gmail_password:
        print("❌ ERROR: Environment variable YAGMAIL_PASSWORD is not set.")
        sys.exit(1)

    try:
        yag = yagmail.SMTP(gmail_user, password=gmail_password)
        yag.send(to=to_email, subject=subject, contents=body)
        print(f"✅ Email sent to {to_email}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python send_email.py 'Subject' 'Body' 'liviabetti1@gmail.com'")
        sys.exit(1)

    _, subject, body, to_email = sys.argv
    send_email(subject, body, to_email)
