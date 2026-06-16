import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- Email config ---
sender_email = "traffiqph@gmail.com"
receiver_email = "johnlouis.lagramada@tup.edu.ph"
password = "uvml fcrn zbpy mtzd"  # not your Gmail password!

# --- Create the email ---
msg = MIMEMultipart()
msg["From"] = sender_email
msg["To"] = receiver_email
msg["Subject"] = "Test Email from Python"

body = "Hello, this is a test email sent from Python!"
msg.attach(MIMEText(body, "plain"))

# --- Send the email ---
try:
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()  # secure connection
        server.login(sender_email, password)
        server.send_message(msg)
        print("✅ Email sent successfully!")
except Exception as e:
    print("❌ Error:", e)
