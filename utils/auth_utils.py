import math
import random
import re
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Any
from dotenv import load_dotenv
import jwt
from datetime import datetime, timedelta
import hashlib

load_dotenv(override=True)

print(os.getenv("EMAIL_ADDRESS"))
print(os.getenv("EMAIL_PASSWORD"))
print(os.getenv("JWT_SECRET"))


SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"

OTP_HTML_TEMPLATE = """
<html>
<head>
    <style>
    body {{ background-color: #ffffff; font-family: Arial, sans-serif; color: #333333; margin: 0; padding: 20px; }}
    .container {{ max-width: 600px; margin: auto; border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; background-color: #fafafa; }}
    .header {{ text-align: center; color: #000000; }}
    .otp {{ text-align: center; margin: 20px 0; }}
    .otp-code {{ display: inline-block; padding: 10px 20px; border: 2px dashed #808080; font-size: 24px; color: #000000; letter-spacing: 4px; }}
    .footer {{ text-align: center; color: #808080; }}
    </style>
</head>
<body>
    <div class="container">
    <h2 class="header">Welcome to MirrorMe</h2>
    <p style="text-align: center; font-size: 18px;">Your One-Time Password (OTP) is:</p>
    <div class="otp">
        <span class="otp-code">{otp}</span>
    </div>
    <p class="footer">Please use this OTP to complete your verification. The OTP is valid for only 5 minutes.</p>
    <p style="color: #000000;">Cheers,<br>The MirrorMe Team</p>
    </div>
</body>
</html>
"""


async def generateOTP() -> str:
    return ''.join(str(random.randint(0, 9)) for _ in range(6))


async def check_email_format(email: str) -> bool:
    pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    return bool(re.match(pattern, email, re.IGNORECASE))


async def send_otp(recipient_email: str) -> dict[str, Any]:
    """
    Sends an email using Gmail SMTP with HTML template for the OTP.
    """
    if not await check_email_format(recipient_email):
        return {"status": False, "error": "Invalid email address format"}

    otp = await generateOTP()
    html_message = OTP_HTML_TEMPLATE.format(otp=otp)

    msg = MIMEMultipart("alternative")
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = recipient_email
    msg["Subject"] = "MirrorMe OTP"
    msg.attach(MIMEText(html_message, "html"))

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, recipient_email, msg.as_string())
        return {"status": True, "otp": otp}
    except smtplib.SMTPAuthenticationError as e:
        return {"status": False, "error": f"Authentication error: {e}"}
    except smtplib.SMTPRecipientsRefused as e:
        return {"status": False, "error": f"Recipient refused: {e}"}
    except Exception as e:
        return {"status": False, "error": f"SMTP error: {e}"}


async def create_jwt_token(data: dict[str, Any]) -> str:
    email = data["email"]
    password = data.get("password", "")
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    payload = {
        "user_id": data.get("user_id"),
        "email": email,
        "password_hash": password_hash,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def verify_jwt_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise Exception("Token has expired")
    except jwt.InvalidTokenError:
        raise Exception("Invalid token")
