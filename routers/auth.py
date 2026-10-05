import os
import re
from datetime import datetime, timedelta, timezone
from random import randint
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

from utils.auth_utils import send_otp

load_dotenv()

router = APIRouter( tags=["Auth"])
security = HTTPBearer()

MONGODB_URI = os.getenv("MONGO_URI")
mongo_client = AsyncIOMotorClient(MONGODB_URI)
db = mongo_client["auth_db"]

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer()

class SignUpRequest(BaseModel):
    email: EmailStr
    username: str
    password: str

class OTPRequest(BaseModel):
    email: EmailStr

class OTPVerifyRequest(BaseModel):
    otp: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# Utility functions
def hash_password(password: str) -> str:
    return pwd_ctx.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)

def validate_password(password: str, email: str, username: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if email.lower() == password.lower() or username.lower() == password.lower():
        return False, "Password cannot be the same as your email or username."
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter."
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit."
    if not re.search(r'[!@#$%^&*(),.?":{}|<>_\-+=~`\\/\[\]]', password):
        return False, "Password must contain at least one special character."
    return True, "Password is valid."

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    token = creds.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise JWTError()
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await db.users.find_one({"email": email})
    if not user or not user.get("is_verified"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials or email not verified",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


# Routes
@router.post("/sign-up", summary="Create new user")
async def sign_up(data: SignUpRequest):
    email = data.email.strip().lower()
    username = data.username.strip()

    if await db.users.find_one({"email": email}):
        raise HTTPException(status_code=409, detail="Email already registered")

    if await db.users.find_one({"username": username}):
        raise HTTPException(status_code=409, detail="Username already taken")

    is_valid, reason = validate_password(data.password, email, username)
    if not is_valid:
        raise HTTPException(status_code=400, detail=reason)

    user = {
        "email": email,
        "username": username,
        "hashed_password": hash_password(data.password),
        "is_verified": False,
    }

    result = await db.users.insert_one(user)
    if not result.inserted_id:
        raise HTTPException(status_code=500, detail="Failed to create user")

    return {"msg": "User created, please request OTP to verify"}


@router.post("/request-otp", summary="Send OTP to user's email")
async def request_otp(data: OTPRequest):
    email = data.email.strip().lower()

    send_result = await send_otp(email)
    if not send_result["status"]:
        raise HTTPException(500, detail=f"Failed to send OTP: {send_result['error']}")

    otp_code = send_result["otp"]
    expires_at = (datetime.utcnow() + timedelta(minutes=5)).isoformat()

    otp_record = {
        "email": email,
        "otp": otp_code,
        "expires_at": expires_at,
    }

    result = await db.otps.insert_one(otp_record)
    if not result.inserted_id:
        raise HTTPException(500, detail="OTP insert failed")

    return {"msg": f"OTP sent to {email}"}


@router.post("/verify-otp", summary="Verify OTP and mark user verified")
async def verify_otp(data: OTPVerifyRequest):
    otp = data.otp.strip()

    otp_doc = await db.otps.find_one({"otp": otp}, sort=[("expires_at", -1)])
    if not otp_doc:
        raise HTTPException(404, detail="OTP not found")

    if datetime.utcnow().isoformat() > otp_doc["expires_at"]:
        raise HTTPException(400, detail="OTP expired")

    result = await db.users.update_one(
        {"email": otp_doc["email"]}, {"$set": {"is_verified": True}}
    )
    if result.modified_count == 0:
        raise HTTPException(500, detail="Failed to verify user")

    return {"msg": f"Email {otp_doc['email']} verified. You can now log in."}


@router.post("/login", response_model=TokenResponse, summary="Login and receive JWT")
async def login(data: LoginRequest):
    email = data.email.strip().lower()
    password = data.password.strip()

    user = await db.users.find_one({"email": email})
    if not user or not verify_password(password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not user.get("is_verified"):
        raise HTTPException(status_code=403, detail="Email not verified")

    token_doc = await db.user_tokens.find_one({"email": email})
    if token_doc:
        expires_at = datetime.fromisoformat(token_doc["expires_at"]).replace(tzinfo=timezone.utc)
        if expires_at > datetime.now(timezone.utc):
            return {"access_token": token_doc["token"]}

    expires_in = timedelta(days=7)
    token = create_access_token(data={"sub": user["email"]}, expires_delta=expires_in)
    expires_at_str = (datetime.utcnow() + expires_in).isoformat()

    token_record = {
        "email": email,
        "token": token,
        "expires_at": expires_at_str,
    }

    await db.user_tokens.update_one(
        {"email": email}, {"$set": token_record}, upsert=True
    )

    return {"access_token": token}
