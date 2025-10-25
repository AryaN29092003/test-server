from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta
from supabase import create_client, Client
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# ---------- Load environment ----------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# print(os.getenv("SUPABASE_URL"))
# print(os.getenv("SUPABASE_KEY"))
# print(os.getenv("JWT_SECRET"))
# print(os.getenv("JWT_ALGORITHM"))

app = FastAPI(title="FastAPI + Supabase Auth System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        #"http://localhost:3000",
       "http://192.168.29.97:3000",
        # For testing on same machine
        "http://192.168.68.66:3000"         # System 2's IP (React frontend)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class SignupModel(BaseModel):
    name: str
    email: EmailStr
    password: str

class LoginModel(BaseModel):
    email: EmailStr
    password: str

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)

def create_access_token(data: dict, expires_minutes: int = 60):
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
    data.update({"exp": expire})
    return jwt.encode(data, JWT_SECRET, algorithm=JWT_ALGORITHM)

@app.post("/signup")
def signup(user: SignupModel):
    # 1️⃣ Check if email already exists
    existing = supabase.table("users").select("*").eq("email", user.email).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="Email already registered")

    # 2️⃣ Hash the password
    hashed_pw = hash_password(user.password)

    # 3️⃣ Insert into Supabase
    new_user = {
        "name": user.name,
        "email": user.email,
        "password_hash": hashed_pw,
    }

    result = supabase.table("users").insert(new_user).execute()
    if not result.data:
        raise HTTPException(status_code=500, detail="Error creating user")

    return {"message": "Signup successful", "user_id": result.data[0]["id"]}

@app.post("/login")
def login(credentials: LoginModel):
    # 1️⃣ Find user by email
    user_data = supabase.table("users").select("*").eq("email", credentials.email).execute()
    if not user_data.data:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    user = user_data.data[0]

    # 2️⃣ Verify password
    if not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    # 3️⃣ Generate token
    token = create_access_token({"sub": user["email"], "id": user["id"]})
    return {"access_token": token, "token_type": "bearer"}

if __name__ == "__main__":
    # Allow running `python main.py` for quick local testing
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)