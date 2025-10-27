from fastapi import FastAPI, HTTPException, status, UploadFile, File
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta
from supabase import create_client, Client
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import speech_recognition as sr
from pydub import AudioSegment
import tempfile

# ---------- Load environment ----------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI(title="VerifyLens Backend - Auth + Speech-to-Text")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://192.168.29.97:3000",
        "http://192.168.68.66:3000",
        "*"  # For Render deployment - replace with your frontend domain in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== MODELS ====================
class SignupModel(BaseModel):
    name: str
    email: EmailStr
    password: str

class LoginModel(BaseModel):
    email: EmailStr
    password: str

# ==================== AUTH HELPERS ====================
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)

def create_access_token(data: dict, expires_minutes: int = 60):
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
    data.update({"exp": expire})
    return jwt.encode(data, JWT_SECRET, algorithm=JWT_ALGORITHM)

# ==================== AUTH ENDPOINTS ====================
@app.get("/")
def root():
    return {
        "message": "VerifyLens Backend API",
        "status": "running",
        "endpoints": {
            "auth": ["/signup", "/login"],
            "speech": ["/transcribe"],
            "health": ["/health"]
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

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
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Invalid email or password"
        )
    
    user = user_data.data[0]
    
    # 2️⃣ Verify password
    if not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Invalid email or password"
        )
    
    # 3️⃣ Generate token
    token = create_access_token({"sub": user["email"], "id": user["id"]})
    
    return {
        "access_token": token, 
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"]
        }
    }

# ==================== SPEECH-TO-TEXT ENDPOINT ====================
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Converts audio file (webm, mp3, wav, etc.) to text using Google Speech Recognition
    """
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        # Convert to WAV format (required for speech_recognition)
        wav_path = tmp_path.replace(".webm", ".wav")
        AudioSegment.from_file(tmp_path, format="webm").export(
            wav_path, 
            format="wav"
        )
        
        # Perform speech recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        
        # Clean up temporary files
        os.remove(tmp_path)
        os.remove(wav_path)
        
        print(f"✅ Transcribed: {text}")
        
        return {
            "success": True,
            "text": text,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except sr.UnknownValueError:
        return {
            "success": False,
            "error": "Could not understand audio",
            "text": ""
        }
    except sr.RequestError as e:
        return {
            "success": False,
            "error": f"Speech Recognition service error: {e}",
            "text": ""
        }
    except Exception as e:
        print(f"❌ Transcription error: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "text": ""
        }

# ==================== RUN SERVER ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

