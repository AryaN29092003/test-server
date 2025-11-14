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
from claim_extractor_groq import ClaimExtractor
from fact_check import verify_claim_with_perplexity
# ---------- Load environment ----------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
claim_extractor = ClaimExtractor(api_key=GROQ_API_KEY)

app = FastAPI(title="VerifyLens Backend - Auth + Speech-to-Text + Claim Extraction")

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
    
class StoreClaimRequest(BaseModel):
    user_id: int
    claim_text: str
    original_text: str
    status: str

class ClaimRequest(BaseModel):
    text: str = ""
    user_id: str = "2"
    
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
            "claims": ["/extract-claims"],
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

# ==================== CLAIM EXTRACTION ENDPOINT ====================
@app.post("/extract-claims")
async def extract_claims(request: ClaimRequest):
    """
    Extract factual claims from text using Groq AI
    """
    try:
        text = request.text
        user = request.user_id
        
        if not text or text.strip() == "":
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        print(f"\n{'='*70}")
        print(f"📝 INPUT TEXT:")
        print(f"{'='*70}")
        print(text)
        print(f"{'='*70}\n")
        
        # Extract claims
        claims = claim_extractor.extract_claims(text)
        
        # Print claims to console
        if claims:
            print(f"\n{'='*70}")
            print(f"✅ EXTRACTED {len(claims)} CLAIM(S):")
            print(f"{'='*70}")
            for i, claim in enumerate(claims, 1):
                print(f"{i}. {claim}")
                # try and except block for store-claim logic
                try:
                    # Insert into Supabase table
                    data = {
                        "user_id": user,
                        "claim_text": claim,
                        "original_text": text,
                        "status": "pending"
                    }
                    response = supabase.table("Claims").insert(data).execute()
    
                    if not response.data:
                        raise HTTPException(status_code=400, detail="Insert failed")
                    else:
                        print("Response from DB after inserting claim: ")
                        print(response.data)
                        print("Inputted claim in DB: "+ claim)
                    # return {"message": "Claim added successfully", "data": response.data[0]}  
                    #Insert clause ends
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
                # fact check module
                fact = verify_claim_with_perplexity(claim)
                verdict = fact["verdict"]
                confidence = fact["confidence"]
                citations = fact["citations"]
                explanation = fact["explanation"]
                # insert checked fact into DB
                try:
                    # Insert into Supabase table
                    data = {
                        "verdict": verdict,
                        "confidence": confidence,
                        "explanation": explanation,
                        "citations": citations
                    }
                    response = supabase.table("Fact_checker").insert(data).execute()
    
                    if not response.data:
                        raise HTTPException(status_code=400, detail="Insert failed")
                    else:
                        print("Inputted fact in DB: ")
                        print(fact)
                    # return {"message": "Claim added successfully", "data": response.data[0]}  
                    #Insert clause ends
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
                    
            print(f"{'='*70}\n")
        else:
            print(f"\n⚠️  No claims extracted from the text\n")
        
        return {
            "success": True,
            "claims": claims,
            "count": len(claims),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Claim extraction error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error extracting claims: {str(e)}"
        )

@app.post("/store-claims")
def create_claim(request: StoreClaimRequest):
    try:
        # Insert into Supabase table
        data = {
            "user_id": request.user_id,
            "claim_text": request.claim_text,
            "original_text": request.original_text,
            "status": request.status
        }
        response = supabase.table("Claims").insert(data).execute()

        if not response.data:
            raise HTTPException(status_code=400, detail="Insert failed")

        return {"message": "Claim added successfully", "data": response.data[0]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
# ==================== RUN SERVER ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)








