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
from fact_checker_verifylens import verify_claim
from datetime import datetime, date


# Suppress pydub warnings
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

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
            "claims": ["/extract-claims", "/get-claims"],
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
    #determine if last accessed today
    last_accessed = user["last_accessed"]

    # Convert to python datetime if Supabase returns string
    if isinstance(last_accessed, str):
        last_accessed = datetime.fromisoformat(last_accessed.replace("Z", ""))

    today = date.today()
    last_accessed_date = last_accessed.date()

    # Determine new credits
    new_credits = user["credits"]

    if last_accessed_date != today:
        # Reset credits based on acc_type
        if user["acc_type"] == "premium":
            new_credits = 1000
        else:
            new_credits = 10
        
        # Update in database
        supabase.table("users").update({
            "credits": new_credits,
            "last_accessed": datetime.utcnow().isoformat()
        }).eq("id", user["id"]).execute()

    else:
        # Only update last_accessed timestamp
        supabase.table("users").update({
            "last_accessed": datetime.utcnow().isoformat()
        }).eq("id", user["id"]).execute()
        
    # 3️⃣ Generate token
    token = create_access_token({"sub": user["email"], "id": user["id"]})
    
    return {
        "access_token": token, 
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "acc_type": user["acc_type"],
            "last_accessed": user["last_accessed"]
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
    Extract factual claims from text using Groq AI and fact-check them
    """
    try:
        text = request.text
        user_id = request.user_id
        
        if not text or text.strip() == "":
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        print(f"\n{'='*70}")
        print(f"📝 INPUT TEXT:")
        print(f"{'='*70}")
        print(text)
        print(f"{'='*70}\n")
        #fetch user record
        user_data = supabase.table("users").select("*").eq("id", user_id).single().execute()

        if user_data.data is None:
            raise HTTPException(
                status_code=404,
                detail="User not found"
            )
        #check if credits available
        credits = user_data.data["credits"]
        if credits==0:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough credits left. Please come back tomorrow."
            )
        # Extract claims
        claims = claim_extractor.extract_claims(text)
        
        # Process each claim
        if claims:
            print(f"\n{'='*70}")
            print(f"✅ EXTRACTED {len(claims)} CLAIM(S):")
            print(f"{'='*70}")
            
            stored_claims = []
            
            for i, claim in enumerate(claims, 1):
                print(f"\n{i}. {claim}")
                
                try:
                    # ============================================
                    # STEP 1: Store claim in database
                    # ============================================
                    claim_data = {
                        "user_id": user_id,
                        "claim_text": claim,
                        "original_text": text,
                        "status": "pending"
                    }
                    
                    claim_response = supabase.table("Claims").insert(claim_data).execute()
                    
                    if not claim_response.data:
                        print(f"❌ Failed to store claim: {claim}")
                        continue
                    
                    claims_id = claim_response.data[0]["claims_id"]
                    print(f"✓ Claim stored with ID: {claims_id}")
                    
                    # ============================================
                    # STEP 2: Fact-check the claim
                    # ============================================
                    print(f"🔍 Fact-checking claim: '{claim}'")
                    
                    fact = verify_claim(claim)
                    
                    print(f"✓ Fact-check result: {fact.get('verdict')} (confidence: {fact.get('confidence')}%)")
                    
                    # ============================================
                    # STEP 3: Store fact-check result
                    # ============================================
                    # CRITICAL FIX: Use claims_id for BOTH fact_id AND claim_id
                    fact_data = {
                        "claim_id": claims_id,  # Foreign key to Claims table
                        "verdict": fact.get("verdict", "unverified"),
                        "confidence": fact.get("confidence", 0),
                        "explanation": fact.get("explanation", ""),
                        "citations": fact.get("citations", [])
                    }
                    
                    fact_response = supabase.table("Fact_checker").insert(fact_data).execute()
                    
                    if not fact_response.data:
                        print(f"❌ Failed to store fact-check for claim ID: {claims_id}")
                    else:
                        print(f"✓ Fact-check stored for claim ID: {claims_id}")
                        stored_claims.append({
                            "claim_id": claims_id,
                            "claim_text": claim,
                            "fact_check": fact
                        })
                    
                except Exception as e:
                    print(f"❌ Error processing claim '{claim}': {str(e)}")
                    continue
            
            print(f"\n{'='*70}")
            print(f"✅ Successfully processed {len(stored_claims)}/{len(claims)} claims")
            print(f"{'='*70}\n")

            #deduct credits from user
            supabase.table("users").update({"credits": credits - 1}).eq("id", user_id).execute()
            
            return {
                "success": True,
                "claims": claims,
                "count": len(claims),
                "stored_claims": stored_claims,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            print(f"\n⚠️  No claims extracted from the text\n")
            return {
                "success": True,
                "claims": [],
                "count": 0,
                "timestamp": datetime.utcnow().isoformat()
            }
        
    except Exception as e:
        print(f"❌ Claim extraction error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error extracting claims: {str(e)}"
        )


# ==================== GET CLAIMS WITH FACT-CHECKS ====================
@app.get("/get-claims")
async def get_claims(user_id: str = "2"):
    """
    Retrieve all claims for a user along with their fact-check results
    """
    try:
        # Fetch all claims for the user
        claims_response = supabase.table("Claims").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        
        if not claims_response.data:
            return {
                "success": True,
                "claims": [],
                "count": 0
            }
        
        claims = claims_response.data
        
        # For each claim, fetch the corresponding fact-check
        claims_with_facts = []
        for claim in claims:
            claim_id = claim["claims_id"]
            
            # Fetch fact-check data using claim_id (not fact_id!)
            fact_response = supabase.table("Fact_checker").select("*").eq("claim_id", claim_id).execute()
            
            claim_data = {
                "claims_id": claim["claims_id"],
                "user_id": claim["user_id"],
                "claim_text": claim["claim_text"],
                "original_text": claim["original_text"],
                "status": claim["status"],
                "created_at": claim["created_at"]
            }
            
            # Add fact-check data if available
            if fact_response.data and len(fact_response.data) > 0:
                fact = fact_response.data[0]
                claim_data["factCheck"] = {
                    "fact_id": fact["fact_id"],
                    "verdict": fact["verdict"],
                    "confidence": fact["confidence"],
                    "explanation": fact["explanation"],
                    "citations": fact.get("citations", [])
                }
            
            claims_with_facts.append(claim_data)
        
        print(f"✅ Retrieved {len(claims_with_facts)} claims for user {user_id}")
        
        return {
            "success": True,
            "claims": claims_with_facts,
            "count": len(claims_with_facts)
        }
        
    except Exception as e:
        print(f"❌ Error fetching claims: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error fetching claims: {str(e)}"
        )


@app.post("/store-claims")
def create_claim(request: StoreClaimRequest):
    """
    Manually store a claim (deprecated - use /extract-claims instead)
    """
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

# ==================== STATS ENDPOINT (OPTIONAL) ====================
@app.get("/fact-checker-stats")
def get_fact_checker_stats():
    """
    Get statistics about RAG vs Perplexity usage
    """
    try:
        from fact_checker_verifylens import get_fact_checker
        checker = get_fact_checker()
        stats = checker.get_stats()
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# ==================== RUN SERVER ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)





