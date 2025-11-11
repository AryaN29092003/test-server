import os
import re
import json
import requests
from typing import Dict, Any, List
from dotenv import load_dotenv

PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

load_dotenv()

def verify_claim_with_perplexity(claim_text: str) -> Dict[str, Any]:
    """
    Verify a factual claim using Perplexity AI API.
    Returns structured dict with verdict, confidence, citations, and explanation.
    """

    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        return {
            "verdict": "unverified",
            "confidence": 50,
            "citations": [
                {
                    "title": "Missing API Key",
                    "url": "https://www.perplexity.ai/api-platform",
                    "snippet": "Set the PERPLEXITY_API_KEY environment variable."
                }
            ],
            "explanation": "Perplexity API key not configured."
        }
    prompt = f"""You are a fact-checking AI. Analyze the given claim and return the result strictly as a valid JSON object, with no text, markdown, or explanations outside it.

Claim:
"{claim_text}"

Return the result ONLY in this exact JSON schema (no extra fields, no nested JSON strings, all keys in double quotes):

{{"verdict": "true|false|misleading|partially-true|unverified","confidence": 85,"explanation": "Brief explanation of why the verdict was given and how the confidence score was inferred.","citations": [{{"title": "string","url": "string","snippet": "string"}},{{"title": "string","url": "string","snippet": "string"}},...]}}

Output Rules:
1. Use only double quotes for all keys and string values.
2. Do not include any markdown, comments, or additional text.
3. The final output must be a single valid JSON object matching the schema above exactly.
4. Do NOT embed another JSON object inside the explanation or any field.
5. The output must be directly parseable by a JSON parser without modification.
"""


    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "You are a fact-checking assistant. Analyze claims and provide accurate verdicts with citations from reliable sources. Always return valid JSON in your response."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300,
        "temperature": 0.2,
        "return_citations": True,
        "return_related_questions": False
    }

    try:
        response = requests.post(PERPLEXITY_API_URL, headers=headers, json=payload, timeout=30)
    except requests.RequestException as e:
        return {
            "verdict": "unverified",
            "confidence": 0,
            "citations": [],
            "explanation": f"Network error: {e}"
        }

    if response.status_code == 429:
        return {"verdict": "unverified", "confidence": 0, "citations": [], "explanation": "Rate limit exceeded (429)."}
    elif response.status_code == 401:
        return {"verdict": "unverified", "confidence": 0, "citations": [], "explanation": "Invalid API key (401)."}
    elif not response.ok:
        return {"verdict": "unverified", "confidence": 0, "citations": [], "explanation": f"HTTP error {response.status_code}: {response.text[:200]}"}
    else:
        try:
            data= response.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", ""))
            if content[0:7]=="```json":
                content = content[7:-3]
            return json.loads(content)
        except json.JSONDecodeError:
            return {"verdict": "unverified", "confidence": 0, "citations": [], "explanation": "Invalid JSON in response."}
          
if __name__ == "__main__":
    claim = input("Enter a claim to fact-check: ").strip()
    result = verify_claim_with_perplexity(claim)
    print(json.dumps(result, indent=2))
