"""
Simplified Claim Extractor - Groq API Version
Extracts only the factual claims from text
"""

from groq import Groq
import json
from typing import List


class ClaimExtractor:
    """
    Extract factual claims from text using Groq API (hosted Llama 3.1)
    """
    
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        """
        Initialize the claim extractor
        
        Args:
            api_key: Your Groq API key
            model: Model to use (default: llama-3.1-8b-instant)
        """
        self.client = Groq(api_key=api_key)
        self.model = model
    
    def extract_claims(self, text: str) -> list:
        """
        Extract factual claims from text
        
        Args:
            text: Input text to extract claims from
            
        Returns:
            List of claim strings
        """
        prompt = self._build_prompt(text)
        
        try:
            response = self._call_groq(prompt)
            claims = self._parse_response(response)
            return claims
            
        except Exception as e:
            print(f"Error extracting claims: {e}")
            return []
    
    def _build_prompt(self, text: str) -> str:
        """Build the prompt for claim extraction"""
        return f"""Extract all factual claims from the following text.

A factual claim is a statement that can be verified as true or false.

Rules:
- Extract only verifiable facts, not opinions or predictions
- Each claim should be standalone and complete
- Replace pronouns with actual names when possible (e.g., "he" → "John Smith")
- Keep claims concise but complete
- Don't add any interpretation or context

Text:
{text}

Return your response as a JSON array of strings with this exact structure:
[
  "First factual claim",
  "Second factual claim",
  "Third factual claim"
]

Return ONLY the JSON array, nothing else."""
    
    def _call_groq(self, prompt: str) -> str:
        """Call Groq API"""
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at extracting factual claims from text. You always return valid JSON arrays of strings."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.3,
                max_tokens=2000,
            )
            
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Groq API error: {str(e)}")
    
    def _parse_response(self, response: str) -> list:
        """Parse JSON response from LLM"""
        try:
            # Extract JSON array from response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx == -1 or end_idx <= start_idx:
                print("Warning: Could not find JSON array in response")
                return []
            
            json_str = response[start_idx:end_idx]
            claims = json.loads(json_str)
            
            # Ensure we have a list of strings
            if isinstance(claims, list):
                return [str(claim) for claim in claims]
            else:
                return []
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"Raw response: {response[:200]}...")
            return []
