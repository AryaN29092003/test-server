import os
from dotenv import load_dotenv
from groq import Groq
from huggingface_hub import InferenceClient  # NEW: Replace SentenceTransformer
import psycopg2
import json
from typing import List, Dict, Tuple
import numpy as np

load_dotenv()

class RAGFactChecker:
    def __init__(self):
        """Initialize RAG Fact Checker with Groq LLM and HuggingFace embeddings"""
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        
        # NEW: Initialize HuggingFace Inference client instead of local model
        print("Initializing HuggingFace API client...")
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable not set. Get it from https://huggingface.co/settings/tokens")
        
        self.hf_client = InferenceClient(api_key=hf_token)
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"✓ Using HuggingFace API for embeddings: {self.embedding_model_name}")
        
        # Connect to Supabase
        print("Connecting to database...")
        self.conn = psycopg2.connect(os.getenv('SUPABASE_CONNECTION_STRING'))
        self.cursor = self.conn.cursor()
        
        print("✓ RAG Fact Checker initialized successfully!")
        print("  Memory saved: ~200MB (using API instead of local model)\n")
    
    def embed_query(self, query: str) -> List[float]:
        """
        Convert query text to embedding vector using HuggingFace API
        
        This replaces the local SentenceTransformer model to save memory.
        Memory saved: ~200-300MB
        """
        try:
            # Call HuggingFace Inference API
            embedding = self.hf_client.feature_extraction(
                query,
                model=self.embedding_model_name
            )
            
            # Normalize the embedding (same as SentenceTransformer did)
            embedding_array = np.array(embedding)
            normalized = embedding_array / np.linalg.norm(embedding_array)
            
            return normalized.tolist()
            
        except Exception as e:
            print(f"⚠️  Error getting embedding from HuggingFace: {e}")
            print("Tip: Check your HF_TOKEN is valid and has not expired")
            raise
    
    def search_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant documents using vector similarity
        
        Args:
            query: User's claim to fact-check
            top_k: Number of relevant documents to retrieve
            
        Returns:
            List of relevant documents with metadata
        """
        print(f"Searching for relevant documents (top {top_k})...")
        
        # Generate query embedding using HuggingFace API
        query_embedding = self.embed_query(query)
        
        # Perform similarity search using cosine distance
        self.cursor.execute("""
            SELECT 
                embedding_id,
                source,
                source_id,
                title,
                content,
                metadata,
                citation,
                period,
                1 - (embedding <=> %s::vector) as similarity
            FROM rag_embeddings
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, query_embedding, top_k))
        
        results = self.cursor.fetchall()
        
        # Format results
        documents = []
        for row in results:
            doc = {
                'embedding_id': row[0],
                'source': row[1],
                'source_id': row[2],
                'title': row[3],
                'content': row[4],
                'metadata': row[5],
                'citation': row[6],
                'period': row[7],
                'similarity': float(row[8])
            }
            documents.append(doc)
            print(f"  [{doc['source']}] {doc['title'][:60]}... (similarity: {doc['similarity']:.3f})")
        
        return documents
    
    def build_context(self, documents: List[Dict]) -> str:
        """Build context string from retrieved documents"""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[Source {i}]")
            context_parts.append(f"Source: {doc['source']}")
            context_parts.append(f"Title: {doc['title']}")
            if doc['period']:
                context_parts.append(f"Period: {doc['period']}")
            context_parts.append(f"Content: {doc['content']}")
            context_parts.append(f"Relevance Score: {doc['similarity']:.3f}")
            context_parts.append("")  # Empty line between sources
        
        return "\n".join(context_parts)
    
    def fact_check_with_llm(self, claim: str, context: str, model: str = "llama-3.3-70b-versatile") -> Dict:
        """
        Use Groq LLM to fact-check the claim based on retrieved context
        
        Available Groq models:
        - llama-3.3-70b-versatile (recommended - fast & accurate)
        - llama-3.1-70b-versatile
        - mixtral-8x7b-32768
        - gemma2-9b-it
        """
        
        print(f"\nFact-checking with {model}...")
        
        # Construct prompt for fact-checking
        prompt = f"""You are a professional fact-checker. Your task is to verify claims using reliable data sources.

CLAIM TO VERIFY:
{claim}

RETRIEVED EVIDENCE FROM TRUSTED SOURCES:
{context}

INSTRUCTIONS:
1. Analyze the claim against the provided evidence
2. Determine if the claim is TRUE, FALSE, PARTIALLY TRUE, or UNVERIFIABLE
3. Provide a confidence score (0-100)
4. Explain your reasoning
5. Cite specific sources that support your verdict

Respond in the following JSON format:
{{
    "verdict": "TRUE/FALSE/PARTIALLY TRUE/UNVERIFIABLE",
    "confidence": 85,
    "explanation": "Detailed explanation of why this verdict was reached",
    "supporting_evidence": ["Evidence point 1", "Evidence point 2"],
    "contradicting_evidence": ["Any contradictions found"],
    "sources_used": ["Source 1", "Source 2"],
    "recommendation": "Brief recommendation for the user"
}}

Provide ONLY the JSON response, no additional text."""

        try:
            # Call Groq API
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional fact-checker that provides accurate, evidence-based verdicts. Always cite your sources and be transparent about uncertainty."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=model,
                temperature=0.1,  # Low temperature for factual accuracy
                max_tokens=2000,
                top_p=0.9
            )
            
            response_text = chat_completion.choices[0].message.content
            
            # Clean and parse JSON response
            response_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            
            print(f"✓ Verdict: {result['verdict']} (Confidence: {result['confidence']}%)")
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"⚠️  Error parsing LLM response: {e}")
            print(f"Raw response: {response_text[:200]}...")
            # Return fallback response
            return {
                "verdict": "UNVERIFIABLE",
                "confidence": 0,
                "explanation": "Error processing fact-check response",
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "sources_used": [],
                "recommendation": "Unable to verify claim due to processing error",
                "raw_response": response_text
            }
        except Exception as e:
            print(f"⚠️  Error calling Groq API: {e}")
            return {
                "verdict": "ERROR",
                "confidence": 0,
                "explanation": str(e),
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "sources_used": [],
                "recommendation": "Error occurred during fact-checking"
            }
    
    def fact_check(self, claim: str, top_k: int = 5, model: str = "llama-3.3-70b-versatile") -> Dict:
        """
        Complete fact-checking pipeline
        
        Args:
            claim: The claim to fact-check
            top_k: Number of relevant documents to retrieve
            model: Groq model to use
            
        Returns:
            Complete fact-check result with verdict and evidence
        """
        print(f"{'='*60}")
        print(f"FACT-CHECKING CLAIM")
        print(f"{'='*60}")
        print(f"Claim: {claim}\n")
        
        # Step 1: Retrieve relevant documents
        documents = self.search_relevant_documents(claim, top_k=top_k)
        
        if not documents:
            return {
                "verdict": "UNVERIFIABLE",
                "confidence": 0,
                "explanation": "No relevant sources found in database",
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "sources_used": [],
                "recommendation": "Unable to verify claim - insufficient data"
            }
        
        # Step 2: Build context from retrieved documents
        context = self.build_context(documents)
        
        # Step 3: Fact-check with LLM
        result = self.fact_check_with_llm(claim, context, model=model)
        
        # Step 4: Add retrieved documents to result
        result['retrieved_documents'] = documents
        result['num_sources_consulted'] = len(documents)
        
        return result
    
    def save_to_database(self, claim: str, user_id: int, fact_check_result: Dict) -> int:
        """
        Save fact-check result to database
        
        Returns:
            fact_id of the saved result
        """
        print("\nSaving to database...")
        
        # Insert claim
        self.cursor.execute("""
            INSERT INTO claims (user_id, claim_text, original_text, status)
            VALUES (%s, %s, %s, 'verified')
            RETURNING claims_id
        """, (user_id, claim, claim))
        
        claim_id = self.cursor.fetchone()[0]
        
        # Prepare citations
        citations = []
        for doc in fact_check_result.get('retrieved_documents', []):
            citations.append({
                'source': doc['source'],
                'title': doc['title'],
                'url': doc['source_id'],
                'similarity': doc['similarity']
            })
        
        # Insert fact check result
        self.cursor.execute("""
            INSERT INTO fact_checker (claim_id, verdict, citations, explanation, confidence)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING fact_id
        """, (
            claim_id,
            fact_check_result['verdict'],
            json.dumps(citations),
            fact_check_result['explanation'],
            fact_check_result['confidence']
        ))
        
        fact_id = self.cursor.fetchone()[0]
        
        # Link sources
        for doc in fact_check_result.get('retrieved_documents', []):
            self.cursor.execute("""
                INSERT INTO fact_checker_sources (fact_id, embedding_id, relevance_score)
                VALUES (%s, %s, %s)
            """, (fact_id, doc['embedding_id'], doc['similarity']))
        
        self.conn.commit()
        
        print(f"✓ Saved (claim_id: {claim_id}, fact_id: {fact_id})")
        
        return fact_id
    
    def close(self):
        """Close database connection"""
        self.cursor.close()
        self.conn.close()


def main():
    """Example usage"""
    
    # Initialize fact checker
    checker = RAGFactChecker()
    
    # Example claims to fact-check
    test_claims = [
        "The unemployment rate in April 2024 was 4.1%",
        "The US population is over 330 million people",
        "The Federal Reserve raised interest rates to 5% in 2023",
        "California has the highest GDP of any US state"
    ]
    
    print(f"{'='*60}")
    print("RAG FACT-CHECKER DEMO (Using HuggingFace API)")
    print(f"{'='*60}\n")
    
    # Fact-check each claim
    for i, claim in enumerate(test_claims, 1):
        print(f"\n{'#'*60}")
        print(f"TEST CLAIM {i}/{len(test_claims)}")
        print(f"{'#'*60}\n")
        
        result = checker.fact_check(claim, top_k=3)
        
        print(f"\n{'='*60}")
        print("RESULT SUMMARY")
        print(f"{'='*60}")
        print(f"Verdict: {result['verdict']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"\nExplanation:")
        print(result['explanation'])
        print(f"\nSources Consulted: {result['num_sources_consulted']}")
        
        # # Save to database
        # fact_id = checker.save_to_database(claim, user_id=1, fact_check_result=result)
        
        print(f"\n{'='*60}\n")
        
        # Optional: Uncomment to wait between claims
        # input("Press Enter to continue to next claim...")
    
    # Close connection
    checker.close()
    
    print("✓ Demo complete!")


if __name__ == "__main__":
    main()
