"""
VerifyLens Fact Checker with RAG Fallback
Drop-in replacement for verify_claim_with_perplexity with RAG → Perplexity fallback

Usage:
    from fact_checker_verifylens import verify_claim
    
    result = verify_claim(claim_text)
    # Returns same format as verify_claim_with_perplexity
"""

import os
import sys
from typing import Dict, Any, List
from dotenv import load_dotenv

# Import your friend's RAG system
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'rag'))
    from rag_fact_checker import RAGFactChecker
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  RAG system not available: {e}")
    RAG_AVAILABLE = False

# Import the original Perplexity function
try:
    from fact_check import verify_claim_with_perplexity
    PERPLEXITY_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Perplexity not available: {e}")
    PERPLEXITY_AVAILABLE = False

load_dotenv()


class FactCheckerWithRAG:
    """
    Fact checker that tries RAG first, then falls back to Perplexity
    Returns the exact same format as verify_claim_with_perplexity
    """
    
    def __init__(self, rag_confidence_threshold: int = 70):
        """
        Initialize the fact checker
        
        Args:
            rag_confidence_threshold: Minimum confidence to trust RAG (0-100)
        """
        self.rag_confidence_threshold = rag_confidence_threshold
        self.rag_checker = None
        
        # Try to initialize RAG
        if RAG_AVAILABLE:
            try:
                print("[FactChecker] Initializing RAG system...")
                self.rag_checker = RAGFactChecker()
                print("[FactChecker] ✓ RAG system initialized")
            except Exception as e:
                print(f"[FactChecker] ✗ Failed to initialize RAG: {e}")
                self.rag_checker = None
        
        # Statistics
        self.stats = {
            'total_checks': 0,
            'rag_successes': 0,
            'perplexity_fallbacks': 0,
            'errors': 0
        }
    
    def verify_claim(self, claim_text: str) -> Dict[str, Any]:
        """
        Verify a claim using RAG → Perplexity fallback
        Returns same format as verify_claim_with_perplexity:
        {
            "verdict": "true|false|misleading|partially-true|unverified",
            "confidence": 85,
            "citations": [...],
            "explanation": "..."
        }
        """
        self.stats['total_checks'] += 1
        print(f"\n[FactChecker] Verifying claim: '{claim_text[:60]}...'")
        
        # Step 1: Try RAG first
        if self.rag_checker:
            try:
                print("[FactChecker] Attempting RAG fact-check...")
                rag_result = self.rag_checker.fact_check(claim_text, top_k=5)
                
                # Check if RAG result is reliable
                if self._is_rag_reliable(rag_result):
                    self.stats['rag_successes'] += 1
                    print(f"[FactChecker] ✓ Using RAG result (confidence: {rag_result['confidence']}%)")
                    
                    # Convert RAG format to Perplexity format
                    return self._convert_rag_to_perplexity_format(rag_result)
                else:
                    print(f"[FactChecker] ⚠️  RAG unreliable (confidence: {rag_result.get('confidence', 0)}%), falling back to Perplexity")
            
            except Exception as e:
                print(f"[FactChecker] ✗ RAG error: {e}")
        
        # Step 2: Fallback to Perplexity
        if PERPLEXITY_AVAILABLE:
            try:
                self.stats['perplexity_fallbacks'] += 1
                print("[FactChecker] Using Perplexity...")
                result = verify_claim_with_perplexity(claim_text)
                print(f"[FactChecker] ✓ Perplexity result: {result['verdict']}")
                return result
            
            except Exception as e:
                print(f"[FactChecker] ✗ Perplexity error: {e}")
                self.stats['errors'] += 1
                return self._error_result(claim_text, str(e))
        
        # No systems available
        self.stats['errors'] += 1
        return self._error_result(claim_text, "No fact-checking systems available")
    
    def _is_rag_reliable(self, rag_result: Dict) -> bool:
        """Check if RAG result is reliable enough"""
        confidence = rag_result.get('confidence', 0)
        verdict = rag_result.get('verdict', 'UNVERIFIABLE')
        num_sources = rag_result.get('num_sources_consulted', 0)
        
        # RAG is reliable if:
        # 1. Confidence >= threshold
        # 2. Verdict is not UNVERIFIABLE or ERROR
        # 3. Has sources
        
        is_reliable = (
            confidence >= self.rag_confidence_threshold and
            verdict not in ['UNVERIFIABLE', 'ERROR'] and
            num_sources > 0
        )
        
        if is_reliable:
            print(f"[FactChecker] RAG is reliable: confidence={confidence}%, verdict={verdict}, sources={num_sources}")
        else:
            print(f"[FactChecker] RAG is unreliable: confidence={confidence}%, verdict={verdict}, sources={num_sources}")
        
        return is_reliable
    
    def _convert_rag_to_perplexity_format(self, rag_result: Dict) -> Dict[str, Any]:
        """
        Convert RAG result format to Perplexity format
        
        RAG format:
        {
            "verdict": "TRUE/FALSE/PARTIALLY TRUE/UNVERIFIABLE",
            "confidence": 85,
            "explanation": "...",
            "sources_used": ["Source 1", "Source 2"],
            "retrieved_documents": [...]
        }
        
        Perplexity format:
        {
            "verdict": "true|false|misleading|partially-true|unverified",
            "confidence": 85,
            "citations": [{title, url, snippet, source}],
            "explanation": "..."
        }
        """
        # Convert verdict to lowercase format
        verdict_map = {
            'TRUE': 'true',
            'FALSE': 'false',
            'PARTIALLY TRUE': 'partially-true',
            'UNVERIFIABLE': 'unverified',
            'ERROR': 'unverified'
        }
        
        rag_verdict = rag_result.get('verdict', 'UNVERIFIABLE')
        converted_verdict = verdict_map.get(rag_verdict, 'unverified')
        
        # Convert retrieved documents to citations format
        citations = []
        retrieved_docs = rag_result.get('retrieved_documents', [])
        
        for doc in retrieved_docs:
            citation = {
                "title": doc.get('title', 'Unknown Title'),
                "url": doc.get('source_id', '#'),
                "snippet": doc.get('content', '')[:200] + '...' if len(doc.get('content', '')) > 200 else doc.get('content', ''),
                "source": doc.get('source', 'RAG Database')
            }
            citations.append(citation)
        
        # Build explanation
        explanation = rag_result.get('explanation', '')
        
        # Add source attribution if from RAG
        explanation_with_source = f"[RAG Database] {explanation}"
        
        return {
            "verdict": converted_verdict,
            "confidence": rag_result.get('confidence', 0),
            "citations": citations,
            "explanation": explanation_with_source
        }
    
    def _error_result(self, claim_text: str, error_msg: str) -> Dict[str, Any]:
        """Return error result in Perplexity format"""
        return {
            "verdict": "unverified",
            "confidence": 0,
            "citations": [],
            "explanation": f"Error during fact-checking: {error_msg}"
        }
    
    def get_stats(self) -> Dict:
        """Get fact-checking statistics"""
        total = self.stats['total_checks']
        return {
            **self.stats,
            'rag_success_rate': f"{(self.stats['rag_successes'] / total * 100):.1f}%" if total > 0 else "0%",
            'perplexity_fallback_rate': f"{(self.stats['perplexity_fallbacks'] / total * 100):.1f}%" if total > 0 else "0%"
        }
    
    def close(self):
        """Clean up resources"""
        if self.rag_checker:
            self.rag_checker.close()


# ============================================================================
# GLOBAL INSTANCE - Initialize once at module import
# ============================================================================

# Initialize the fact checker (will be reused across requests)
_fact_checker_instance = None

def get_fact_checker() -> FactCheckerWithRAG:
    """Get or create the global fact checker instance"""
    global _fact_checker_instance
    if _fact_checker_instance is None:
        threshold = int(os.getenv('RAG_CONFIDENCE_THRESHOLD', 70))
        _fact_checker_instance = FactCheckerWithRAG(rag_confidence_threshold=threshold)
    return _fact_checker_instance


# ============================================================================
# DROP-IN REPLACEMENT FUNCTION
# ============================================================================

def verify_claim(claim_text: str) -> Dict[str, Any]:
    """
    Drop-in replacement for verify_claim_with_perplexity
    
    This function has the same signature and return format as 
    verify_claim_with_perplexity, but tries RAG first before falling back.
    
    Args:
        claim_text: The claim to verify
        
    Returns:
        {
            "verdict": "true|false|misleading|partially-true|unverified",
            "confidence": 85,
            "citations": [...],
            "explanation": "..."
        }
    """
    checker = get_fact_checker()
    return checker.verify_claim(claim_text)


# ============================================================================
# TEST CODE
# ============================================================================

if __name__ == "__main__":
    import json
    
    # Test claims
    test_claims = [
        "The unemployment rate in April 2024 was 4.1%",
        "The Earth is flat",
        "Water boils at 100 degrees Celsius at sea level"
    ]
    
    print("\n" + "="*70)
    print("FACT-CHECKING TEST WITH RAG FALLBACK")
    print("="*70)
    
    for claim in test_claims:
        result = verify_claim(claim)
        
        print("\n" + "-"*70)
        print(f"Claim: {claim}")
        print(f"Result:")
        print(json.dumps(result, indent=2))
    
    # Show stats
    checker = get_fact_checker()
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    print(json.dumps(checker.get_stats(), indent=2))
    
    # Clean up
    checker.close()
