"""
LLM explanation generator for fraud detection.

Generates natural language explanations by:
1. Taking transaction context + SHAP values + retrieved fraud patterns
2. Crafting a prompt grounded in patterns
3. Calling LLM (Ollama Mistral locally, OpenAI GPT-4o-mini in prod)
4. Returning plain-English explanation

Supports two LLM backends:
  - Ollama (local, free, Mistral 7B): http://localhost:11434
  - OpenAI API (prod, gpt-4o-mini): via OPENAI_API_KEY
"""

import os
import warnings
from typing import Optional

warnings.filterwarnings("ignore")

# Configuration
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai").lower()  # "ollama" or "openai"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = "mistral"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# - System prompt

SYSTEM_PROMPT = """You are an expert fraud analyst explaining why a transaction was flagged as fraudulent.

Your explanations should be:
1. Grounded in evidence: Reference SHAP feature drivers and retrieved fraud patterns
2. Actionable: Explain what indicators triggered the flag
3. Concise: 2-3 sentences max, avoid jargon
4. Honest: If confidence is low, say so; don't overstate certainty

Always start with the top SHAP driver, then mention supporting patterns.
End with the fraud score (probability).

Example:
"This transaction was flagged due to email domain mismatch (r_emaildomain: +0.31) combined with 
above-average amount (+0.18), matching 'Card-Not-Present Email Mismatch' fraud pattern (35% fraud rate). 
Fraud probability: 83%."
"""


# - LLM generators


class OpenAIGenerator:
    """Generate explanations using OpenAI GPT-4o-mini."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI client."""
        from openai import OpenAI
        key = api_key or OPENAI_API_KEY
        if not key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=key)

    def generate(
        self,
        fraud_score: float,
        shap_top: list[tuple[str, float]],
        patterns: list[dict],
        transaction_context: str = "",
    ) -> str:
        """
        Generate explanation using OpenAI GPT-4o-mini.

        Args:
            fraud_score: Fraud probability (0-1)
            shap_top: List of (feature, value) tuples, sorted by |value|
            patterns: List of retrieved fraud patterns
            transaction_context: Optional transaction summary

        Returns:
            Plain-English explanation
        """
        shap_text = "\n".join(
            [f"  - {feat}: {val:+.3f}" for feat, val in shap_top[:5]]
        )

        patterns_text = "\n".join(
            [
                f"  - {p['name']} ({p['fraud_rate_pct']:.1f}% fraud rate): "
                f"{p['description'][:100]}..."
                for p in patterns[:3]
            ]
        ) if patterns else "  (no patterns retrieved)"

        prompt = f"""Transaction Fraud Analysis Request:

SHAP Feature Drivers (top 5):
{shap_text}

Fraud Score: {fraud_score:.1%}

Retrieved Fraud Patterns:
{patterns_text}

{f'Transaction Context: {transaction_context}' if transaction_context else ''}

Please explain why this transaction was flagged as fraudulent, grounding your
explanation in the SHAP drivers and retrieved patterns. Keep it to 2-3 sentences,
end with the fraud probability percentage."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=256,
            temperature=0.5,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        return response.choices[0].message.content.strip()


class OllamaGenerator:
    """Generate explanations using Ollama (local Mistral)."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        """Initialize Ollama client."""
        self.base_url = base_url
        # Test connection
        try:
            import requests
            response = requests.get(f"{base_url}/api/tags", timeout=2)
            if response.status_code != 200:
                raise ConnectionError(f"Ollama not responding: {response.status_code}")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Ollama at {base_url}: {e}")

    def generate(
        self,
        fraud_score: float,
        shap_top: list[tuple[str, float]],
        patterns: list[dict],
        transaction_context: str = "",
    ) -> str:
        """
        Generate explanation using Ollama Mistral.

        Args:
            fraud_score: Fraud probability (0-1)
            shap_top: List of (feature, value) tuples, sorted by |value|
            patterns: List of retrieved fraud patterns
            transaction_context: Optional transaction summary

        Returns:
            Plain-English explanation
        """
        import requests

        # Build context
        shap_text = "\n".join(
            [f"  - {feat}: {val:+.3f}" for feat, val in shap_top[:5]]
        )

        patterns_text = "\n".join(
            [
                f"  - {p['name']} ({p['fraud_rate_pct']:.1f}% fraud rate): "
                f"{p['description'][:80]}..."
                for p in patterns[:3]
            ]
        )

        prompt = f"""Transaction Fraud Analysis Request:

SHAP Feature Drivers (top 5):
{shap_text}

Fraud Score: {fraud_score:.1%}

Retrieved Fraud Patterns:
{patterns_text}

{f'Transaction Context: {transaction_context}' if transaction_context else ''}

Explain why this transaction was flagged. Keep to 2-3 sentences, mention top SHAP drivers and patterns."""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "system": SYSTEM_PROMPT,
                    "stream": False,
                    "temperature": 0.5,
                },
                timeout=10,
            )
            response.raise_for_status()
            return response.json()["response"].strip()

        except Exception as e:
            print(f"[Ollama] Error generating explanation: {e}")
            return f"Fraud detected (score: {fraud_score:.1%}). Analysis unavailable."


# - Factory


class ExplanationGenerator:
    """Factory for LLM-based explanation generation."""

    def __init__(self, backend: Optional[str] = None):
        """
        Initialize with specified backend.

        Args:
            backend: "openai" or "ollama" (defaults to LLM_BACKEND env var)
        """
        self.backend = backend or LLM_BACKEND
        self._generator = None
        self._init_generator()

    def _init_generator(self):
        """Initialize the backend generator."""
        if self.backend == "openai":
            self._generator = OpenAIGenerator()
            print(f"[Generator] Using OpenAI GPT-4o-mini backend")
        elif self.backend == "ollama":
            self._generator = OllamaGenerator()
            print(f"[Generator] Using Ollama (Mistral) backend")
        else:
            raise ValueError(f"Unknown LLM backend: {self.backend}")

    def generate(
        self,
        fraud_score: float,
        shap_values: dict[str, float],
        patterns: list[dict],
        transaction_context: str = "",
    ) -> str:
        """
        Generate explanation for a fraudulent transaction.

        Args:
            fraud_score: Fraud probability (0-1)
            shap_values: Dict of {feature: shap_value}
            patterns: List of retrieved fraud patterns (from retriever)
            transaction_context: Optional transaction summary

        Returns:
            Plain-English explanation (2-3 sentences)
        """
        # Sort SHAP values by absolute value
        shap_top = sorted(
            shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        return self._generator.generate(
            fraud_score=fraud_score,
            shap_top=shap_top,
            patterns=patterns,
            transaction_context=transaction_context,
        )

    def is_available(self) -> bool:
        """Check if LLM backend is available."""
        return self._generator is not None


# Singleton instance
_generator: Optional[ExplanationGenerator] = None


def get_generator(backend: Optional[str] = None) -> ExplanationGenerator:
    """Lazy-load and return singleton generator."""
    global _generator
    if _generator is None:
        _generator = ExplanationGenerator(backend=backend)
    return _generator
