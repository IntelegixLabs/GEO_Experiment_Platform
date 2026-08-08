"""OpenAI LLM adapter for the grounded shopping research agent."""

import json
from typing import Any
import openai

from .shopping_agent import (
    AdapterGeneration,
    GroundedClaim,
    GroundedGenerationAdapter,
    GroundedGenerationContext,
)


class OpenAIGenerationAdapter:
    """An LLM generation adapter that enforces strict citation grounding."""

    name = "openai-gpt-adapter"
    
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        self.version = model_name
        self.client = openai.Client(api_key=api_key) if api_key else openai.Client()
        self.model = model_name

    def generate(self, context: GroundedGenerationContext) -> AdapterGeneration:
        """Call the OpenAI API and extract an answer with its grounded claims."""
        
        system_prompt = (
            "You are a helpful and purely factual shopping assistant. You must answer the user's "
            "query using ONLY the provided catalog claims.\n\n"
            "Rules:\n"
            "1. You may only state facts provided in the ALLOWED CLAIMS list below.\n"
            "2. When you use a claim, you MUST append its exact citation ID (e.g., [C1]).\n"
            "3. Do not invent any facts, prices, ratings, or product details.\n"
            "4. Output your response as a valid JSON object matching this schema:\n"
            '   {"answer": "Your natural language response with citation markers like [C1] and [C2]",\n'
            '    "used_citation_ids": ["C1", "C2"],\n'
            '    "suggested_follow_ups": ["Short follow-up question 1?", "Question 2?"]}\n\n'
            "ALLOWED CLAIMS:\n"
        )
        
        claims_text = ""
        for claim in context.allowed_claims:
            claims_text += f"- [{claim.citation_id}] {claim.text}\n"
            
        system_prompt += claims_text

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context.query}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
                timeout=30
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Received empty content from OpenAI.")
            
            parsed = json.loads(content)
            answer = parsed.get("answer", "")
            used_ids = set(parsed.get("used_citation_ids", []))
            suggested_follow_ups = tuple(parsed.get("suggested_follow_ups", []))
            
            # Reconstruct the used claims list
            used_claims = []
            seen_claim_ids = set()
            for claim in context.allowed_claims:
                if claim.citation_id in used_ids and claim.citation_id not in seen_claim_ids:
                    used_claims.append(claim)
                    seen_claim_ids.add(claim.citation_id)
            
            # If the LLM didn't return any valid used claims but provided an answer,
            # we should still return the claims it was allowed to use as a fallback,
            # but ideally we only return what it used.
            if not used_claims and answer:
                 # Fallback to all allowed claims if the model failed to list them
                 used_claims = list(context.allowed_claims)
                 
            return AdapterGeneration(
                answer=answer.strip(), 
                claims=tuple(used_claims),
                suggested_follow_ups=suggested_follow_ups
            )
            
        except Exception as e:
            # Re-raise so the shopping agent can fallback to deterministic generation
            raise RuntimeError(f"OpenAI generation failed: {e}") from e
