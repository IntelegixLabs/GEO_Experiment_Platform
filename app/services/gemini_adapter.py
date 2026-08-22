"""Gemini LLM adapter for the grounded shopping research agent."""

import json
from typing import Any
from google import genai
from google.genai import types

from .shopping_agent import (
    AdapterGeneration,
    GroundedClaim,
    GroundedGenerationAdapter,
    GroundedGenerationContext,
)


class GeminiGenerationAdapter:
    """An LLM generation adapter that enforces strict citation grounding via Gemini."""

    name = "gemini-adapter"
    
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: str | None = None) -> None:
        self.version = model_name
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.model = model_name

    def generate(self, context: GroundedGenerationContext) -> AdapterGeneration:
        """Call the Gemini API and extract an answer with its grounded claims."""
        
        system_prompt = (
            "You are a helpful and purely factual shopping assistant. You must answer the user's "
            "query using ONLY the provided catalog claims.\n\n"
            "Rules:\n"
            "1. You may only state facts provided in the ALLOWED CLAIMS list below.\n"
            "2. When you use a claim, you MUST append its exact citation ID (e.g., [C1]).\n"
            "3. Do not invent any facts, prices, ratings, or product details.\n"
            "4. End your `answer` by actively interacting with the user or asking an engaging question to keep the conversation going.\n"
            "5. Output your response as a valid JSON object matching this schema:\n"
            '   {"answer": "Your natural language response with citation markers like [C1] and [C2] ending with a conversational question",\n'
            '    "used_citation_ids": ["C1", "C2"],\n'
            '    "suggested_follow_ups": ["Can you tell me if [Product] has a warranty?", "What are the alternatives to [Brand]?"]}\n\n'
            "Note: `suggested_follow_ups` MUST be 2-3 preemptive questions phrased from the USER's perspective that they might want to click next.\n\n"
            "ALLOWED CLAIMS:\n"
        )
        
        claims_text = ""
        for claim in context.allowed_claims:
            claims_text += f"- [{claim.citation_id}] {claim.text}\n"
            
        system_prompt += claims_text

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=context.query,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            content = response.text
            if not content:
                raise ValueError("Received empty content from Gemini.")
            
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
            
            if not used_claims and answer:
                 used_claims = list(context.allowed_claims)
                 
            return AdapterGeneration(
                answer=answer.strip(), 
                claims=tuple(used_claims),
                suggested_follow_ups=suggested_follow_ups
            )
            
        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {e}") from e
