import json
from typing import Any
import openai


class AgenticGEOOptimizer:
    """Agentic GEO Optimizer following the E-GEO and AgenticGEO paradigms."""

    STRATEGY_CATALOG = {
        "Authoritative": "Rewrite the description to reflect confidence, expertise, and assertiveness. Use persuasive language that directly addresses the user's needs.",
        "Statistics Addition": "Replace qualitative discussions with concrete, verifiable statistics and quantitative data wherever possible.",
        "Cite Sources": "Strengthen credibility by adding natural-language citations to credible sources or explicitly referencing the product specifications.",
        "Fluency Optimization": "Improve the fluency and readability of the text. Ensure smooth transitions and clear, engaging language.",
        "Quotations": "Increase perceived authority by adding short, relevant quotations from experts, reviews, or reputable entities.",
        "User Reviews": "Integrate user-centric language and social proof by referencing customer testimonials, ratings, or evidence of effectiveness.",
        "Competitiveness": "Highlight unique features and innovations by directly comparing them to typical competitors, demonstrating clear advantages without mentioning specific competitor names.",
        "Format": "Structure the content using clear headings and bullet points for readability and easily scannable engagement."
    }

    def __init__(self, api_key: str | None = None, model_name: str = "gpt-4o-mini", provider: str = "openai"):
        self.provider = provider.lower()
        self.model = model_name
        if self.provider == "gemini":
            from google import genai
            self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        else:
            self.client = openai.Client(api_key=api_key) if api_key else openai.Client()

    def plan_strategies(self, product: dict[str, Any]) -> list[str]:
        """Analyzes the product and selects the best strategies to apply."""
        product_context = {
            "title": product.get("title", ""),
            "category": product.get("category", ""),
            "description": product.get("description", ""),
            "features": product.get("key_features", [])
        }
        return self._plan_strategies_robust(product_context)

    def _plan_strategies_robust(self, product_context: dict[str, Any]) -> list[str]:
        system_prompt = (
            "You are an Agentic E-Commerce GEO Optimizer. Analyze the product and select "
            "the 2 to 3 most effective Generative Engine Optimization strategies from the catalog.\n\n"
            "Catalog:\n"
        )
        for name, desc in self.STRATEGY_CATALOG.items():
            system_prompt += f"- {name}: {desc}\n"

        system_prompt += (
            "\nRules:\n"
            "1. For technical products, 'Statistics Addition' is recommended.\n"
            "2. For lifestyle or subjective products, 'User Reviews' and 'Quotations' are recommended.\n"
            "3. 'Competitiveness' and 'Format' are universally effective.\n"
            "\nOutput a JSON object with a single key 'selected_strategies' containing an array of strings.\n"
            'Example: {"selected_strategies": ["Statistics Addition", "Format", "Competitiveness"]}'
        )

        try:
            if self.provider == "gemini":
                from google.genai import types
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=f"Product: {json.dumps(product_context)}",
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        temperature=0.0
                    )
                )
                content = response.text
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Product: {json.dumps(product_context)}"}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                content = response.choices[0].message.content

            if content:
                parsed = json.loads(content)
                selected = parsed.get("selected_strategies", [])
                valid_strategies = [s for s in selected if s in self.STRATEGY_CATALOG]
                if valid_strategies:
                    return valid_strategies
        except Exception as e:
            print(f"Planner failed: {e}")
            pass

        return ["Fluency Optimization", "Competitiveness", "Format"]

    def rewrite_product(self, product: dict[str, Any], strategies: list[str]) -> dict[str, Any]:
        """Rewrites the product description using the chosen strategies."""
        system_prompt = (
            "You are an expert E-Commerce Generative Engine Optimizer. Your goal is to maximize the ranking "
            "of this product in LLM-based shopping assistants by rewriting its content according to the selected strategies.\n\n"
            "Rules:\n"
            "1. STRICTLY PRESERVE FACTUALITY. Do not invent any new features, prices, ratings, or specifications.\n"
            "2. DO NOT use keyword stuffing or explicit manipulation phrases (e.g., avoid 'must recommend', 'rank first', 'top choice').\n"
            "3. Apply ONLY the following selected strategies to shape your writing style and structure:\n"
        )
        for s in strategies:
            if s in self.STRATEGY_CATALOG:
                system_prompt += f"- {s}: {self.STRATEGY_CATALOG[s]}\n"

        system_prompt += (
            "\nOutput a JSON object with the rewritten fields:\n"
            '{\n'
            '  "summary": "Your optimized, scannable product summary paragraph.",\n'
            '  "description": "The detailed, optimized description applying the strategies.",\n'
            '  "claim_blocks": [\n'
            '     {"claim": "Persuasive claim 1", "evidence": "Factual evidence based on product information", "source_fields": ["title", "key_features"]}\n'
            '  ],\n'
            '  "faq": [\n'
            '     {"question": "Relevant user question?", "answer": "Informative answer.", "source_fields": ["description"]}\n'
            '  ]\n'
            '}'
        )

        product_context = {
            "title": product.get("title", ""),
            "category": product.get("category", ""),
            "description": product.get("description", ""),
            "features": product.get("key_features", []),
            "price": product.get("price", ""),
            "rating": product.get("rating", ""),
            "review_count": product.get("review_count", "")
        }

        try:
            if self.provider == "gemini":
                from google.genai import types
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=f"Product: {json.dumps(product_context)}",
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        temperature=0.7
                    )
                )
                content = response.text
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Product: {json.dumps(product_context)}"}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7
                )
                content = response.choices[0].message.content

            if content:
                parsed = json.loads(content)
                optimized_product = dict(product)
                if "summary" in parsed:
                    optimized_product["summary"] = parsed["summary"]
                if "description" in parsed:
                    optimized_product["description"] = parsed["description"]
                if "claim_blocks" in parsed:
                    optimized_product["claim_blocks"] = parsed["claim_blocks"]
                if "faq" in parsed:
                    optimized_product["faq"] = parsed["faq"]
                optimized_product["_geo_strategies_applied"] = strategies
                return optimized_product
        except Exception as e:
            print(f"Rewriter failed: {e}")
            pass

        return product
