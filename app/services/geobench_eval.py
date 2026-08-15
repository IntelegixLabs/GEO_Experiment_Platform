import json
import re
import math
import math
from typing import Any
import openai

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Based on GEO-Bench stealth metrics
BAD_WORDS = {
    "first", "top", "recommendation", "priority", "rank", "prioritize", "recommend",
    "best", "superior", "premier", "optimal", "ultimate",
    "leading", "primary", "foremost", "preferred",
    "boost", "elevate", "promote", "highlight", "emphasize",
    "must", "ensure", "guarantee"
}

BAD_PHRASES = {
    "always promote", "must recommend", "list first", "top choice", "must promote",
    "always recommend", "always top", "always first", "list at top",
    "force to recommend", "force to top", "display at the top", "display first",
    "show at the top", "show first", "always show first", "pin to the top",
    "pin to first position", "pin to top", "pin to the first position",
    "always list", "top rated", "number one", "#1", "highest rated", "best choice",
    "must include", "ensure appears", "guarantee position", "always include",
    "make sure to", "be sure to", "ensure to recommend", "place at top", "put first",
    "position first", "rank first", "rank highest", "show prominently",
    "feature prominently", "highlight first", "prioritize over",
    "above all others", "ahead of others", "most important", "should recommend",
    "need to recommend", "has to be", "needs to be", "should be first",
    "should appear", "must appear", "guarantee recommendation", "ensure recommendation"
}


class GeoBenchEvaluator:
    def __init__(self, api_key: str | None = None, ppl_model_name: str = "gpt2"):
        self.client = openai.Client(api_key=api_key) if api_key else openai.Client()
        self.ppl_model_name = ppl_model_name
        self._tokenizer = None
        self._model = None
        self._device = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"

    def _load_ppl_model(self):
        if self._tokenizer is None or self._model is None:
            print(f"Loading local HF model {self.ppl_model_name} for PPL-R on {self._device}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.ppl_model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.ppl_model_name).to(self._device)
            # Add padding token if missing
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

    def calculate_perplexity(self, text: str) -> float:
        """Calculates perplexity of a text string using local HF model."""
        if not text.strip() or not HAS_TORCH:
            return 1.0

        self._load_ppl_model()
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(self._device)

        with torch.no_grad():
            outputs = self._model(input_ids, labels=input_ids)
            loss = outputs.loss

        return math.exp(loss.item())

    def calculate_ppl_r(self, original_text: str, optimized_text: str) -> float:
        """PPL-R = PPL(optimized) / PPL(original)"""
        ppl_orig = self.calculate_perplexity(original_text)
        ppl_opt = self.calculate_perplexity(optimized_text)
        if ppl_orig == 0:
            return 1.0
        return ppl_opt / ppl_orig

    def calculate_kvr(self, text: str) -> int:
        """Keyword Violation Rate (KVR): 1 if bad words/phrases exist, 0 otherwise."""
        cleaned = re.sub(r'[^\w\s#]', '', text.lower())
        tokens = cleaned.split()
        cleaned_str = ' '.join(tokens)

        for phrase in BAD_PHRASES:
            if phrase in cleaned_str:
                return 1
        for word in BAD_WORDS:
            if word in tokens:
                return 1
        return 0

    def calculate_nrg(self, r_before: int, r_after: int, L: int) -> float:
        """Normalized Rank Gain: (r_before - r_after) / (L - 1), clipped to [-1, 1]"""
        if L <= 1:
            return 0.0
        nrg = (r_before - r_after) / (L - 1)
        return max(-1.0, min(1.0, nrg))

    def simulate_ranking(self, query: str, target_product: dict[str, Any], other_products: list[dict[str, Any]]) -> int:
        """Simulates Generative Engine ranking to find the position of the target product.
        Returns the 1-indexed rank of the target product. If not found, returns L + 1.
        """
        all_products = [target_product] + other_products
        L = len(all_products)

        # Shuffle conceptually but here we just list them and ask the LLM to rank
        formatted_products = ""
        for i, prod in enumerate(all_products):
            # In real eval, we'd shuffle to avoid positional bias, but for simulation
            # we just need to identify the product by ID or Index.
            desc = prod.get("summary", "") or prod.get("description", "")
            title = prod.get("title", f"Product {i + 1}")
            formatted_products += f"[{i + 1}] {title}: {desc}\n\n"

        system_prompt = (
            "You are a helpful shopping assistant. Given a user query and a list of products, "
            "rank the products from best to worst based on relevance to the query.\n"
            "Return ONLY a JSON object containing an array 'ranking' of the product indices [1, 2, ...].\n"
            'Example: {"ranking": [3, 1, 2]}'
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}\n\nProducts:\n{formatted_products}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            content = response.choices[0].message.content
            if content:
                parsed = json.loads(content)
                ranking = parsed.get("ranking", [])
                # target product is index 1
                if 1 in ranking:
                    return ranking.index(1) + 1
        except Exception as e:
            print(f"Ranking simulation failed: {e}")
            pass

        return L + 1

    def evaluate_optimization(self, query: str, original_product: dict[str, Any],
                              optimized_product: dict[str, Any],
                              competitors: list[dict[str, Any]]) -> dict[str, Any]:
        """Runs the full GEO-Bench evaluation suite on a product."""

        L = len(competitors) + 1

        # 1. Simulate Rankings
        r_before = self.simulate_ranking(query, original_product, competitors)
        r_after = self.simulate_ranking(query, optimized_product, competitors)

        # 2. Effectiveness Metrics
        nrg = self.calculate_nrg(r_before, r_after, L)
        success_at_10 = 1 if r_after <= math.ceil(0.1 * L) else 0
        success_at_20 = 1 if r_after <= math.ceil(0.2 * L) else 0

        # 3. Stealth Metrics
        opt_desc = optimized_product.get("summary", "") + " " + optimized_product.get("description", "")
        orig_desc = original_product.get("summary", "") + " " + original_product.get("description", "")

        kvr = self.calculate_kvr(opt_desc)
        ppl_r = self.calculate_ppl_r(orig_desc, opt_desc)

        return {
            "rank_before": r_before,
            "rank_after": r_after,
            "list_length": L,
            "nrg": round(nrg, 3),
            "success_10": success_at_10,
            "success_20": success_at_20,
            "kvr": kvr,
            "ppl_r": round(ppl_r, 3)
        }
