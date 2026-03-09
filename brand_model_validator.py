import json
import time
import random
import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class BrandModelValidator:
    # basic validator setup
    def __init__(self, api_key: str, model: str = "openai/gpt-4o-mini"):
        self.api_key = (api_key or "").strip()
        self.model = model

    @staticmethod
    def _extract_json(text: str):
        """
        Extract JSON object/list from model output,
        even if the model adds extra text around it.
        """
        if not isinstance(text, str):
            return None

        t = text.strip()

        if "[" in t and "]" in t:
            start = t.find("[")
            end = t.rfind("]")
            if start != -1 and end != -1 and end > start:
                return t[start:end + 1]

        if "{" in t and "}" in t:
            start = t.find("{")
            end = t.rfind("}")
            if start != -1 and end != -1 and end > start:
                return t[start:end + 1]

        return t

    @staticmethod
    # safe fallback when parsing fails
    def _safe_pair_result(brand: str, model: str) -> dict:
        return {
            "corrected_brand": str(brand).strip().lower(),
            "corrected_model": str(model).strip().lower()
        }

    @staticmethod
    def _normalize_output(value: str) -> str:
        return str(value).strip().lower()

    # shared correction rules prompt
    def _base_rules(self) -> str:
        return """
You are an expert in global car manufacturers and vehicle models.

Your task is to correct brand and model values in used-car listings.

Rules:
1) Correct spelling mistakes in brand and model names.
2) If the model belongs to a different manufacturer, you MUST fix the brand.
3) If the model is a well-known model of another manufacturer, change the brand to that manufacturer.
4) Use only real car brands and real vehicle models.
5) Prefer the most common global spelling.
6) If the input is already correct, keep it unchanged.
7) If unsure, keep the original values.
8) Do NOT invent new models or trims.

Examples:
cupra | atka -> cupra | ateca
cupra | borne -> cupra | born
toyot | corola -> toyota | corolla
volksvagen | golf -> volkswagen | golf
mercdes | c200 -> mercedes-benz | c200
renault | u5 -> aiways | u5
aeways | u5 -> aiways | u5
""".strip()

    # request wrapper with retry/backoff
    def _post_with_retries(self, payload: dict, max_attempts: int = 5) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Used Car Market Analysis",
        }

        last_err = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.post(
                    OPENROUTER_URL,
                    headers=headers,
                    json=payload,
                    timeout=60
                )

                if response.status_code != 200:
                    raise RuntimeError(
                        f"OpenRouter error {response.status_code}: {response.text[:800]}"
                    )

                return response.json()

            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ) as e:
                last_err = e

                if attempt == max_attempts:
                    raise RuntimeError(f"Network failure after retries: {e}") from e

                sleep_s = (2 ** attempt) + random.random()
                print(
                    f"Network issue (attempt {attempt}/{max_attempts}): {e}. "
                    f"Retry in {sleep_s:.1f}s..."
                )
                time.sleep(sleep_s)

        raise RuntimeError(f"Unexpected error (last_err={last_err})")

    # validate one brand/model pair
    def validate(self, brand: str, model: str) -> dict:
        """
        Validate a single brand/model pair.
        Returns:
        {"corrected_brand": "...", "corrected_model": "..."}
        """
        brand = str(brand).strip().lower()
        model = str(model).strip().lower()

        prompt = f"""
{self._base_rules()}

Input:
Brand: {brand}
Model: {model}

Return ONLY JSON exactly:
{{"corrected_brand":"...","corrected_model":"..."}}
""".strip()

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }

        data = self._post_with_retries(payload)

        content = data["choices"][0]["message"]["content"]
        json_text = self._extract_json(content)

        try:
            parsed = json.loads(json_text)
            return {
                "corrected_brand": self._normalize_output(
                    parsed.get("corrected_brand", brand)
                ),
                "corrected_model": self._normalize_output(
                    parsed.get("corrected_model", model)
                ),
            }
        except Exception:
            return self._safe_pair_result(brand, model)

    # validate multiple pairs in one call
    def validate_batch(self, pairs: list) -> list:
        """
        Validate a batch of pairs.
        Input:
            [{"brand": "...", "model": "..."}, ...]
        Returns:
            [
              {
                "brand": "...",
                "model": "...",
                "corrected_brand": "...",
                "corrected_model": "..."
              },
              ...
            ]
        """
        clean_pairs = [
            {
                "brand": str(p["brand"]).strip().lower(),
                "model": str(p["model"]).strip().lower()
            }
            for p in pairs
        ]

        prompt = f"""
{self._base_rules()}

INPUT (JSON array):
{json.dumps(clean_pairs, ensure_ascii=False)}

Return ONLY a JSON array, in the same order and same length as input.
Each item must include:
brand, model, corrected_brand, corrected_model

Example output:
[
  {{"brand":"cupra","model":"atka","corrected_brand":"cupra","corrected_model":"ateca"}},
  {{"brand":"renault","model":"u5","corrected_brand":"aiways","corrected_model":"u5"}}
]
""".strip()

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }

        data = self._post_with_retries(payload)

        content = data["choices"][0]["message"]["content"]
        json_text = self._extract_json(content)

        parsed = json.loads(json_text)

        if not isinstance(parsed, list) or len(parsed) != len(clean_pairs):
            raise RuntimeError("Batch parse failed or length mismatch")

        out = []
        for i, item in enumerate(parsed):
            orig = clean_pairs[i]

            out.append({
                "brand": orig["brand"],
                "model": orig["model"],
                "corrected_brand": self._normalize_output(
                    item.get("corrected_brand", orig["brand"])
                ),
                "corrected_model": self._normalize_output(
                    item.get("corrected_model", orig["model"])
                ),
            })

        return out
