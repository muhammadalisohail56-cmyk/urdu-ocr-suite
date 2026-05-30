import json
import logging
from typing import List, Dict, Any
from PIL import Image
import google.generativeai as genai
import asyncio

logger = logging.getLogger("judge")

JUDGE_PROMPT = """You are an expert Urdu OCR Judge. You are given an image of a document and a list of uncertain words from an OCR ensemble.
For each item, you are given the surrounding context (previous and next words) and the candidate votes from different OCR models.
Look at the image, find the corresponding text, and adjudicate the correct word.

Input JSON format:
[
  {
    "id": "item_1",
    "context_before": "...",
    "context_after": "...",
    "candidates": {"gemini": "word1", "openai": "word2"}
  }
]

Output EXACTLY JSON in this format:
[
  {
    "id": "item_1",
    "correct_text": "...",
    "confidence": 0.9,
    "rationale": "..."
  }
]
Do not include any markdown formatting like ```json.
"""

class JudgeAgent:
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    async def judge(self, image: Image.Image, uncertain_tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not uncertain_tokens:
            return []

        # Prepare payload
        payload = []
        for i, t in enumerate(uncertain_tokens):
            payload.append({
                "id": str(i),
                "context_before": t.get("context_before", ""),
                "context_after": t.get("context_after", ""),
                "candidates": t.get("votes", {})
            })

        try:
            response = await self.model.generate_content_async([
                JUDGE_PROMPT,
                json.dumps(payload, ensure_ascii=False),
                image
            ])
            text = (response.text or "").strip()
            if text.startswith("```"):
                text = text.strip("`")
                if text.lower().startswith("json"):
                    text = text[4:]
                text = text.strip()
            
            results = json.loads(text)
            
            # Map back to original tokens
            judged = []
            for r in results:
                idx = int(r["id"])
                judged.append({
                    "index": idx,
                    "text": r.get("correct_text", payload[idx]["candidates"].get("gemini", "")),
                    "confidence": float(r.get("confidence", 0.9)),
                    "rationale": r.get("rationale", "")
                })
            return judged
        except Exception as e:
            logger.error(f"Judge Agent failed: {e}")
            return []
