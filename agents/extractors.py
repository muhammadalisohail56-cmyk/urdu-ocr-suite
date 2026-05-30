import abc
import base64
import json
import logging
from typing import Dict, Any, List
from io import BytesIO
from PIL import Image
import asyncio

import google.generativeai as genai
from openai import AsyncOpenAI

logger = logging.getLogger("extractors")

def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def parse_json_lines(raw: str) -> List[str]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
        return data.get("lines", [])
    except Exception:
        return []

PROMPT = (
    "Transcribe ALL Urdu text exactly, preserving line breaks and right-to-left order. "
    "Output ONLY the Urdu text as JSON: {\"lines\": [\"...\", \"...\"]}. "
    "Do not translate, summarize, or add commentary."
)

class ExtractorAgent(abc.ABC):
    def __init__(self, provider: str, model_name: str, api_key: str, timeout: int = 60, retries: int = 1):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries

    @abc.abstractmethod
    async def _extract(self, image: Image.Image) -> str:
        pass

    async def extract(self, image: Image.Image) -> Dict[str, Any]:
        last_exc = None
        for attempt in range(self.retries + 1):
            try:
                raw_response = await asyncio.wait_for(self._extract(image), timeout=self.timeout)
                lines = parse_json_lines(raw_response)
                return {"lines": lines, "raw": raw_response, "error": None}
            except Exception as e:
                last_exc = e
                logger.warning(f"{self.provider} attempt {attempt+1} failed: {e}")
        return {"lines": [], "raw": "", "error": str(last_exc)}

class GeminiExtractor(ExtractorAgent):
    def __init__(self, model_name: str, api_key: str, timeout: int = 60, retries: int = 1):
        super().__init__("gemini", model_name, api_key, timeout, retries)
        # Configuring locally or assuming it's already configured. For async it's fine.
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    async def _extract(self, image: Image.Image) -> str:
        response = await self.model.generate_content_async([PROMPT, image])
        return response.text

class OpenAIExtractor(ExtractorAgent):
    def __init__(self, model_name: str, api_key: str, timeout: int = 60, retries: int = 1):
        super().__init__("openai", model_name, api_key, timeout, retries)
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def _extract(self, image: Image.Image) -> str:
        b64_img = image_to_base64(image)
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_img}"}
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=2000
        )
        return response.choices[0].message.content


