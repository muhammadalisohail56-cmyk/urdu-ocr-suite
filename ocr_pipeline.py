import os
import math
import logging
import asyncio
import json
from typing import List, Tuple, Dict, Any

import fitz  # PyMuPDF
from PIL import Image

from agents.extractors import GeminiExtractor, OpenAIExtractor
from agents.reconciler import ReconcilerAgent
from agents.judge import JudgeAgent
from agents.linguist import LinguistAgent

logger = logging.getLogger("ocr-pipeline")

PDF_RENDER_DPI = int(os.getenv("PDF_RENDER_DPI", "200"))
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-2.5-pro")

IMAGE_TOKENS_PER_TILE = 258
PROMPT_TOKENS = 100

MODEL_PRICING = {
    "gemini-2.5-pro": {"provider": "gemini", "in": 1.25, "out": 10.0, "out_tokens_per_page": 1000, "sec_per_page": 20, "label": "Gemini Pro (Accurate)"},
    "gemini-2.5-flash": {"provider": "gemini", "in": 0.30, "out": 2.50, "out_tokens_per_page": 1000, "sec_per_page": 10, "label": "Gemini Flash (Fast)"},
    "gpt-4o": {"provider": "openai", "in": 5.00, "out": 15.00, "out_tokens_per_page": 1000, "sec_per_page": 20, "label": "GPT-4o (Strong)"}
}

def _get_api_key(provider: str) -> str:
    if provider == "gemini":
        return os.getenv("GEMINI_API_KEY", "")
    elif provider == "openai":
        return os.getenv("OPENAI_API_KEY", "")
    return ""

def _image_tokens(width: int, height: int) -> int:
    if width <= 384 and height <= 384:
        return IMAGE_TOKENS_PER_TILE
    crop = max(1, math.floor(min(width, height) / 1.5))
    tiles = math.ceil(width / crop) * math.ceil(height / crop)
    return tiles * IMAGE_TOKENS_PER_TILE

def _human_time(seconds: float) -> str:
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}h {m}m"
    return f"{m}m {s}s" if m else f"{s}s"

def page_dimensions(pdf_path: str, dpi: int = PDF_RENDER_DPI) -> List[Tuple[int, int]]:
    dims = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            r = page.rect
            dims.append((int(r.width / 72 * dpi), int(r.height / 72 * dpi)))
    return dims

def render_page(pdf_path: str, page_index: int, dpi: int = PDF_RENDER_DPI) -> Image.Image:
    with fitz.open(pdf_path) as doc:
        pix = doc[page_index].get_pixmap(dpi=dpi)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

def estimate_usage(pdf_path: str) -> dict:
    dims = page_dimensions(pdf_path)
    n = len(dims)
    
    input_tokens = 0
    for (w, h) in dims:
        img_tok = _image_tokens(w, h)
        input_tokens += img_tok + PROMPT_TOKENS

    models = {}
    for name, p in MODEL_PRICING.items():
        if not _get_api_key(p["provider"]):
            continue # skip models without keys
        
        output_tokens = p["out_tokens_per_page"] * n
        cost = input_tokens / 1e6 * p["in"] + output_tokens / 1e6 * p["out"]
        secs = p["sec_per_page"] * n
        models[name] = {
            "label": p["label"],
            "provider": p["provider"],
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(input_tokens + output_tokens),
            "est_cost_usd": round(cost, 4),
            "est_seconds": int(secs),
            "est_time": _human_time(secs),
        }

    return {
        "num_pages": n,
        "models": models,
        # The user selects multiple, but we provide estimates per model so the UI can sum them up
    }

async def transcribe_page_multi_agent(image: Image.Image, selected_models: List[str]) -> dict:
    extractors = []
    for m in selected_models:
        provider = MODEL_PRICING[m]["provider"]
        api_key = _get_api_key(provider)
        if not api_key:
            continue
        
        if provider == "gemini":
            extractors.append(GeminiExtractor(m, api_key))
        elif provider == "openai":
            extractors.append(OpenAIExtractor(m, api_key))
            
    if not extractors:
        return {"text": "", "confidence": 0.0, "notes": "No valid extractors configured.", "tokens_json": "[]", "agent_logs": "{}"}
        
    # Fan out to extractors
    tasks = [ext.extract(image) for ext in extractors]
    results = await asyncio.gather(*tasks)
    
    agent_outputs = {}
    agent_logs = {}
    for ext, res in zip(extractors, results):
        if res["lines"]:
            agent_outputs[ext.model_name] = res["lines"]
        agent_logs[ext.model_name] = res
        
    # Reconcile
    reconciler = ReconcilerAgent()
    reconciled_result = reconciler.reconcile(agent_outputs)
    tokens = reconciled_result["tokens"]
    
    # Adjudicate low/medium confidence tokens
    uncertain_tokens = [t for t in tokens if t["confidence"] < 0.8]
    if uncertain_tokens:
        judge_key = _get_api_key("gemini") # Assuming Gemini for Judge as default
        if judge_key:
            judge = JudgeAgent(JUDGE_MODEL, judge_key)
            judgments = await judge.judge(image, uncertain_tokens)
            
            # Update tokens with judgments
            judgment_map = {j["index"]: j for j in judgments}
            # We need the original index to map back
            uncertain_indices = [i for i, t in enumerate(tokens) if t["confidence"] < 0.8]
            
            for ui, tok_idx in enumerate(uncertain_indices):
                if ui in judgment_map:
                    j = judgment_map[ui]
                    tokens[tok_idx]["text"] = j["text"]
                    tokens[tok_idx]["confidence"] = j["confidence"]
                    tokens[tok_idx]["rationale"] = f"Judge: {j['rationale']}"
                    
    # Linguist Post-processing
    linguist = LinguistAgent()
    for i in range(len(tokens)):
        tokens[i] = linguist.process(tokens[i])
        
    # Final assembly
    final_text = " ".join([t["text"] for t in tokens])
    overall_confidence = sum([t["confidence"] for t in tokens]) / len(tokens) if tokens else 0.0
    
    return {
        "text": final_text,
        "confidence": overall_confidence,
        "notes": f"Ensemble of {len(extractors)} models.",
        "tokens_json": json.dumps(tokens, ensure_ascii=False),
        "agent_logs": json.dumps(agent_logs, ensure_ascii=False)
    }
