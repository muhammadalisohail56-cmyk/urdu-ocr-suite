"""
ocr_pipeline.py
===============
The OCR engine layer for the suite, factored out of the web app.

Two modes (controlled by GEMINI_ONLY, default on):
  * Gemini-only  — one Gemini call per page returning {text, confidence, notes}.
  * Hybrid       — local line-segmented UTRNet + Gemini, reconciled by Gemini.
                   Confidence blends UTRNet's CTC confidence with Gemini's
                   reconciliation confidence.

Also provides PDF rendering and pre-flight token/cost estimation so the UI can
show "Expected Gemini API Usage" before a run starts.
"""

from __future__ import annotations

import os
import io
import json
import math
import logging
from typing import List, Tuple

import fitz  # PyMuPDF
from PIL import Image
import google.generativeai as genai

logger = logging.getLogger("ocr-pipeline")

# Fallback model when a run doesn't specify one. Defaulted to Flash after a
# cost review: Pro's heavy "thinking" tokens (billed as output at $10/1M)
# dominated real billing and far exceeded estimates. The user picks per-run.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_ONLY = os.getenv("GEMINI_ONLY", "1").strip().lower() not in ("0", "false", "no", "off")
PDF_RENDER_DPI = int(os.getenv("PDF_RENDER_DPI", "200"))

# Robustness: cap each Gemini call so a hung request can't freeze a run forever.
GEMINI_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "120"))
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))

IMAGE_TOKENS_PER_TILE = 258
PROMPT_TOKENS = 90  # rough fixed prompt overhead per call

# Per-model pricing ($/1M tokens) + rough per-page estimates that drive the
# pre-flight "expected usage" panel the user sees before starting a run.
#   out_tokens_per_page bakes in typical reasoning/thinking volume — Pro thinks
#   a lot, which is what made it expensive; sec_per_page is a rough throughput
#   figure for the time projection. Tune these from real billing over time.
MODEL_PRICING = {
    "gemini-2.5-pro":        {"in": 1.25, "out": 10.0, "out_tokens_per_page": 3000,
                              "sec_per_page": 60, "label": "Pro — most accurate, priciest"},
    "gemini-2.5-flash":      {"in": 0.30, "out": 2.50, "out_tokens_per_page": 1200,
                              "sec_per_page": 12, "label": "Flash — balanced (recommended)"},
    "gemini-2.5-flash-lite": {"in": 0.10, "out": 0.40, "out_tokens_per_page": 900,
                              "sec_per_page": 6,  "label": "Flash-Lite — cheapest"},
}
RECOMMENDED_MODEL = os.getenv("RECOMMENDED_MODEL", "gemini-2.5-flash")

_configured = False

# --- Prompts ---------------------------------------------------------------- #
GEMINI_ONLY_PROMPT = (
    "You are an expert Urdu OCR engine. Transcribe ALL Urdu text in this image "
    "exactly as written, preserving line breaks and Urdu punctuation. Do not "
    "translate or transliterate.\n"
    "Respond with STRICT JSON only (no markdown fences), exactly these keys:\n"
    '{"text": "<the Urdu transcription>", '
    '"confidence": <float 0-1, your confidence in the transcription>, '
    '"notes": "<brief note on anything illegible or uncertain>"}'
)

TRANSCRIBE_PROMPT = (
    "You are an expert Urdu OCR engine. Transcribe ALL Urdu text visible in this "
    "image exactly as written, preserving line breaks and Urdu punctuation. "
    "Return ONLY the transcribed Urdu text, no explanations."
)

RECONCILE_PROMPT = (
    "You are an expert Urdu OCR adjudicator. You are given the SAME source image "
    "and two transcriptions:\n\n"
    "--- A (local UTRNet) ---\n{utrnet_text}\n\n"
    "--- B (Gemini) ---\n{gemini_text}\n\n"
    "Produce the single most accurate Urdu transcription, correcting OCR errors. "
    "Do not translate or transliterate.\n"
    "Respond with STRICT JSON only (no fences), exactly these keys:\n"
    '{{"reconciled_text": "<final Urdu text>", '
    '"confidence": <float 0-1>, "notes": "<brief note on corrections>"}}'
)


def configure(api_key: str) -> None:
    global _configured
    genai.configure(api_key=api_key)
    _configured = True
    logger.info("Gemini configured (model=%s, mode=%s)",
                GEMINI_MODEL, "gemini-only" if GEMINI_ONLY else "hybrid")


# --- PDF helpers ------------------------------------------------------------ #
def count_pdf_pages(data: bytes) -> int:
    with fitz.open(stream=data, filetype="pdf") as doc:
        return doc.page_count


def render_page(pdf_path: str, page_index: int, dpi: int = PDF_RENDER_DPI) -> Image.Image:
    """Render a single 0-based page of a PDF to a PIL image."""
    with fitz.open(pdf_path) as doc:
        pix = doc[page_index].get_pixmap(dpi=dpi)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def page_dimensions(pdf_path: str, dpi: int = PDF_RENDER_DPI) -> List[Tuple[int, int]]:
    dims = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            r = page.rect
            dims.append((int(r.width / 72 * dpi), int(r.height / 72 * dpi)))
    return dims


# --- Token / cost estimation ------------------------------------------------ #
def _image_tokens(width: int, height: int) -> int:
    """Gemini image token model: 258 tokens per 768x768 tile."""
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


def estimate_usage(pdf_path: str) -> dict:
    """Pre-flight estimate of Gemini usage for EVERY selectable model, so the
    user can compare cost + time before starting a run.

    Hybrid mode issues two Gemini calls per page (transcribe + reconcile);
    Gemini-only issues one. Image input tokens are identical across models; only
    output-token volume and per-token price differ."""
    dims = page_dimensions(pdf_path)
    n = len(dims)
    calls_per_page = 1 if GEMINI_ONLY else 2

    # Input tokens (same for every model).
    input_tokens = 0
    for (w, h) in dims:
        img_tok = _image_tokens(w, h)
        per_page_in = img_tok + PROMPT_TOKENS
        if not GEMINI_ONLY:
            # reconcile call re-sends the image + both transcriptions
            per_page_in += img_tok + PROMPT_TOKENS + 2 * 1200
        input_tokens += per_page_in

    models = {}
    for name, p in MODEL_PRICING.items():
        output_tokens = p["out_tokens_per_page"] * n * calls_per_page
        cost = input_tokens / 1e6 * p["in"] + output_tokens / 1e6 * p["out"]
        secs = p["sec_per_page"] * n * calls_per_page
        models[name] = {
            "label": p["label"],
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(input_tokens + output_tokens),
            "est_cost_usd": round(cost, 4),
            "est_seconds": int(secs),
            "est_time": _human_time(secs),
        }

    return {
        "num_pages": n,
        "mode": "gemini-only" if GEMINI_ONLY else "hybrid",
        "calls_per_page": calls_per_page,
        "recommended": RECOMMENDED_MODEL,
        "models": models,
    }


# --- JSON parsing helper ---------------------------------------------------- #
def _parse_json(raw: str, text_key: str, fallback: str) -> dict:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
        return {
            "text": str(data.get(text_key, fallback)).strip(),
            "confidence": float(data.get("confidence", 0.0)),
            "notes": str(data.get("notes", "")).strip(),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"text": (raw or fallback).strip(), "confidence": 0.0,
                "notes": "Model output was not structured JSON."}


# --- Generation with timeout + retry ---------------------------------------- #
def _generate(model, parts):
    """Call Gemini with a per-request timeout and a few retries, so a hung or
    transient call fails the page cleanly instead of freezing the whole run."""
    last_exc = None
    for attempt in range(GEMINI_MAX_RETRIES + 1):
        try:
            return model.generate_content(
                parts, request_options={"timeout": GEMINI_TIMEOUT_SECONDS}
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("Gemini call failed (attempt %d/%d): %s",
                           attempt + 1, GEMINI_MAX_RETRIES + 1, exc)
    raise RuntimeError(
        f"Gemini call failed after {GEMINI_MAX_RETRIES + 1} attempts: {last_exc}"
    )


# --- Per-page transcription ------------------------------------------------- #
def transcribe_page(image: Image.Image, model_name: str | None = None) -> dict:
    """Transcribe one page image with the chosen model. Returns
    {text, confidence, notes, utrnet_text, gemini_text}."""
    model_name = model_name or GEMINI_MODEL
    if GEMINI_ONLY:
        model = genai.GenerativeModel(model_name)
        resp = _generate(model, [GEMINI_ONLY_PROMPT, image])
        parsed = _parse_json(resp.text or "", "text", fallback="")
        return {
            "text": parsed["text"],
            "confidence": parsed["confidence"],
            "notes": parsed["notes"],
            "utrnet_text": "",
            "gemini_text": parsed["text"],
        }

    # Hybrid: UTRNet (line-segmented) + Gemini, reconciled.
    try:
        from utr_wrapper import run_utrnet_page  # lazy import (heavy ML stack)
    except ImportError as exc:  # torch/numpy/etc. not installed
        raise RuntimeError(
            "Hybrid OCR mode (GEMINI_ONLY=0) requires the UTRNet ML stack, which "
            "is not installed. Run `pip install -r requirements-hybrid.txt`, or "
            "set GEMINI_ONLY=1 to use the lean Gemini-only mode."
        ) from exc
    model = genai.GenerativeModel(model_name)

    utr = run_utrnet_page(image)
    gem_resp = _generate(model, [TRANSCRIBE_PROMPT, image])
    gemini_text = (gem_resp.text or "").strip()

    if not utr["text"]:
        return {"text": gemini_text, "confidence": utr.get("confidence", 0.0),
                "notes": "UTRNet empty; Gemini only.", "utrnet_text": "",
                "gemini_text": gemini_text}

    prompt = RECONCILE_PROMPT.format(utrnet_text=utr["text"], gemini_text=gemini_text or "(none)")
    rec_resp = _generate(model, [prompt, image])
    parsed = _parse_json(rec_resp.text or "", "reconciled_text", fallback=gemini_text)
    # Blend UTRNet CTC confidence with Gemini's reported reconciliation confidence.
    blended = round((utr.get("confidence", 0.0) + parsed["confidence"]) / 2, 3)
    return {
        "text": parsed["text"],
        "confidence": blended,
        "notes": parsed["notes"],
        "utrnet_text": utr["text"],
        "gemini_text": gemini_text,
    }
