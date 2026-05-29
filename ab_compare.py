"""
ab_compare.py
=============
A/B comparison harness to decide whether the hybrid (UTRNet + Gemini) pipeline
is worth its cost on YOUR documents, versus a cheaper single-engine setup.

For each sampled page it runs up to four variants and compares them:

    A) utrnet        — local, line-segmented UTRNet (free, offline)
    B) gemini_pro    — Gemini Pro, single transcription call
    C) gemini_flash  — Gemini Flash, single transcription call (cheaper)
    D) reconciled    — the full hybrid: Pro reconciles UTRNet + Pro

What it measures
----------------
* **Agreement** between transcriptions, via character-level similarity
  (`difflib` ratio, 0–1). NOTE: this is engine-vs-engine *agreement*, not
  accuracy — there is no ground truth here. Use it to decide:
    - utrnet↔pro HIGH  → UTRNet corroborates Gemini → hybrid adds confidence.
    - utrnet↔pro LOW   → UTRNet disagrees; on a doc where Gemini is right, that
                         means UTRNet is mostly noise (drop it or expect little).
    - flash↔pro HIGH   → Flash is nearly as good as Pro → use Flash, save ~75%.
    - reconciled↔pro   → how much the reconciliation actually changed Pro's text.
* **Real cost**, from each call's `usage_metadata` token counts × current rates.
* **Latency** per variant.

Usage
-----
    export GEMINI_API_KEY=...
    ./venv/bin/python ab_compare.py --pdf "/path/doc.pdf" --pages 3 --extrapolate 170
"""

from __future__ import annotations

import os
import json
import time
import difflib
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import fitz
from PIL import Image
import google.generativeai as genai

from utr_wrapper import run_utrnet_page
from main import TRANSCRIBE_PROMPT, RECONCILE_PROMPT, _parse_reconcile_json

# --- Pricing ($ per 1M tokens). Edit if Google's rates change. -------------- #
# Verified gemini-2.5-pro from ai.google.dev/gemini-api/docs/pricing (≤200k ctx).
# Flash rates are the published 2.5-flash standard rates (approximate).
RATES = {
    "pro":   {"in": 1.25, "out": 10.00},
    "flash": {"in": 0.30, "out": 2.50},
}


@dataclass
class CallStat:
    text: str = ""
    in_tokens: int = 0
    out_tokens: int = 0
    seconds: float = 0.0
    cost_usd: float = 0.0
    extra: dict = field(default_factory=dict)


def _similarity(a: str, b: str) -> float:
    """Character-level similarity ratio in [0, 1] (1.0 = identical)."""
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()


def _usage_cost(usage, tier: str) -> tuple[int, int, float]:
    in_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
    out_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
    # Reasoning ("thinking") tokens are billed as output; include if reported.
    thoughts = int(getattr(usage, "thoughts_token_count", 0) or 0)
    out_tok += thoughts
    rate = RATES[tier]
    cost = in_tok / 1e6 * rate["in"] + out_tok / 1e6 * rate["out"]
    return in_tok, out_tok, cost


def _gemini_transcribe(image: Image.Image, model_name: str, tier: str) -> CallStat:
    t0 = time.time()
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content([TRANSCRIBE_PROMPT, image])
    secs = time.time() - t0
    in_tok, out_tok, cost = _usage_cost(resp.usage_metadata, tier)
    return CallStat((resp.text or "").strip(), in_tok, out_tok, secs, cost)


def _gemini_reconcile(image, utr_text, gem_text, model_name, tier) -> CallStat:
    t0 = time.time()
    model = genai.GenerativeModel(model_name)
    prompt = RECONCILE_PROMPT.format(
        utrnet_text=utr_text or "(none)", gemini_text=gem_text or "(none)"
    )
    resp = model.generate_content([prompt, image])
    secs = time.time() - t0
    parsed = _parse_reconcile_json(resp.text or "", fallback=gem_text or utr_text)
    in_tok, out_tok, cost = _usage_cost(resp.usage_metadata, tier)
    return CallStat(parsed["reconciled_text"], in_tok, out_tok, secs, cost,
                    extra={"notes": parsed["notes"]})


def _utrnet(image: Image.Image) -> CallStat:
    t0 = time.time()
    res = run_utrnet_page(image)
    secs = time.time() - t0
    return CallStat(res["text"], 0, 0, secs, 0.0,
                    extra={"lines": res.get("num_lines"), "confidence": res["confidence"]})


def _render_pages(pdf_path: str, n: int, dpi: int,
                  page_list: Optional[List[int]] = None) -> List[tuple[int, Image.Image]]:
    """Render pages to (page_number, image). If page_list (1-based) is given it
    takes precedence over the first-`n` behaviour."""
    doc = fitz.open(pdf_path)
    if page_list:
        indices = [p - 1 for p in page_list if 0 <= p - 1 < doc.page_count]
    else:
        indices = list(range(min(n, doc.page_count)))
    out = []
    for i in indices:
        pix = doc[i].get_pixmap(dpi=dpi)
        out.append((i + 1, Image.frombytes("RGB", (pix.width, pix.height), pix.samples)))
    doc.close()
    return out


def run(args: argparse.Namespace) -> dict:
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY not set.")
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    page_list = [int(x) for x in args.page_list.split(",")] if args.page_list else None
    pages = _render_pages(args.pdf, args.pages, args.dpi, page_list)
    print(f"Comparing {len(pages)} page(s) from {os.path.basename(args.pdf)}\n")

    per_page = []
    for pos, (idx, image) in enumerate(pages, start=1):
        print(f"--- page {idx} ({pos}/{len(pages)}) ---")
        utr = _utrnet(image)
        pro = _gemini_transcribe(image, args.pro, "pro")
        flash = _gemini_transcribe(image, args.flash, "flash") if not args.no_flash else None
        recon = None
        if not args.no_reconcile:
            recon = _gemini_reconcile(image, utr.text, pro.text, args.pro, "pro")

        row = {
            "page": idx,
            "utrnet_lines": utr.extra.get("lines"),
            "utrnet_conf": round(utr.extra.get("confidence", 0.0), 3),
            "chars": {
                "utrnet": len(utr.text), "pro": len(pro.text),
                "flash": len(flash.text) if flash else None,
                "reconciled": len(recon.text) if recon else None,
            },
            "agreement": {
                "utrnet_vs_pro": round(_similarity(utr.text, pro.text), 3),
                "flash_vs_pro": round(_similarity(flash.text, pro.text), 3) if flash else None,
                "reconciled_vs_pro": round(_similarity(recon.text, pro.text), 3) if recon else None,
            },
            "cost_usd": {
                "pro_transcribe": round(pro.cost_usd, 5),
                "flash_transcribe": round(flash.cost_usd, 5) if flash else None,
                "reconcile": round(recon.cost_usd, 5) if recon else None,
            },
            "seconds": {
                "utrnet": round(utr.seconds, 2), "pro": round(pro.seconds, 2),
                "flash": round(flash.seconds, 2) if flash else None,
                "reconcile": round(recon.seconds, 2) if recon else None,
            },
            "reconcile_notes": recon.extra.get("notes") if recon else None,
        }
        per_page.append(row)
        a = row["agreement"]
        print(f"  utrnet_lines={row['utrnet_lines']} utrnet_conf={row['utrnet_conf']}")
        print(f"  agreement  utrnet↔pro={a['utrnet_vs_pro']}  "
              f"flash↔pro={a['flash_vs_pro']}  reconciled↔pro={a['reconciled_vs_pro']}")
        print()

    summary = _summarize(per_page, args)
    out = {"pdf": args.pdf, "pages_compared": len(pages),
           "models": {"pro": args.pro, "flash": args.flash},
           "summary": summary, "per_page": per_page}

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        print(f"Wrote detailed results to {args.out}")
    return out


def _mean(vals: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def _summarize(per_page: List[dict], args: argparse.Namespace) -> dict:
    n = len(per_page)
    util_pro = _mean([p["agreement"]["utrnet_vs_pro"] for p in per_page])
    flash_pro = _mean([p["agreement"]["flash_vs_pro"] for p in per_page])
    recon_pro = _mean([p["agreement"]["reconciled_vs_pro"] for p in per_page])

    # Per-page cost of each strategy.
    pro_cost = _mean([p["cost_usd"]["pro_transcribe"] for p in per_page]) or 0.0
    flash_cost = _mean([p["cost_usd"]["flash_transcribe"] for p in per_page])
    recon_cost = _mean([p["cost_usd"]["reconcile"] for p in per_page])

    hybrid_pp = pro_cost + (recon_cost or 0.0)          # pro transcribe + reconcile
    gemini_only_pp = pro_cost                            # pro transcribe only
    flash_only_pp = flash_cost                           # flash transcribe only

    summary = {
        "mean_agreement": {
            "utrnet_vs_pro": util_pro,
            "flash_vs_pro": flash_pro,
            "reconciled_vs_pro": recon_pro,
        },
        "cost_per_page_usd": {
            "hybrid (pro transcribe + reconcile)": round(hybrid_pp, 5),
            "gemini_pro_only": round(gemini_only_pp, 5),
            "gemini_flash_only": round(flash_only_pp, 5) if flash_only_pp is not None else None,
        },
    }
    if args.extrapolate:
        N = args.extrapolate
        summary["projected_cost_usd_for_%d_pages" % N] = {
            "hybrid": round(hybrid_pp * N, 2),
            "gemini_pro_only": round(gemini_only_pp * N, 2),
            "gemini_flash_only": round(flash_only_pp * N, 2) if flash_only_pp is not None else None,
        }
    return summary


def _print_summary(out: dict) -> None:
    s = out["summary"]
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    ma = s["mean_agreement"]
    print("Mean agreement (char similarity, 1.0 = identical):")
    print(f"  UTRNet   vs Gemini-Pro : {ma['utrnet_vs_pro']}")
    print(f"  Flash    vs Gemini-Pro : {ma['flash_vs_pro']}")
    print(f"  Reconciled vs Gemini-Pro: {ma['reconciled_vs_pro']}")
    print()
    print("Cost per page (USD, from real token counts):")
    for k, v in s["cost_per_page_usd"].items():
        print(f"  {k:42s}: ${v}")
    proj = next((v for k, v in s.items() if k.startswith("projected_cost")), None)
    if proj:
        label = [k for k in s if k.startswith("projected_cost")][0]
        print()
        print(label.replace("_", " ") + ":")
        for k, v in proj.items():
            print(f"  {k:42s}: ${v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="A/B compare OCR engines on a PDF.")
    ap.add_argument("--pdf", required=True, help="Path to the PDF.")
    ap.add_argument("--pages", type=int, default=3, help="Pages from start to compare.")
    ap.add_argument("--page-list", default=None,
                    help="Comma-separated 1-based page numbers (e.g. '7,8,9'); overrides --pages.")
    ap.add_argument("--dpi", type=int, default=200, help="Render DPI.")
    ap.add_argument("--pro", default="gemini-2.5-pro", help="Pro model name.")
    ap.add_argument("--flash", default="gemini-2.5-flash", help="Flash model name.")
    ap.add_argument("--no-flash", action="store_true", help="Skip the Flash variant.")
    ap.add_argument("--no-reconcile", action="store_true", help="Skip the reconciliation variant.")
    ap.add_argument("--extrapolate", type=int, default=0, help="Project cost to this many pages.")
    ap.add_argument("--out", default="ab_results.json", help="JSON output path.")
    args = ap.parse_args()

    result = run(args)
    _print_summary(result)
