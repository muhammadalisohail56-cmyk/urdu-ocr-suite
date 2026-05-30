# Urdu OCR Suite — Project Handoff

_Last updated: 2026-05-30. Paste this into a new chat to continue with full context._

---

## 1. What this project is

A private, **2-user web app** ("Urdu OCR Suite") that transcribes scanned Urdu
PDFs using AI OCR. Built on top of an OCR pipeline that can run in two modes:

- **Gemini-only (DEFAULT):** one Gemini API call per page. Cheapest, simplest.
- **Hybrid (opt-in):** local PyTorch **UTRNet** (line-segmented) + Gemini, with
  Gemini reconciling the two. Only worthwhile for *modern printed* Urdu.

The owner is deploying it so a **supervisor** can log in over the internet and
proofread transcriptions.

- **Working dir:** `/Users/muhammadali/urdu-ocr-system`
- **GitHub (public):** https://github.com/muhammadalisohail56-cmyk/urdu-ocr-suite
- **Latest commit:** `b2ea50a` "Per-run model selection with cost+time estimates; robustness fixes"
- **Local dev:** Python 3.13 venv at `./venv`; run `./venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000`

---

## 2. Tech stack

- **Backend:** FastAPI, served with uvicorn. SQLite via SQLAlchemy ORM.
- **Frontend:** Jinja2 templates + Tailwind (CDN) + Alpine.js (CDN). RTL Urdu via Noto Nastaliq Urdu (Google Fonts).
- **OCR:** `google-generativeai` (Gemini). Hybrid extras (torch/torchvision/numpy/matplotlib) are **optional** — split into `requirements-hybrid.txt`.
- **Other:** python-docx (export), PyMuPDF/`fitz` (PDF render), WebSockets (live progress), stdlib HMAC auth.

---

## 3. File map

| File | Purpose |
|---|---|
| `main.py` | FastAPI app: auth gating, upload+estimate, async OCR worker, WebSocket progress, dashboard/document APIs, inline edits, per-page re-run, DOCX/TXT export, orphan recovery |
| `ocr_pipeline.py` | OCR engine layer: per-model pricing/estimate, `transcribe_page()`, Gemini calls with timeout+retry, PDF render |
| `utr_wrapper.py` | UTRNet (PyTorch) inference + `recognize_page()` (line-segmented). Lazy-imported; only used in hybrid mode |
| `line_segmenter.py` | Lightweight text-line detector (projection profile + local-background subtraction) for UTRNet |
| `database.py` | SQLAlchemy engine/session + additive column "migration" (`_ensure_columns`) |
| `models.py` | `Document` and `Page` ORM models |
| `auth.py` | 2-user HMAC-signed-cookie auth from `USER1/2_CREDENTIALS` env |
| `model.py`, `modules/`, `utils.py`, `UrduGlyphs.txt` | Vendored official UTRNet source + charset |
| `templates/` | `login.html`, `dashboard.html`, `document.html` |
| `previews/` | Standalone static HTML mockups (sample data, no backend) |
| `ab_compare.py` | CLI A/B tool comparing engines/models on a PDF (cost + agreement) |
| `requirements.txt` / `requirements-hybrid.txt` | Lean base / base+ML stack |
| `.replit` | Replit config (python-3.12 module; Reserved VM deploy target) |
| `.env` | Secrets (gitignored). `.env.example` is the template |
| `saved_models/UTRNet-Large/best_norm_ED.pth` | UTRNet weights (181 MB, gitignored — provided locally) |

---

## 4. Features (all built & verified)

1. **Login-gated**, exactly 2 users from env vars (HMAC cookie sessions).
2. **Drag-drop PDF upload** → **pre-flight estimate** showing **cost + time per model**.
3. **Per-run model selection** (user picks model in the confirm modal; see §6).
4. **Async processing** with **WebSocket live progress + rolling ETA**.
5. **Dashboard** table with **inline-editable** Title/Author/Year; shows pages OCRed, confidence, status, model used, failure reason.
6. **Split-screen workspace**: page image (left) | editable text (right), **confidence overlay** (amber/green), pagination, **per-page re-run**, **"Verified" toggle**, **"Next to review"**, copy-to-clipboard.
7. **Export**: DOCX (RTL Urdu) and plain TXT.
8. **Robustness:** Gemini calls have timeout + retry; **orphan recovery** resets docs stuck in `processing` after a restart.

---

## 5. Key decisions & findings (important context)

- **Hybrid UTRNet adds ~nothing on antique Nastaliq.** A/B on the real document
  (1899 *Behes Tansikh*) showed UTRNet↔Gemini agreement ≈ **0.05**. On a full
  page UTRNet originally emitted 1 garbage char; line-segmentation fixed the
  *wiring* (25 lines, real text) but accuracy is still low on this hard script.
  → **Gemini-only is the default.** Hybrid is for modern print only.
- **Cost overrun:** user reported **£6.33** for May 29–30, far above the ~$0.68/170pp
  projection. Cause: **`gemini-2.5-pro`'s "thinking" tokens** (billed as output @ $10/1M),
  which the SDK's `usage_metadata` did **not** report — so measured A/B costs were a
  floor that missed the biggest driver. Part of it was also dev/testing this session.
- **Response:** default model switched **Pro → Flash**; added the **per-run model
  selector** so whoever starts a run sees cost+time and chooses.
- Flash↔Pro agreement on hard scans was low (~0.27) — Flash is cheaper but diverges;
  Flash-Lite cheaper still. The selector lets the user trade off per run.

---

## 6. Model selection (current behaviour)

`ocr_pipeline.MODEL_PRICING` defines selectable Gemini models with per-1M pricing,
rough output-tokens/page, sec/page, and label:

| Model | ~Cost (2pp) | ~Time | Note |
|---|---|---|---|
| `gemini-2.5-pro` | $0.072 | ~2m | most accurate, priciest |
| `gemini-2.5-flash` | $0.009 | ~24s | **recommended default** |
| `gemini-2.5-flash-lite` | $0.002 | ~12s | cheapest |

Upload returns a per-model estimate; the user picks in the modal; `/api/documents/{id}/start`
accepts `{model}`; the choice is stored on `Document.model` and used for processing + re-runs.

---

## 7. Credentials & secrets

Stored in `.env` (gitignored, `chmod 600`). Passwords are SHA-256 hashed in the file.

**Login accounts (plaintext passwords — for testing/handoff):**
- `wafi momin` / `WrBZIvE-fda4lpI-pBh3t9Q`
- `admin` / `C_47HnU-gLjfTOU-_5YFMWo`

**Env vars** (see `.env.example` for the full list): `GEMINI_API_KEY`, `GEMINI_MODEL`,
`GEMINI_ONLY=1`, `USER1_CREDENTIALS`, `USER2_CREDENTIALS`, `SESSION_SECRET`,
`COOKIE_SECURE` (set `1` on HTTPS), `PDF_RENDER_DPI`, storage/DB paths.

> ⚠️ **SECURITY — do first:** The **Gemini API key** and a **GitHub classic token**
> (`ghp_…`, broad scopes) were both pasted in the prior chat in plaintext.
> **Rotate the Gemini key** and **revoke the GitHub token**. Generate a fresh
> `SESSION_SECRET` for any production host.

---

## 8. Deployment status

- **GitHub:** public repo created & pushed. ✅
- **Replit:** repo imported. Hit a Nix `pyexpat` error on the python-3.11 module →
  fixed by switching `.replit` to **python-3.12** (pushed). A divergent-branch on
  pull was resolved with `git reset --hard origin/main`. A run got **stuck on page
  8/10** (hung Gemini call) → fixed by the timeout + orphan-recovery code. **Not
  confirmed fully live yet** — next: pull latest, Run, re-process picking Flash,
  then **Deploy → Reserved VM** (needed for SQLite + file persistence; do NOT use
  Autoscale/Cloud Run without moving DB→Cloud SQL and files→GCS).
- **GCE VM:** a full step-by-step guide was written (VM create → SSH → install →
  systemd service → DuckDNS + Caddy auto-HTTPS). Not executed. (Replit Reserved VM
  is the easier path and was recommended.)
- **Google AI Studio / Antigravity:** clarified these can't *host* the app
  (AI Studio = prototyping; Antigravity = agentic IDE). Hosting still needs
  Reserved VM / GCE / Cloud Run.

---

## 9. OUTSTANDING WORK (was in progress when handed off)

**Add OpenAI (GPT vision) + Anthropic Claude (vision) as selectable OCR providers.**
User chose these two (declined Qwen/Mistral). **Not started in code** — the
`pip install openai anthropic` step was interrupted. Plan:

1. Add `openai` + `anthropic` to `requirements.txt` (pin to installed versions).
2. In `ocr_pipeline.py`: add a **provider abstraction** — extend `MODEL_PRICING`
   with a `provider` field; add models e.g. `gpt-4o`, `gpt-4o-mini`,
   `claude-sonnet-4-6`, `claude-haiku-4-5-20251001` (Anthropic IDs: Opus 4.8 =
   `claude-opus-4-8`, Sonnet 4.6 = `claude-sonnet-4-6`, Haiku 4.5 =
   `claude-haiku-4-5-20251001`).
3. Dispatch in `transcribe_page()` by provider prefix; each does a single
   image→JSON `{text, confidence, notes}` call (reuse `_parse_json`), wrapped in
   timeout+retry. Base64-encode the page PNG for OpenAI/Anthropic.
4. `configure()` reads `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` from env; track
   available providers; **`estimate_usage` only lists models whose provider key
   is set** (so the modal hides providers without keys). Frontend already loops
   over `estimate.models`, so it needs no change.
5. Add the new keys to `.env.example`. Test, commit, push.

**Other open offers (not started):** lower default DPI 200→150 for cost; a hard
per-document spend cap; a head-to-head DeepSeek-OCR test (user explored, undecided).

---

## 10. How to run / verify locally

```bash
cd /Users/muhammadali/urdu-ocr-system
./venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000   # needs .env present
# open http://127.0.0.1:8000  → log in as admin / C_47HnU-gLjfTOU-_5YFMWo
```
Test assets: `/tmp/urdu_test.pdf` (2-page synthetic), real doc at
`/Users/muhammadali/Desktop/Behes Tansikh -1.pdf` (10 pages). Saved transcription
of the real doc: `outputs/Behes_Tansikh_text.txt`.
