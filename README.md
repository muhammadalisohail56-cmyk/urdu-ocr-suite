# Urdu OCR System — Hybrid UTRNet + Gemini

A FastAPI backend for Urdu OCR that combines a **local PyTorch UTRNet model**
with **Google's Gemini vision API** in a dual-engine reconciliation pipeline.
Each uploaded image is transcribed by both engines in parallel, and Gemini then
adjudicates the two candidate transcriptions against the original image to
produce a single, corrected result.

---

## Architecture Overview

> **Default mode is Gemini-Pro-only** (`GEMINI_ONLY=1`): one Gemini call per
> page, no UTRNet, no reconciliation. A/B testing on aged Nastaliq scans showed
> UTRNet agreed with Gemini only ~0.05 (char similarity) — i.e. it added no
> usable signal on that material — so the default is the cheapest, simplest path
> with equivalent output. The full hybrid below is **opt-in** via `GEMINI_ONLY=0`
> and is worthwhile for **modern printed Urdu**, which is UTRNet's trained domain.
> See `ab_compare.py` to measure agreement/cost on your own corpus.

The full hybrid pipeline (when enabled) runs in two stages:

```
                        ┌─────────────────────────┐
                        │   POST /ocr  (image)     │
                        └────────────┬────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 │           Stage 1 — parallel           │
                 ▼                                        ▼
      ┌─────────────────────┐                ┌─────────────────────┐
      │  Local UTRNet        │                │  Gemini vision       │
      │  (PyTorch, on-device)│                │  (image → Urdu text) │
      │  → utrnet_text       │                │  → gemini_text       │
      └──────────┬───────────┘                └──────────┬──────────┘
                 │                                        │
                 └───────────────────┬────────────────────┘
                                     ▼
                        ┌─────────────────────────┐
                        │   Stage 2 — reconcile    │
                        │  Gemini receives image + │
                        │  BOTH transcriptions and │
                        │  returns the corrected   │
                        │  final text as JSON      │
                        └────────────┬────────────┘
                                     ▼
                        ┌─────────────────────────┐
                        │  reconciled_text +       │
                        │  confidence + notes      │
                        └─────────────────────────┘
```

**Why two engines?** UTRNet is a high-resolution Urdu recognizer (HRNet feature
extractor → DBiLSTM → CTC) that is strong on Urdu glyph shapes and ligatures but
is a fixed-vocabulary model. Gemini brings broad language priors and contextual
understanding. Reconciling the two — having Gemini adjudicate token-by-token
against the source image — typically beats either engine alone, especially on
degraded scans.

**Line segmentation (important).** UTRNet is a *line-level* recognizer: it was
trained on single cropped lines (height 32). Handing it a whole page makes it
read a squashed full page as one "line" and emit garbage. So before UTRNet runs,
`line_segmenter.py` splits the page into individual line crops (horizontal
projection profile with local-background subtraction + adaptive thresholding,
robust to aged/uneven scans). Each line is recognized separately and the results
are joined top-to-bottom. This is what makes UTRNet a meaningful contributor on
documents rather than dead weight. Toggle with `UTRNET_SEGMENT_LINES`. Note:
Gemini still runs **once per page** for transcription and **once** for
reconciliation — line segmentation does **not** multiply Gemini API cost.

**Graceful degradation:** the two engines fail independently.
- If **UTRNet** is unavailable (missing weights, load error), the API returns
  Gemini's transcription alone (`utrnet_available: false`).
- If **Gemini's first pass** fails, the request errors with `502` — Gemini is
  also the adjudicator, so it is required.

---

## Vendored UTRNet Architecture

The released UTRNet checkpoint (`best_norm_ED.pth`) is a **state_dict** — just the
learned weights, with no architecture definition. To load it, the matching
network code must be present. This project **vendors the official UTRNet source**
directly from the
[UTRNet repository](https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition):

| Vendored path        | Purpose                                                    |
|----------------------|------------------------------------------------------------|
| `model.py`           | The `Model` class assembling feature/sequence/prediction stages |
| `modules/`           | Feature extractors (incl. `modules/cnn/` — HRNet, etc.), sequence models, dropout, prediction |
| `utils.py`           | `CTCLabelConverter` used to decode CTC output into text    |

`utr_wrapper.py` imports `Model` from `model.py` and `CTCLabelConverter` from
`utils.py`, builds the exact **UTRNet-Large** configuration used by the
repository's `read.py` inference script, and loads the state_dict with a strict
match:

- Feature extraction: **HRNet** (`output_channel = 32`)
- Sequence modeling: **DBiLSTM**
- Prediction head: **CTC**
- `hidden_size = 256`, input `32 × 400`, grayscale (`input_channel = 1`)

Preprocessing faithfully mirrors `read.py`, including the **left-right flip**
(Urdu is right-to-left and the model was trained on flipped crops),
aspect-ratio-preserving resize, and `NormalizePAD` (scale to `[-1, 1]` then
right-pad with the border column). `NormalizePAD` is reimplemented inline in
`utr_wrapper.py` to avoid vendoring `dataset.py` and its heavy `lmdb`/`natsort`
dependencies.

> **Note:** The checkpoint was saved from a `DataParallel`-wrapped model, so the
> wrapper strips the `module.` key prefix before `load_state_dict`. A successful
> strict load (no missing/unexpected keys) confirms the architecture matches.

---

## Project Layout

```
urdu-ocr-system/
├── main.py                         # FastAPI app (endpoints + reconciliation)
├── utr_wrapper.py                  # UTRNet loading + inference wrapper
├── line_segmenter.py               # lightweight text-line detector (page → line crops)
├── requirements.txt
├── README.md
├── model.py                        # ← vendored from UTRNet repo
├── utils.py                        # ← vendored from UTRNet repo
├── modules/                        # ← vendored from UTRNet repo (+ cnn/)
├── UrduGlyphs.txt                  # charset (one glyph per line)
└── saved_models/
    └── UTRNet-Large/
        └── best_norm_ED.pth        # UTRNet state_dict checkpoint (~181 MB)
```

---

## Requirements

- **Python 3.13** (the pinned `torch==2.8.0` / `torchvision==0.23.0` wheels do
  not yet build for Python 3.14).
- A **Gemini API key** ([Google AI Studio](https://aistudio.google.com/apikey)).
- The UTRNet **checkpoint** (`best_norm_ED.pth`) and **charset** (`UrduGlyphs.txt`),
  both available from the UTRNet repository / its releases.

---

## Environment Variables

| Variable              | Required | Default                                          | Description |
|-----------------------|----------|--------------------------------------------------|-------------|
| `GEMINI_API_KEY`      | **Yes**  | _(none)_                                         | Google Generative AI key. Without it, `/ocr` returns `503`. |
| `GEMINI_MODEL`        | No       | `gemini-2.5-pro`                                 | Gemini model used for transcription (and reconciliation in hybrid mode). Must be multimodal. |
| `GEMINI_ONLY`         | No       | `1` (on)                                         | **Default mode**: one Gemini call per page, UTRNet + reconciliation skipped. Set `0` to enable the full hybrid pipeline (worthwhile for modern printed Urdu). |
| `UTRNET_MODEL_PATH`   | No       | `saved_models/UTRNet-Large/best_norm_ED.pth`     | Path to the UTRNet state_dict checkpoint. |
| `UTRNET_CHARSET_PATH` | No       | `UrduGlyphs.txt`                                 | Path to the glyph list (one glyph per line). |
| `UTRNET_DEVICE`       | No       | auto (`cuda` → `mps` → `cpu`)                    | Force the inference device. |
| `UTRNET_SEGMENT_LINES`| No       | `1` (on)                                         | Segment each page into line crops before UTRNet (its native input). Set `0` to feed UTRNet the whole page (not recommended for documents). |
| `PDF_RENDER_DPI`      | No       | `200`                                            | DPI used to rasterize PDF pages before OCR. |
| `PDF_MAX_PAGES`       | No       | `50`                                             | Safety cap on PDF page count (returns `413` if exceeded). |
| `HOST`                | No       | `0.0.0.0`                                        | Bind host (when run via `python main.py`). |
| `PORT`                | No       | `8000`                                           | Bind port (when run via `python main.py`). |
| `RELOAD`              | No       | _(unset)_                                        | Set to any value to enable auto-reload via `python main.py`. |

> **Security:** treat `GEMINI_API_KEY` as a secret. Do not commit it. Prefer a
> local `.env` file (loaded automatically via `python-dotenv`) or your shell
> environment, and rotate the key if it is ever exposed.

---

## Setup & Run

### 1. Create the virtual environment and install dependencies

```bash
cd /Users/muhammadali/urdu-ocr-system

# Create a Python 3.13 virtual environment
python3.13 -m venv venv

# Install dependencies into it
./venv/bin/pip install --upgrade pip

# Default (Gemini-only) — lean, ~80 MB, installs in seconds:
./venv/bin/pip install -r requirements.txt

# OR full hybrid (adds the ~1 GB PyTorch/UTRNet stack) — only if GEMINI_ONLY=0:
# ./venv/bin/pip install -r requirements-hybrid.txt
```

> **Dependency split:** `requirements.txt` is the lean base for the default
> Gemini-only mode (no ML stack). `requirements-hybrid.txt` includes the base
> **plus** torch/torchvision/numpy/matplotlib for the local UTRNet path. The app
> never imports the ML stack at startup, so the lean install runs fine; if you
> set `GEMINI_ONLY=0` without the hybrid deps, the first OCR run fails with a
> clear message telling you to install `requirements-hybrid.txt`. On Replit, the
> lean base means much faster cold builds.

### 2. Place the model assets

Ensure these exist (already in place if you've run the setup):

```
saved_models/UTRNet-Large/best_norm_ED.pth
UrduGlyphs.txt
```

### 3. Configure the API key

Either export it in your shell:

```bash
export GEMINI_API_KEY="your-key-here"
```

…or create a `.env` file in the project root (auto-loaded at startup):

```dotenv
GEMINI_API_KEY=your-key-here
# Optional overrides:
# GEMINI_MODEL=gemini-2.5-pro
# UTRNET_MODEL_PATH=saved_models/UTRNet-Large/best_norm_ED.pth
# UTRNET_CHARSET_PATH=UrduGlyphs.txt
# UTRNET_DEVICE=cpu
```

### 4. Start the Uvicorn server

```bash
./venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

On a successful start you should see UTRNet warm up in the logs:

```
INFO:urdu-ocr:Gemini configured with model 'gemini-2.5-pro'.
INFO:utrnet:UTRNet loaded (device=cpu, num_class=182, HRNet/DBiLSTM/CTC)
INFO:     Application startup complete.
```

---

## API

### `GET /health`

Liveness probe and configuration check.

```bash
curl -s http://127.0.0.1:8000/health
```

```json
{ "status": "ok", "gemini_configured": true, "gemini_model": "gemini-2.5-pro" }
```

### `POST /ocr`

Multipart upload of **either an image** (PNG/JPG/…) **or a PDF**. The file type
is detected automatically from its content (magic bytes), so the extension/MIME
type does not have to be correct.

```bash
# Image
curl -s -X POST http://127.0.0.1:8000/ocr -F "file=@/path/to/urdu_image.png"

# PDF (multi-page)
curl -s -X POST http://127.0.0.1:8000/ocr -F "file=@/path/to/urdu_doc.pdf"
```

**Image response** (`type: "image"`):

```json
{
  "type": "image",
  "reconciled_text": "اردو",
  "confidence": 1.0,
  "notes": "Both transcriptions were correct.",
  "utrnet_text": "اردو",
  "gemini_text": "اردو",
  "utrnet_confidence": 0.9999712,
  "utrnet_available": true
}
```

| Field               | Description |
|---------------------|-------------|
| `type`              | `"image"` for a single image, `"pdf"` for a PDF (see below). |
| `reconciled_text`   | Final adjudicated transcription. |
| `confidence`        | Gemini's confidence in the reconciled output (`0`–`1`). |
| `notes`             | Brief note on what was corrected during reconciliation. |
| `utrnet_text`       | Raw local UTRNet transcription. |
| `gemini_text`       | Raw Gemini first-pass transcription. |
| `utrnet_confidence` | Mean CTC softmax confidence over kept timesteps. |
| `utrnet_lines`      | Number of line crops UTRNet read (when line segmentation is on; `0` = whole-image fallback, `null` = UTRNet unavailable). |
| `utrnet_available`  | `false` if UTRNet failed and the result is Gemini-only. |

**PDF response** (`type: "pdf"`): each page is rasterized at `PDF_RENDER_DPI`
and run through the same hybrid pipeline. The response contains a combined
`full_text` plus a `pages` array, where each entry is the image response above
with an added 1-based `page` number.

```json
{
  "type": "pdf",
  "num_pages": 2,
  "full_text": "--- Page 1 ---\nاردو\n\n--- Page 2 ---\nکتاب",
  "pages": [
    { "type": "page", "page": 1, "reconciled_text": "اردو",  "confidence": 1.0, "...": "..." },
    { "type": "page", "page": 2, "reconciled_text": "کتاب", "confidence": 1.0, "...": "..." }
  ]
}
```

> PDF pages are processed **sequentially** to stay within Gemini rate limits, so
> a long document takes roughly `num_pages × (single-image latency)`. PDFs over
> `PDF_MAX_PAGES` pages are rejected with `413`.

Interactive docs (Swagger UI) are available at **`/docs`**.

---

## Troubleshooting

| Symptom | Cause / Fix |
|--------|-------------|
| `503` on `/ocr` | `GEMINI_API_KEY` not set. Export it or add it to `.env`. |
| `502 … model is not found for API version` | The configured Gemini model name is retired/unavailable for your key. List available models and set `GEMINI_MODEL` to a supported multimodal model (e.g. `gemini-2.5-pro`). |
| Log warns *"UTRNet warmup failed"* | Checkpoint or charset path wrong, or the architecture doesn't match. Verify `UTRNET_MODEL_PATH` / `UTRNET_CHARSET_PATH`; the API still runs in Gemini-only mode. |
| `Could not find a version that satisfies torch==…` | You're on Python 3.14+. Use Python 3.13 for the venv. |
| `RuntimeError: Error(s) in loading state_dict` | The vendored architecture and the checkpoint disagree. Ensure the vendored `model.py` / `modules/` come from the same UTRNet revision that produced your checkpoint. |

---

## Credits

UTRNet: *"UTRNet: High-Resolution Urdu Text Recognition In Printed Documents"*,
Abdur Rahman, Arjun Ghosh, Chetan Arora — ICDAR 2023.
[Repository](https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition)
· licensed CC BY-NC 4.0 (note the **non-commercial** restriction on the UTRNet
model and vendored source).
