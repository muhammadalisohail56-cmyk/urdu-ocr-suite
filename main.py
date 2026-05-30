"""
main.py — Urdu OCR Suite
========================
A private, two-user, production-style web app over the Urdu OCR pipeline.

Features
--------
* Login-gated (exactly two accounts from env vars).
* Drag-and-drop PDF upload with pre-flight Gemini token/cost estimation.
* Asynchronous OCR runs as background tasks.
* Real-time progress over WebSocket (live status bar + rolling ETA).
* Dashboard with inline-editable document metadata.
* Split-screen verification workspace (page image | editable text) with a
  confidence overlay (amber = low, green = high).
* One-click .docx export of the full transcription.

Run locally:  uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import io
import time
import asyncio
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Set, Optional

from dotenv import load_dotenv
from fastapi import (FastAPI, Request, UploadFile, File, Form, Body, Depends,
                     HTTPException, WebSocket, WebSocketDisconnect)
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import ocr_pipeline
from database import init_db, get_db, SessionLocal
from models import (Document, Page, STATUS_PENDING, STATUS_PROCESSING,
                    STATUS_COMPLETED, STATUS_FAILED)
import auth

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocr-suite")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "storage/uploads")
IMAGE_DIR = os.getenv("IMAGE_DIR", "storage/pages")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

templates = Jinja2Templates(directory="templates")


# --------------------------------------------------------------------------- #
# Real-time progress hub (in-memory pub/sub, fine for single-process Replit).
# --------------------------------------------------------------------------- #
class ProgressHub:
    def __init__(self) -> None:
        self._subs: Dict[int, Set[asyncio.Queue]] = defaultdict(set)
        self._last: Dict[int, dict] = {}

    def subscribe(self, doc_id: int) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subs[doc_id].add(q)
        if doc_id in self._last:  # replay latest state to a fresh subscriber
            q.put_nowait(self._last[doc_id])
        return q

    def unsubscribe(self, doc_id: int, q: asyncio.Queue) -> None:
        self._subs[doc_id].discard(q)

    async def publish(self, doc_id: int, message: dict) -> None:
        self._last[doc_id] = message
        for q in list(self._subs.get(doc_id, ())):
            await q.put(message)


hub = ProgressHub()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set — Gemini OCR runs will fail.")
    creds = auth._load_credentials()
    logger.info("Auth configured for %d user(s): %s", len(creds), ", ".join(creds) or "(none!)")

    # Orphan recovery: any doc left 'processing' means a previous run was
    # interrupted (restart/crash). Mark it failed so it can be re-run.
    db = SessionLocal()
    try:
        stuck = db.query(Document).filter(Document.status == STATUS_PROCESSING).all()
        for d in stuck:
            d.status = STATUS_FAILED
            d.error = "Interrupted by a server restart — re-run to retry."
        if stuck:
            db.commit()
            logger.info("Reset %d interrupted document(s) to 'failed'.", len(stuck))
    finally:
        db.close()
    yield


app = FastAPI(title="Urdu OCR Suite", version="1.0.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Auth dependencies
# --------------------------------------------------------------------------- #
def require_user(request: Request) -> str:
    """API guard: 401 JSON if not logged in."""
    user = auth.current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def _redirect_if_anonymous(request: Request) -> Optional[RedirectResponse]:
    if not auth.current_user(request):
        return RedirectResponse("/login", status_code=302)
    return None


# --------------------------------------------------------------------------- #
# Auth routes
# --------------------------------------------------------------------------- #
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if auth.current_user(request):
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    if not auth.verify_login(username, password):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid username or password."},
            status_code=401,
        )
    resp = RedirectResponse("/", status_code=302)
    resp.set_cookie(
        auth.COOKIE_NAME, auth.create_token(username),
        httponly=True, samesite="lax", max_age=auth.TOKEN_TTL_SECONDS,
        secure=os.getenv("COOKIE_SECURE", "0") == "1",
    )
    return resp


@app.get("/logout")
async def logout():
    resp = RedirectResponse("/login", status_code=302)
    resp.delete_cookie(auth.COOKIE_NAME)
    return resp


# --------------------------------------------------------------------------- #
# HTML pages (gated)
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    if (r := _redirect_if_anonymous(request)):
        return r
    return templates.TemplateResponse(
        "dashboard.html", {"request": request, "user": auth.current_user(request)}
    )


@app.get("/documents/{doc_id}", response_class=HTMLResponse)
async def document_workspace(request: Request, doc_id: int):
    if (r := _redirect_if_anonymous(request)):
        return r
    return templates.TemplateResponse(
        "document.html",
        {"request": request, "user": auth.current_user(request), "doc_id": doc_id},
    )


# --------------------------------------------------------------------------- #
# API: upload + estimate + start
# --------------------------------------------------------------------------- #
@app.post("/api/upload")
async def upload(
    request: Request,
    file: UploadFile = File(...),
    user: str = Depends(require_user),
    db: Session = Depends(get_db),
):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file.")
    if raw[:5] != b"%PDF-":
        raise HTTPException(400, "Only PDF files are supported.")

    doc = Document(
        title=os.path.splitext(file.filename or "document")[0],
        filename=file.filename or "document.pdf",
        pdf_path="",  # set below
        status=STATUS_PENDING,
        date_added=datetime.utcnow(),
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    pdf_path = os.path.join(UPLOAD_DIR, f"{doc.id}.pdf")
    with open(pdf_path, "wb") as fh:
        fh.write(raw)
    doc.pdf_path = pdf_path

    # Pre-flight estimate across all selectable models.
    estimate = await run_in_threadpool(ocr_pipeline.estimate_usage, pdf_path)
    doc.num_pages = estimate["num_pages"]
    
    # We no longer calculate a single total est_cost_usd here, since it depends on
    # which agents the user selects. The UI will sum them.
    db.commit()
    db.refresh(doc)

    return {"document": doc.to_dict(), "estimate": estimate}


@app.post("/api/documents/{doc_id}/start")
async def start_processing(doc_id: int, payload: dict = Body(default={}),
                           user: str = Depends(require_user),
                           db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found.")
    if doc.status == STATUS_PROCESSING:
        raise HTTPException(409, "Already processing.")
    if not GEMINI_API_KEY:
        raise HTTPException(503, "GEMINI_API_KEY not configured.")

    selected_models = payload.get("models", [])
    if not selected_models:
        raise HTTPException(400, "You must select at least one OCR model.")
    
    total_cost = 0.0
    for m in selected_models:
        if m not in ocr_pipeline.MODEL_PRICING:
            raise HTTPException(400, f"Unknown model '{m}'.")
        p = ocr_pipeline.MODEL_PRICING[m]
        # Rough re-calculation based on pages
        # (Properly we should re-run estimate_usage or accept the UI's calculation)
        in_tok = 258 + 100 # rough per page
        out_tok = p["out_tokens_per_page"]
        total_cost += (in_tok / 1e6 * p["in"] + out_tok / 1e6 * p["out"]) * doc.num_pages
        
    import json
    doc.selected_agents = json.dumps(selected_models)
    doc.est_cost_usd = round(total_cost, 4)

    doc.status = STATUS_PROCESSING
    doc.pages_ocred = 0
    doc.error = ""
    # Clear any previous pages (re-runs).
    for p in list(doc.pages):
        db.delete(p)
    db.commit()

    asyncio.create_task(_process_document(doc_id))
    return {"status": "started", "document_id": doc_id}


# --------------------------------------------------------------------------- #
# Background OCR worker
# --------------------------------------------------------------------------- #
async def _process_document(doc_id: int) -> None:
    """Render + OCR every page, persisting results and streaming progress."""
    db = SessionLocal()
    try:
        doc = db.get(Document, doc_id)
        if not doc:
            return
        num_pages = doc.num_pages
        pdf_path = doc.pdf_path
        import json
        selected_models = json.loads(doc.selected_agents)
        img_subdir = os.path.join(IMAGE_DIR, str(doc_id))
        os.makedirs(img_subdir, exist_ok=True)
        
        start = time.time()
        await hub.publish(doc_id, {"type": "start", "num_pages": num_pages})

        for i in range(num_pages):
            page_no = i + 1
            image = await run_in_threadpool(ocr_pipeline.render_page, pdf_path, i)
            img_path = os.path.join(img_subdir, f"{page_no}.png")
            await run_in_threadpool(image.save, img_path)

            # Await the multi-agent OCR directly (it's fully async)
            result = await ocr_pipeline.transcribe_page_multi_agent(image, selected_models)

            page = Page(
                document_id=doc_id, page_number=page_no, image_path=img_path,
                text=result["text"], confidence=result["confidence"],
                notes=result["notes"], tokens_json=result["tokens_json"],
                agent_logs=result["agent_logs"],
            )
            db.add(page)
            doc.pages_ocred = page_no
            db.commit()

            elapsed = time.time() - start
            avg = elapsed / page_no
            eta = avg * (num_pages - page_no)
            await hub.publish(doc_id, {
                "type": "progress", "page": page_no, "num_pages": num_pages,
                "percent": round(page_no / num_pages * 100, 1),
                "eta_seconds": round(eta),
                "avg_seconds_per_page": round(avg, 1),
                "confidence": round(result["confidence"], 3),
            })

        doc.status = STATUS_COMPLETED
        db.commit()
        await hub.publish(doc_id, {"type": "complete", "num_pages": num_pages,
                                   "total_seconds": round(time.time() - start)})
    except Exception as exc:  # noqa: BLE001
        logger.exception("Processing failed for doc %s", doc_id)
        doc = db.get(Document, doc_id)
        if doc:
            doc.status = STATUS_FAILED
            doc.error = str(exc)
            db.commit()
        await hub.publish(doc_id, {"type": "error", "message": str(exc)})
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# API: dashboard + document data + inline edits
# --------------------------------------------------------------------------- #
@app.get("/api/documents")
async def list_documents(user: str = Depends(require_user), db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.date_added.desc()).all()
    return {"documents": [d.to_dict() for d in docs]}


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: int, user: str = Depends(require_user),
                       db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found.")
    d = doc.to_dict()
    d["pages"] = [p.to_dict() for p in doc.pages]
    return d


@app.patch("/api/documents/{doc_id}")
async def update_document(doc_id: int, payload: dict, user: str = Depends(require_user),
                          db: Session = Depends(get_db)):
    """Inline metadata edits from the dashboard (title/author/year)."""
    doc = db.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found.")
    for field in ("title", "author", "year"):
        if field in payload:
            setattr(doc, field, str(payload[field]))
    db.commit()
    db.refresh(doc)
    return {"document": doc.to_dict()}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: int, user: str = Depends(require_user),
                          db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found.")
    db.delete(doc)
    db.commit()
    return {"deleted": doc_id}


@app.get("/api/documents/{doc_id}/pages/{page_no}")
async def get_page(doc_id: int, page_no: int, user: str = Depends(require_user),
                   db: Session = Depends(get_db)):
    page = (db.query(Page)
            .filter(Page.document_id == doc_id, Page.page_number == page_no)
            .first())
    if not page:
        raise HTTPException(404, "Page not found.")
    return page.to_dict()


@app.patch("/api/pages/{page_id}")
async def update_page(page_id: int, payload: dict, user: str = Depends(require_user),
                      db: Session = Depends(get_db)):
    """Save edits to a page's reconciled text and/or its proofread status."""
    page = db.get(Page, page_id)
    if not page:
        raise HTTPException(404, "Page not found.")
    if "text" in payload:
        page.text = str(payload["text"])
    if "verified" in payload:
        page.verified = bool(payload["verified"])
    db.commit()
    return page.to_dict()


@app.post("/api/pages/{page_id}/rerun")
async def rerun_page(page_id: int, user: str = Depends(require_user),
                     db: Session = Depends(get_db)):
    """Re-render and re-OCR a single page (e.g. a bad page, or after tuning).
    Runs inline (one page) and resets the page's verified flag."""
    page = db.get(Page, page_id)
    if not page:
        raise HTTPException(404, "Page not found.")
    if not GEMINI_API_KEY:
        raise HTTPException(503, "GEMINI_API_KEY not configured.")
    doc = db.get(Document, page.document_id)

    import json
    selected_models = json.loads(doc.selected_agents)
    
    image = await run_in_threadpool(ocr_pipeline.render_page, doc.pdf_path, page.page_number - 1)
    img_subdir = os.path.join(IMAGE_DIR, str(doc.id))
    os.makedirs(img_subdir, exist_ok=True)
    img_path = os.path.join(img_subdir, f"{page.page_number}.png")
    await run_in_threadpool(image.save, img_path)

    result = await ocr_pipeline.transcribe_page_multi_agent(image, selected_models)
    page.image_path = img_path
    page.text = result["text"]
    page.confidence = result["confidence"]
    page.notes = result["notes"]
    page.tokens_json = result["tokens_json"]
    page.agent_logs = result["agent_logs"]
    page.verified = False  # re-run supersedes any prior human verification
    db.commit()
    return page.to_dict()


@app.get("/api/pages/{page_id}/image")
async def page_image(page_id: int, user: str = Depends(require_user),
                     db: Session = Depends(get_db)):
    page = db.get(Page, page_id)
    if not page or not page.image_path or not os.path.exists(page.image_path):
        raise HTTPException(404, "Image not found.")
    with open(page.image_path, "rb") as fh:
        data = fh.read()
    return StreamingResponse(io.BytesIO(data), media_type="image/png")


# --------------------------------------------------------------------------- #
# API: DOCX export
# --------------------------------------------------------------------------- #
@app.get("/api/documents/{doc_id}/export")
async def export_docx(doc_id: int, user: str = Depends(require_user),
                      db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found.")
    data = await run_in_threadpool(_build_docx, doc)
    fname = f"{(doc.title or 'document').strip().replace(' ', '_')}.docx"
    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.get("/api/documents/{doc_id}/export.txt")
async def export_txt(doc_id: int, user: str = Depends(require_user),
                     db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found.")
    parts = [f"--- صفحہ {p.page_number} ---\n{p.text}" for p in doc.pages]
    data = ("\n\n".join(parts)).encode("utf-8")
    fname = f"{(doc.title or 'document').strip().replace(' ', '_')}.txt"
    return StreamingResponse(
        io.BytesIO(data), media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


def _build_docx(doc: Document) -> bytes:
    """Compile all pages (in order) into a right-to-left Urdu .docx."""
    from docx import Document as Docx
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Pt

    d = Docx()
    style = d.styles["Normal"]
    style.font.name = "Jameel Noori Nastaleeq"
    style.font.size = Pt(14)

    def _rtl(par):
        par.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        pPr = par._p.get_or_add_pPr()
        bidi = pPr.makeelement(qn("w:bidi"), {})
        pPr.append(bidi)

    title = d.add_heading(doc.title or "Untitled", level=0)
    _rtl(title)
    meta = d.add_paragraph(
        f"Author: {doc.author or '—'}   |   Year: {doc.year or '—'}   |   "
        f"Pages: {doc.pages_ocred}/{doc.num_pages}"
    )
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for page in doc.pages:
        d.add_page_break()
        hdr = d.add_paragraph(f"— صفحہ {page.page_number} —")
        hdr.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for line in (page.text or "").split("\n"):
            _rtl(d.add_paragraph(line))

    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# WebSocket: live progress
# --------------------------------------------------------------------------- #
@app.websocket("/ws/documents/{doc_id}/progress")
async def ws_progress(websocket: WebSocket, doc_id: int):
    # Authenticate via the session cookie sent with the WS handshake.
    if not auth.current_user_ws(websocket):
        await websocket.close(code=4401)
        return
    await websocket.accept()
    q = hub.subscribe(doc_id)
    try:
        while True:
            message = await q.get()
            await websocket.send_json(message)
            if message.get("type") in ("complete", "error"):
                # keep the socket open briefly so the client renders the final state
                pass
    except WebSocketDisconnect:
        pass
    finally:
        hub.unsubscribe(doc_id, q)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "gemini_configured": bool(GEMINI_API_KEY),
            "mode": "multi-agent"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=os.getenv("HOST", "0.0.0.0"),
                port=int(os.getenv("PORT", "8000")), reload=bool(os.getenv("RELOAD")))
