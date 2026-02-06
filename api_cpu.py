import os
import tempfile
import time
from typing import Any, Dict, List, Optional

import cv2
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import PlainTextResponse
from paddleocr import PaddleOCR
from pdf2image import convert_from_path

app = FastAPI(title="PaddleOCR CPU API (Image + PDF)")

ocr = None

@app.on_event("startup")
def startup():
    global ocr
    ocr = PaddleOCR(lang="tr", use_gpu=False, use_angle_cls=True)

@app.get("/health")
def health():
    return {"ok": True}

def conf_stats(lines):
    if not lines:
        return (0.0, 0.0, 0.0)
    cs = [float(l["confidence"]) for l in lines]
    return (sum(cs) / len(cs), min(cs), max(cs))

def run_ocr(image_path: str, min_conf: float) -> List[Dict[str, Any]]:
    result = ocr.ocr(image_path, cls=True)
    lines: List[Dict[str, Any]] = []
    if result and result[0] is not None:
        for item in result[0]:
            box = item[0]
            text = item[1][0]
            conf = float(item[1][1])
            if conf >= min_conf:
                lines.append({"box": box, "text": text, "confidence": conf})
    return lines

def lines_to_txt(lines, filename: str, duration_sec: float) -> str:
    avg_c, min_c, max_c = conf_stats(lines)
    out = []
    out.append(f"filename:\t{filename}")
    out.append(f"duration_ms:\t{duration_sec*1000:.1f}")
    out.append(f"lines:\t{len(lines)}")
    out.append(f"conf_avg:\t{avg_c:.4f}")
    out.append(f"conf_min:\t{min_c:.4f}")
    out.append(f"conf_max:\t{max_c:.4f}")
    out.append("")
    for l in lines:
        out.append(f"{float(l['confidence']):.4f}\t{l['text'].replace(chr(10),' ').strip()}")
    out.append("")
    return "\n".join(out)

def pages_to_txt(pages_out, filename: str, duration_sec: float) -> str:
    all_lines = []
    for p in pages_out:
        all_lines.extend(p["results"])
    avg_c, min_c, max_c = conf_stats(all_lines)
    total_lines = len(all_lines)

    out = []
    out.append(f"filename:\t{filename}")
    out.append(f"type:\tpdf")
    out.append(f"duration_ms:\t{duration_sec*1000:.1f}")
    out.append(f"pages:\t{len(pages_out)}")
    out.append(f"total_lines:\t{total_lines}")
    out.append(f"conf_avg:\t{avg_c:.4f}")
    out.append(f"conf_min:\t{min_c:.4f}")
    out.append(f"conf_max:\t{max_c:.4f}")
    out.append("")

    for p in pages_out:
        page_num = p["page"]
        lines = p["results"]
        p_avg, p_min, p_max = conf_stats(lines)
        out.append(f"--- page {page_num} ---")
        out.append(f"lines:\t{len(lines)}\tconf_avg:\t{p_avg:.4f}\tconf_min:\t{p_min:.4f}\tconf_max:\t{p_max:.4f}")
        for l in lines:
            out.append(f"{float(l['confidence']):.4f}\t{l['text'].replace(chr(10),' ').strip()}")
        out.append("")
    return "\n".join(out)

@app.post("/ocr")
async def ocr_image(
    file: UploadFile = File(...),
    min_conf: float = Query(0.5, ge=0.0, le=1.0),
):
    suffix = os.path.splitext(file.filename)[1].lower() or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        tmp.write(await file.read())

    img = cv2.imread(tmp_path)
    if img is None:
        try: os.remove(tmp_path)
        except: pass
        return {"error": "Image could not be decoded", "filename": file.filename}

    t0 = time.perf_counter()
    lines = run_ocr(tmp_path, min_conf=min_conf)
    dt = time.perf_counter() - t0

    try: os.remove(tmp_path)
    except: pass

    avg_c, min_c, max_c = conf_stats(lines)
    return {
        "filename": file.filename,
        "type": "image",
        "duration_ms": round(dt * 1000, 2),
        "count": len(lines),
        "conf_avg": round(avg_c, 4),
        "conf_min": round(min_c, 4),
        "conf_max": round(max_c, 4),
        "results": lines,
    }

@app.post("/ocr_txt", response_class=PlainTextResponse)
async def ocr_image_txt(
    file: UploadFile = File(...),
    min_conf: float = Query(0.5, ge=0.0, le=1.0),
):
    suffix = os.path.splitext(file.filename)[1].lower() or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        tmp.write(await file.read())

    img = cv2.imread(tmp_path)
    if img is None:
        try: os.remove(tmp_path)
        except: pass
        return "ERROR: Image could not be decoded\n"

    t0 = time.perf_counter()
    lines = run_ocr(tmp_path, min_conf=min_conf)
    dt = time.perf_counter() - t0

    try: os.remove(tmp_path)
    except: pass

    return lines_to_txt(lines, file.filename, dt)

@app.post("/ocr_pdf")
async def ocr_pdf(
    file: UploadFile = File(...),
    min_conf: float = Query(0.5, ge=0.0, le=1.0),
    dpi: int = Query(200, ge=72, le=400),
    first_page: Optional[int] = Query(None, ge=1),
    last_page: Optional[int] = Query(None, ge=1),
):
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Please upload a PDF file (.pdf)", "filename": file.filename}

    t0 = time.perf_counter()

    with tempfile.TemporaryDirectory() as tdir:
        pdf_path = os.path.join(tdir, "input.pdf")
        with open(pdf_path, "wb") as f:
            f.write(await file.read())

        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=first_page,
            last_page=last_page,
        )

        pages_out = []
        base_page = first_page or 1

        for idx, pil_img in enumerate(images):
            page_no = base_page + idx
            img_path = os.path.join(tdir, f"page_{page_no:04d}.jpg")
            pil_img.save(img_path, "JPEG", quality=95)

            lines = run_ocr(img_path, min_conf=min_conf)
            pages_out.append({"page": page_no, "count": len(lines), "results": lines})

    dt = time.perf_counter() - t0

    # global stats
    all_lines = []
    for p in pages_out:
        all_lines.extend(p["results"])
    avg_c, min_c, max_c = conf_stats(all_lines)

    return {
        "filename": file.filename,
        "type": "pdf",
        "dpi": dpi,
        "duration_ms": round(dt * 1000, 2),
        "pages": len(pages_out),
        "total_count": len(all_lines),
        "conf_avg": round(avg_c, 4),
        "conf_min": round(min_c, 4),
        "conf_max": round(max_c, 4),
        "results_by_page": pages_out,
    }

@app.post("/ocr_pdf_txt", response_class=PlainTextResponse)
async def ocr_pdf_txt(
    file: UploadFile = File(...),
    min_conf: float = Query(0.5, ge=0.0, le=1.0),
    dpi: int = Query(200, ge=72, le=400),
    first_page: Optional[int] = Query(None, ge=1),
    last_page: Optional[int] = Query(None, ge=1),
):
    if not file.filename.lower().endswith(".pdf"):
        return "ERROR: Please upload a PDF file (.pdf)\n"

    t0 = time.perf_counter()

    with tempfile.TemporaryDirectory() as tdir:
        pdf_path = os.path.join(tdir, "input.pdf")
        with open(pdf_path, "wb") as f:
            f.write(await file.read())

        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=first_page,
            last_page=last_page,
        )

        pages_out = []
        base_page = first_page or 1

        for idx, pil_img in enumerate(images):
            page_no = base_page + idx
            img_path = os.path.join(tdir, f"page_{page_no:04d}.jpg")
            pil_img.save(img_path, "JPEG", quality=95)

            lines = run_ocr(img_path, min_conf=min_conf)
            pages_out.append({"page": page_no, "results": lines})

    dt = time.perf_counter() - t0
    return pages_to_txt(pages_out, file.filename, dt)
