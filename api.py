import os
import tempfile
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from fastapi.responses import PlainTextResponse
import time

app = FastAPI(title="PaddleOCR GPU API (Image + PDF)")

ocr = None

@app.on_event("startup")
def startup():
    global ocr
    ocr = PaddleOCR(lang="tr", use_gpu=True, use_angle_cls=True)

@app.get("/health")
def health():
    return {"ok": True}
def conf_stats(lines):
    if not lines:
        return (0.0, 0.0, 0.0)
    cs = [float(l["confidence"]) for l in lines]
    return (sum(cs) / len(cs), min(cs), max(cs))

def lines_to_text_with_scores(lines, duration_sec, header_title="RESULT"):
    avg_c, min_c, max_c = conf_stats(lines)

    out = []
    out.append(f"## {header_title}")
    out.append(f"duration_ms:\t{duration_sec * 1000:.1f}")
    out.append(f"lines:\t{len(lines)}")
    out.append(f"conf_avg:\t{avg_c:.4f}")
    out.append(f"conf_min:\t{min_c:.4f}")
    out.append(f"conf_max:\t{max_c:.4f}")
    out.append("")  # boş satır

    # Satır satır: confidence + text
    # TSV gibi: 0.9321 <tab> metin
    for l in lines:
        c = float(l["confidence"])
        t = l["text"].replace("\n", " ").strip()
        out.append(f"{c:.4f}\t{t}")

    out.append("")  # final newline
    return "\n".join(out)

def pages_to_text_with_scores(results_by_page, total_duration_sec):
    # results_by_page: [{"page": n, "results": [lines...]}, ...]
    total_lines = sum(len(p["results"]) for p in results_by_page)

    # global stats
    all_lines = []
    for p in results_by_page:
        all_lines.extend(p["results"])
    avg_c, min_c, max_c = conf_stats(all_lines)

    out = []
    out.append("## PDF RESULT")
    out.append(f"duration_ms:\t{total_duration_sec * 1000:.1f}")
    out.append(f"pages:\t{len(results_by_page)}")
    out.append(f"total_lines:\t{total_lines}")
    out.append(f"conf_avg:\t{avg_c:.4f}")
    out.append(f"conf_min:\t{min_c:.4f}")
    out.append(f"conf_max:\t{max_c:.4f}")
    out.append("")

    for p in results_by_page:
        page_num = p["page"]
        lines = p["results"]
        p_avg, p_min, p_max = conf_stats(lines)
        out.append(f"--- page {page_num} ---")
        out.append(f"lines:\t{len(lines)}\tconf_avg:\t{p_avg:.4f}\tconf_min:\t{p_min:.4f}\tconf_max:\t{p_max:.4f}")
        for l in lines:
            out.append(f"{float(l['confidence']):.4f}\t{l['text'].replace(chr(10),' ').strip()}")
        out.append("")

    return "\n".join(out)
def lines_to_text(lines):
    # Basit: her satırı alt alta
    # İstersen confidence ekleyebiliriz.
    return "\n".join([l["text"] for l in lines])
def pages_to_text(results_by_page):
    chunks = []
    for p in results_by_page:
        chunks.append(f"--- page {p['page']} ---")
        chunks.append("\n".join([l["text"] for l in p["results"]]))
    return "\n".join(chunks).strip() + "\n"
def run_ocr_on_image_path(image_path: str, min_conf: float) -> List[Dict[str, Any]]:
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

def pil_to_tmp_jpg(pil_img, dir_path: str, page_index: int) -> str:
    # pil_img RGB gelir; OpenCV BGR ister ama PaddleOCR path üzerinden okuyabildiği için JPG yazıyoruz
    out_path = os.path.join(dir_path, f"page_{page_index+1:04d}.jpg")
    pil_img.save(out_path, "JPEG", quality=95)
    return out_path
@app.post("/ocr_pdf_txt", response_class=PlainTextResponse)
async def ocr_pdf_txt(
    file: UploadFile = File(...),
    min_conf: float = Query(0.5, ge=0.0, le=1.0),
    dpi: int = Query(200, ge=72, le=400),
    first_page: Optional[int] = Query(None, ge=1),
    last_page: Optional[int] = Query(None, ge=1),
):
    t0 = time.perf_counter()

    if not file.filename.lower().endswith(".pdf"):
        return "ERROR: Please upload a PDF file (.pdf)\n"

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
        for idx, pil_img in enumerate(images):
            img_path = pil_to_tmp_jpg(pil_img, tdir, idx)
            lines = run_ocr_on_image_path(img_path, min_conf=min_conf)
            pages_out.append({
                "page": (first_page or 1) + idx,
                "results": lines
            })

        dt = time.perf_counter() - t0
        return pages_to_text_with_scores(pages_out, dt)

@app.post("/ocr")
async def ocr_image(
    file: UploadFile = File(...),
    min_conf: float = Query(0.5, ge=0.0, le=1.0),
) -> Dict[str, Any]:
    # Görsel upload endpoint’i (jpg/png)
    suffix = os.path.splitext(file.filename)[1].lower() or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        tmp.write(await file.read())

    img = cv2.imread(tmp_path)
    if img is None:
        try: os.remove(tmp_path)
        except: pass
        return {"error": "Image could not be decoded"}

    lines = run_ocr_on_image_path(tmp_path, min_conf=min_conf)

    try: os.remove(tmp_path)
    except: pass

    return {"filename": file.filename, "type": "image", "count": len(lines), "results": lines}

@app.post("/ocr_pdf")
async def ocr_pdf(
    file: UploadFile = File(...),
    min_conf: float = Query(0.5, ge=0.0, le=1.0),
    dpi: int = Query(200, ge=72, le=400),
    first_page: Optional[int] = Query(None, ge=1),
    last_page: Optional[int] = Query(None, ge=1),
) -> Dict[str, Any]:
    # PDF upload endpoint’i
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Please upload a PDF file (.pdf)"}

    with tempfile.TemporaryDirectory() as tdir:
        pdf_path = os.path.join(tdir, "input.pdf")
        with open(pdf_path, "wb") as f:
            f.write(await file.read())

        # PDF -> PIL image list
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=first_page,
            last_page=last_page,
        )

        pages_out: List[Dict[str, Any]] = []
        for idx, pil_img in enumerate(images):
            img_path = pil_to_tmp_jpg(pil_img, tdir, idx)
            lines = run_ocr_on_image_path(img_path, min_conf=min_conf)
            pages_out.append({
                "page": (first_page or 1) + idx,
                "count": len(lines),
                "results": lines
            })

        total = sum(p["count"] for p in pages_out)
        return {
            "filename": file.filename,
            "type": "pdf",
            "dpi": dpi,
            "pages": len(pages_out),
            "total_count": total,
            "results_by_page": pages_out
        }
@app.post("/ocr_txt", response_class=PlainTextResponse)
async def ocr_image_txt(
    file: UploadFile = File(...),
    min_conf: float = Query(0.5, ge=0.0, le=1.0),
):
    t0 = time.perf_counter()

    suffix = os.path.splitext(file.filename)[1].lower() or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        tmp.write(await file.read())

    img = cv2.imread(tmp_path)
    if img is None:
        try: os.remove(tmp_path)
        except: pass
        return "ERROR: Image could not be decoded\n"

    lines = run_ocr_on_image_path(tmp_path, min_conf=min_conf)

    try: os.remove(tmp_path)
    except: pass

    dt = time.perf_counter() - t0
    return lines_to_text_with_scores(lines, dt, header_title=f"IMAGE: {file.filename}")
