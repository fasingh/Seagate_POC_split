import os
import re
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageDraw

import easyocr


# -----------------------------
# Helpers
# -----------------------------
KEYWORDS = [
    "invoice", "tax invoice", "statement", "receipt",
    "offer letter", "application", "policy", "confidential",
    "page", "total", "amount", "due", "bill to", "ship to",
    "signature", "signed", "date"
]

PAGE_PATTERNS = [
    re.compile(r"\bpage\s*(\d+)\s*(?:of|/)\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\bpage\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\b(\d+)\s*/\s*(\d+)\b")
]


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def polygon_to_bbox(poly: List[List[float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def to_native(obj: Any) -> Any:
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def extract_region_results(results: List[Dict[str, Any]], h: int, region: str) -> List[Dict[str, Any]]:
    """
    region: 'top' or 'bottom' (20% bands)
    """
    band = 0.20
    if region == "top":
        y_min, y_max = 0, int(h * band)
    else:
        y_min, y_max = int(h * (1 - band)), h

    out = []
    for r in results:
        y1, y2 = r["bbox"][1], r["bbox"][3]
        cy = (y1 + y2) / 2.0
        if y_min <= cy <= y_max:
            out.append(r)
    return out


def join_text(results: List[Dict[str, Any]], max_lines: int = 20) -> str:
    """
    Just join by reading order approximation (top-to-bottom).
    """
    # sort by y then x
    sorted_r = sorted(results, key=lambda r: (r["bbox"][1], r["bbox"][0]))
    lines = [r["text"] for r in sorted_r if r["text"].strip()]
    if max_lines is not None:
        lines = lines[:max_lines]
    return "\n".join(lines)


def find_keywords(text: str) -> List[str]:
    t = text.lower()
    found = []
    for k in KEYWORDS:
        if k in t:
            found.append(k)
    return sorted(set(found))


def guess_page_number(text: str) -> Dict[str, Any]:
    """
    Try to detect patterns like 'Page 1 of 3' or '1/3'
    """
    t = " ".join(text.split())
    for pat in PAGE_PATTERNS:
        m = pat.search(t)
        if not m:
            continue
        groups = m.groups()
        if len(groups) == 2:
            return {
                "matched": m.group(0),
                "page": int(groups[0]),
                "of": int(groups[1]),
            }
        if len(groups) == 1:
            return {
                "matched": m.group(0),
                "page": int(groups[0]),
                "of": None,
            }
    return {"matched": None, "page": None, "of": None}


def confidence_stats(confs: List[float]) -> Dict[str, Any]:
    if not confs:
        return {"avg": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    arr = np.array(confs, dtype=np.float32)
    return {
        "avg": float(arr.mean()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "count": int(arr.size),
    }


def draw_boxes_on_image(img: Image.Image, results: List[Dict[str, Any]]) -> Image.Image:
    """
    Draw rectangle boxes for each OCR result.
    """
    out = img.copy()
    draw = ImageDraw.Draw(out)

    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        # rectangle
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        # optional: small label (conf)
        # keep minimal to avoid clutter
    return out


# -----------------------------
# Main OCR pipeline
# -----------------------------
def ocr_pdf(
    pdf_path: str,
    out_dir: str,
    dpi: int = 250,
    lang: str = "en",
    gpu: bool = False,
    write_json_each_page: bool = False,
    progress_callback: Optional[Any] = None,
    save_content_and_annotated: bool = True,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "content.json")

    # EasyOCR language list
    langs = [l.strip() for l in lang.split(",") if l.strip()]
    reader = easyocr.Reader(langs, gpu=gpu)

    doc = fitz.open(pdf_path)
    page_count = doc.page_count

    if write_json_each_page:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(to_native({
                "source_pdf": os.path.basename(pdf_path),
                "page_count": page_count,
                "dpi": dpi,
                "pages": [],
            }), f, ensure_ascii=False, indent=2)
        print(f"JSON (live): {json_path}")

    # compute zoom for requested DPI (PDF default is 72 DPI)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    all_pages = []
    annotated_images = []

    for i in range(page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # OCR expects numpy array
        img_np = np.array(img)

        # EasyOCR output: [ (bbox, text, conf), ... ] bbox is 4 points
        raw = reader.readtext(img_np, detail=1)

        # normalize into structured results
        results = []
        confs = []
        for item in raw:
            poly, text, conf = item
            if text is None:
                text = ""
            conf = safe_float(conf, 0.0)
            confs.append(conf)

            x1, y1, x2, y2 = polygon_to_bbox(poly)
            results.append({
                "text": text,
                "conf": conf,
                "poly": poly,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })

        h = img.height

        # top & bottom bands
        top_results = extract_region_results(results, h, "top")
        bottom_results = extract_region_results(results, h, "bottom")

        top_text = join_text(top_results, max_lines=20)
        bottom_text = join_text(bottom_results, max_lines=20)
        full_text = join_text(results, max_lines=None)

        # page number guess prefers bottom band, but fallback to full
        page_num = guess_page_number(bottom_text if bottom_text.strip() else full_text)

        # keywords found (based on full text)
        found_kw = find_keywords(full_text)

        meta = {
            "page_index": i + 1,  # 1-based for humans
            "image_width": img.width,
            "image_height": img.height,
            "dpi": dpi,
            "ocr": {
                "confidence": confidence_stats(confs),
                "keywords_found": found_kw,
                "page_number_guess": page_num,
                "top_text": top_text,
                "bottom_text": bottom_text,
                "full_text": full_text,
                "items": results,  # includes bbox + poly + conf + text
            },
        }

        all_pages.append(meta)

        if progress_callback:
            try:
                progress_callback(i + 1, page_count)
            except Exception:
                pass

        # annotated image (only if saving full output)
        if save_content_and_annotated:
            annotated = draw_boxes_on_image(img, results)
            annotated_images.append(annotated)

        if write_json_each_page and save_content_and_annotated:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(to_native({
                    "source_pdf": os.path.basename(pdf_path),
                    "page_count": page_count,
                    "dpi": dpi,
                    "pages": all_pages,
                }), f, ensure_ascii=False, indent=2)

        print(f"Processed page {i+1}/{page_count} | OCR items: {len(results)} | avg conf: {meta['ocr']['confidence']['avg']:.3f}")

    page_text_path = os.path.join(out_dir, "page_text.json")
    page_text_pages = [
        {"page_index": p.get("page_index"), "text": ((p.get("ocr") or {}).get("full_text") or "").strip()}
        for p in all_pages
    ]
    with open(page_text_path, "w", encoding="utf-8") as f:
        json.dump({"source_pdf": os.path.basename(pdf_path), "page_count": len(page_text_pages), "pages": page_text_pages}, f, ensure_ascii=False, indent=2)

    annotated_pdf_path = None
    if save_content_and_annotated:
        out_json = {
            "source_pdf": os.path.basename(pdf_path),
            "page_count": page_count,
            "dpi": dpi,
            "pages": all_pages,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(to_native(out_json), f, ensure_ascii=False, indent=2)
        annotated_pdf_path = os.path.join(out_dir, "annotated_boxes.pdf")
        out_pdf = fitz.open()
        for img in annotated_images:
            import io
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            w_pt = img.width * 72.0 / dpi
            h_pt = img.height * 72.0 / dpi
            p = out_pdf.new_page(width=w_pt, height=h_pt)
            rect = fitz.Rect(0, 0, w_pt, h_pt)
            p.insert_image(rect, stream=img_bytes)
        out_pdf.save(annotated_pdf_path)
        out_pdf.close()

    doc.close()

    return {
        "json_path": json_path if save_content_and_annotated else None,
        "page_text_path": page_text_path,
        "annotated_pdf_path": annotated_pdf_path,
        "page_count": page_count,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Input PDF path")
    ap.add_argument("--out", default="out_ocr", help="Output directory")
    ap.add_argument("--dpi", type=int, default=250, help="Render DPI (200-300 recommended)")
    ap.add_argument("--lang", default="en", help="EasyOCR languages, comma-separated (e.g., 'en' or 'en,ms')")
    ap.add_argument("--gpu", action="store_true", help="Use GPU for EasyOCR if available")
    ap.add_argument("--incremental", action="store_true", help="Write content.json after each page (see progress live)")
    args = ap.parse_args()

    res = ocr_pdf(
        pdf_path=args.pdf,
        out_dir=args.out,
        dpi=args.dpi,
        lang=args.lang,
        gpu=args.gpu,
        write_json_each_page=args.incremental,
    )
    print("\nDone.")
    print("content.json:", res["json_path"])
    print("page_text.json:", res["page_text_path"])
    print("annotated PDF:", res["annotated_pdf_path"])
    print("pages:", res["page_count"])


if __name__ == "__main__":
    main()
