import os
import re
import json
import base64
import uuid
import argparse
import time
from typing import Dict, Any, List, Tuple, Optional, Callable

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI


# ----------------------------
# Config
# ----------------------------
DEFAULT_MODEL = "gpt-4.1-mini"  # cheap + vision-capable in docs examples :contentReference[oaicite:1]{index=1}

SYSTEM_PROMPT = """You are a strict document-boundary detector.
You will be shown TWO consecutive pages from a long combined PDF: previous page and current page.

Task:
Decide whether the CURRENT page starts a NEW document, or continues the SAME document as the previous page.

Rules:
- Use only visible evidence from the pages (layout, headers, logos, titles, IDs, page numbers, envelopes, etc.)
- If you are unsure, set new_doc_start=false and confidence<0.7.
- Return ONLY valid JSON exactly matching the schema.

JSON schema:
{
  "new_doc_start": boolean,
  "confidence": number,  // 0.0 to 1.0
  "reasons": [string]    // short bullet reasons
}
"""

USER_PROMPT_TEMPLATE = """Previous page index: {prev_i}
Current page index: {curr_i}

OCR text (previous page):
{prev_text}

OCR text (current page):
{curr_text}

Question: Does the CURRENT page (index {curr_i}) start a NEW document?
Return JSON only.
"""

TITLE_SYSTEM_PROMPT = """You suggest a very short, professional document title based on the first page content.
Rules: 1-3 words only. Use only letters, numbers; no vulgar or offensive content. Professional and neutral.
Return ONLY the title as plain text, nothing else. Example: Offer_Letter or Tax_Invoice_2024."""
TITLE_USER_TEMPLATE = """First page OCR text of this document:

{text}

Suggest a 1-3 word title for this document. Reply with only the title, words separated by underscores. No quotes or punctuation."""


def load_page_text_json(path: str) -> Tuple[str, int, Dict[int, str]]:
    """
    Expected format:
    {
      "source_pdf": "...",
      "page_count": N,
      "pages": [{"page_index": 1, "text": "..."}, ...]
    }
    """
    resolved = os.path.normpath(path)
    if not os.path.isfile(resolved):
        resolved = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"File not found: {path}")
    with open(resolved, "r", encoding="utf-8") as f:
        data = json.load(f)

    source_pdf = data.get("source_pdf", "")
    page_count = int(data.get("page_count", 0) or 0)

    pages = data.get("pages", [])
    if not pages:
        raise ValueError("page_text.json has no pages[]")

    text_by_page = {}
    for p in pages:
        i = int(p.get("page_index", 0) or 0)
        text_by_page[i] = p.get("text", "") or ""

    # If page_count is missing, infer
    if page_count <= 0:
        page_count = max(text_by_page.keys())

    return source_pdf, page_count, text_by_page


def render_page_png_bytes(pdf_path: str, page_index_1based: int, dpi: int = 200) -> bytes:
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_index_1based - 1)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


def to_data_url_png(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def call_vlm_boundary(
    client: OpenAI,
    model: str,
    prev_img_url: str,
    curr_img_url: str,
    prev_i: int,
    curr_i: int,
    prev_text: str,
    curr_text: str,
    max_retries: int = 3,
) -> Dict[str, Any]:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        prev_i=prev_i,
        curr_i=curr_i,
        prev_text=(prev_text or "")[:6000],
        curr_text=(curr_text or "")[:6000],
    )

    # OpenAI vision input pattern via image_url blocks :contentReference[oaicite:2]{index=2}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": prev_img_url}},
                {"type": "image_url", "image_url": {"url": curr_img_url}},
            ],
        },
    ]

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                timeout=120.0,
            )
            raw = resp.choices[0].message.content

            # Expect raw JSON in message.content
            parsed = json.loads(raw)

            # Normalize
            return {
                "new_doc_start": bool(parsed.get("new_doc_start", False)),
                "confidence": safe_float(parsed.get("confidence", 0.0), 0.0),
                "reasons": parsed.get("reasons", []) or [],
                "raw": raw,
            }

        except Exception as e:
            last_err = str(e)
            # Simple backoff
            time.sleep(0.8 * attempt)

    # If it keeps failing, be conservative
    return {
        "new_doc_start": False,
        "confidence": 0.0,
        "reasons": [f"api_error:{last_err}"] if last_err else ["api_error"],
        "raw": None,
    }


def sanitize_title(s: str) -> str:
    """1-3 words, underscore-separated, safe for filename."""
    s = (s or "").strip()
    s = re.sub(r"[^\w\s-]", "", s)
    words = [w for w in re.split(r"[\s_]+", s) if w][:3]
    return "_".join(words) if words else "document"


def call_vlm_title(
    client: OpenAI,
    model: str,
    first_page_text: str,
    first_page_img_url: Optional[str] = None,
    max_retries: int = 2,
) -> str:
    """Get a 1-3 word title for a document from its first page."""
    text = (first_page_text or "")[:4000]
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": TITLE_USER_TEMPLATE.format(text=text)},
    ]
    if first_page_img_url:
        content.append({"type": "image_url", "image_url": {"url": first_page_img_url}})
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": TITLE_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                temperature=0.3,
                timeout=120.0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            raw = re.sub(r"^[\"']|[\"']$", "", raw)
            return sanitize_title(raw)
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return "document"


def build_segments_from_starts(starts: List[int], page_count: int) -> List[Dict[str, int]]:
    starts = sorted(set(starts))
    segments = []
    for idx, s in enumerate(starts):
        e = (starts[idx + 1] - 1) if idx + 1 < len(starts) else page_count
        segments.append({"doc_index": idx + 1, "start_page": s, "end_page": e})
    return segments


def split_pdf_by_segments(
    pdf_path: str,
    segments: List[Dict[str, int]],
    output_dir: str,
    titles: Optional[List[str]] = None,
) -> List[str]:
    """Write one PDF per segment to output_dir. If titles provided, use {guid}_split_{title}.pdf."""
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    out_paths = []
    src = fitz.open(pdf_path)
    try:
        for i, seg in enumerate(segments):
            doc_index = seg["doc_index"]
            start = seg["start_page"]
            end = seg["end_page"]
            if titles and i < len(titles) and titles[i]:
                guid = uuid.uuid4().hex[:8]
                safe = re.sub(r"[^\w\-]", "_", (titles[i] or "document").strip())[:80]
                out_name = f"{guid}_split_{safe}.pdf"
            else:
                out_name = f"{base_name}_doc_{doc_index:03d}_p{start}-{end}.pdf"
            out_path = os.path.join(output_dir, out_name)
            out_doc = fitz.open()
            for p in range(start - 1, end):
                out_doc.insert_pdf(src, from_page=p, to_page=p)
            out_doc.save(out_path)
            out_doc.close()
            out_paths.append(out_path)
    finally:
        src.close()
    return out_paths


def run_segmentation(
    pdf_path: str,
    page_text_json_path: str,
    output_dir: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    conf: float = 0.70,
    dpi: int = 200,
) -> Dict[str, Any]:
    """
    Run VLM boundary detection and split PDF. progress_callback(current_transition, total_transitions).
    Returns { "doc_starts", "segments", "out_paths", "segments_path" }.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment/.env")
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    _, json_page_count, text_by_page = load_page_text_json(page_text_json_path)
    pdf_doc = fitz.open(pdf_path)
    try:
        pdf_page_count = pdf_doc.page_count
    finally:
        pdf_doc.close()
    page_count = min(pdf_page_count, json_page_count)
    if page_count < 1:
        raise ValueError("No pages to segment")

    # Single-page PDF: one segment (whole doc), no VLM calls
    total_transitions = max(0, page_count - 1)
    client = OpenAI(api_key=api_key)
    doc_starts = [1]
    debug = []

    for curr in range(2, page_count + 1):
        prev = curr - 1
        prev_text = text_by_page.get(prev, "")
        curr_text = text_by_page.get(curr, "")
        prev_png = render_page_png_bytes(pdf_path, prev, dpi=dpi)
        curr_png = render_page_png_bytes(pdf_path, curr, dpi=dpi)
        prev_url = to_data_url_png(prev_png)
        curr_url = to_data_url_png(curr_png)
        decision = call_vlm_boundary(
            client=client, model=model,
            prev_img_url=prev_url, curr_img_url=curr_url,
            prev_i=prev, curr_i=curr, prev_text=prev_text, curr_text=curr_text,
        )
        split_applied = bool(decision["new_doc_start"] and decision["confidence"] >= conf)
        time.sleep(0.25)  # reduce rate-limit errors on large PDFs
        if split_applied:
            doc_starts.append(curr)
        debug.append({
            "prev_page": prev, "curr_page": curr,
            "new_doc_start_at_curr": decision["new_doc_start"],
            "confidence": round(decision["confidence"], 3),
            "split_applied": split_applied,
            "reasons": decision["reasons"],
        })
        transition_index = curr - 1
        if progress_callback:
            try:
                progress_callback(transition_index, total_transitions)
            except Exception:
                pass

    segments = build_segments_from_starts(doc_starts, page_count)
    titles: List[str] = []
    for seg in segments:
        start = seg["start_page"]
        first_text = text_by_page.get(start, "")
        first_url: Optional[str] = None
        try:
            first_png = render_page_png_bytes(pdf_path, start, dpi=dpi)
            first_url = to_data_url_png(first_png)
        except Exception:
            pass
        title = call_vlm_title(client, model, first_text, first_url)
        titles.append(title)
    out_paths = split_pdf_by_segments(pdf_path, segments, output_dir, titles=titles)
    output_files = [os.path.basename(p) for p in out_paths]
    segments_path = os.path.join(output_dir, "segments.json")
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump({
            "pdf": pdf_path, "page_count_used": page_count,
            "doc_starts": doc_starts, "segments": segments,
            "output_files": output_files, "titles": titles,
        }, f, ensure_ascii=False, indent=2)
    return {
        "doc_starts": doc_starts,
        "segments": segments,
        "out_paths": out_paths,
        "segments_path": segments_path,
    }


# ----------------------------
# Main
# ----------------------------
def main():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not found in environment/.env")

    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Input combined PDF path")
    ap.add_argument("--page_text_json", required=True, help="OCR page_text.json path")
    ap.add_argument("--out", default="segments.json", help="Output segments JSON")
    ap.add_argument("--debug_out", default="debug_transitions.json", help="Debug decisions JSON")
    ap.add_argument("--dpi", type=int, default=200, help="Render DPI for page images")
    ap.add_argument("--conf", type=float, default=0.70, help="Confidence threshold to apply split")
    ap.add_argument("--max_pages", type=int, default=0, help="Limit pages for testing (0 = all)")
    ap.add_argument("--output_pdf", default="output_pdf", help="Folder to save split PDFs (one PDF per segment)")
    args = ap.parse_args()

    # Load OCR text
    src_pdf_name, json_page_count, text_by_page = load_page_text_json(args.page_text_json)

    # Verify PDF page count
    pdf_doc = fitz.open(args.pdf)
    try:
        pdf_page_count = pdf_doc.page_count
    finally:
        pdf_doc.close()

    page_count = min(pdf_page_count, json_page_count)
    if args.max_pages and args.max_pages > 0:
        page_count = min(page_count, args.max_pages)

    if page_count < 2:
        raise SystemExit(f"Not enough pages to segment. page_count={page_count}")

    client = OpenAI(api_key=api_key)

    # Indexing rule:
    # doc_starts holds pages where a new doc begins. First doc always starts at 1.
    doc_starts = [1]
    debug = []

    for curr in range(2, page_count + 1):
        prev = curr - 1

        prev_text = text_by_page.get(prev, "")
        curr_text = text_by_page.get(curr, "")

        prev_png = render_page_png_bytes(args.pdf, prev, dpi=args.dpi)
        curr_png = render_page_png_bytes(args.pdf, curr, dpi=args.dpi)

        prev_url = to_data_url_png(prev_png)
        curr_url = to_data_url_png(curr_png)

        decision = call_vlm_boundary(
            client=client,
            model=model,
            prev_img_url=prev_url,
            curr_img_url=curr_url,
            prev_i=prev,
            curr_i=curr,
            prev_text=prev_text,
            curr_text=curr_text,
        )

        split_applied = bool(decision["new_doc_start"] and decision["confidence"] >= args.conf)
        if split_applied:
            doc_starts.append(curr)

        debug.append({
            "prev_page": prev,
            "curr_page": curr,
            "new_doc_start_at_curr": decision["new_doc_start"],
            "confidence": round(decision["confidence"], 3),
            "split_applied": split_applied,
            "reasons": decision["reasons"],
        })

        print(
            f"[{prev}->{curr}] new_doc={decision['new_doc_start']} "
            f"conf={decision['confidence']:.2f} split={split_applied}"
        )

    segments = build_segments_from_starts(doc_starts, page_count)

    print("\nGenerating titles for each document...")
    titles = []
    for seg in segments:
        start = seg["start_page"]
        first_text = text_by_page.get(start, "")
        first_url = None
        try:
            first_png = render_page_png_bytes(args.pdf, start, dpi=args.dpi)
            first_url = to_data_url_png(first_png)
        except Exception:
            pass
        title = call_vlm_title(client, model, first_text, first_url)
        titles.append(title)
        print(f"  Doc {seg['doc_index']}: {title}")

    out_pdfs = split_pdf_by_segments(args.pdf, segments, args.output_pdf, titles=titles)
    output_files = [os.path.basename(p) for p in out_pdfs]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pdf": args.pdf,
                "page_count_used": page_count,
                "doc_starts": doc_starts,
                "segments": segments,
                "output_files": output_files,
                "titles": titles,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    seg_in_output = os.path.join(args.output_pdf, "segments.json")
    with open(seg_in_output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pdf": args.pdf,
                "page_count_used": page_count,
                "doc_starts": doc_starts,
                "segments": segments,
                "output_files": output_files,
                "titles": titles,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(args.debug_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pdf": args.pdf,
                "page_count_used": page_count,
                "transitions": debug,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nSplit PDFs ({len(out_pdfs)}) -> {args.output_pdf}/")
    for p in out_pdfs:
        print(" ", os.path.basename(p))

    print("\nDoc starts:", doc_starts)
    print("Segments:", segments)
    print(f"Wrote: {args.out}")
    print(f"Wrote: {args.debug_out}")


if __name__ == "__main__":
    main()
