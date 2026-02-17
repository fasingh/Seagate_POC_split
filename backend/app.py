"""
Flask backend for PDF OCR + AI split. Run from project root so ocr and segment_vlm_pdf are importable.
"""
import json
import os
import re
import sys
import uuid
import threading

# Run from project root (parent of backend/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

UPLOAD_DIR = os.path.join(ROOT, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def check_gpu_available() -> bool:
    """Check if CUDA GPU is available and can actually execute kernels."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        # Test if we can actually run CUDA operations
        try:
            test_tensor = torch.zeros(1).cuda()
            _ = test_tensor + 1
            del test_tensor
            torch.cuda.empty_cache()
            return True
        except Exception:
            # CUDA available but can't execute (e.g., unsupported compute capability)
            return False
    except ImportError:
        return False

# job_id -> { "ocr": { "status", "progress", "total", "current", "error" }, "split": { ... } }
progress_store = {}
progress_lock = threading.Lock()

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB


def run_ocr_job(job_id: str, pdf_path: str):
    out_dir = os.path.join(UPLOAD_DIR, job_id)
    with progress_lock:
        progress_store[job_id]["ocr"] = {"status": "running", "progress": 0, "total": 0, "current": 0}
    try:
        from ocr import ocr_pdf

        def cb(current: int, total: int):
            pct = int(100 * current / total) if total else 0
            with progress_lock:
                progress_store[job_id]["ocr"].update(
                    status="running", progress=min(100, pct), total=total, current=current
                )

        use_gpu = check_gpu_available()
        if not use_gpu:
            raise RuntimeError("GPU is required but not available or cannot execute CUDA operations. RTX 5070 (sm_120) requires PyTorch with sm_120 support.")
        ocr_pdf(
            pdf_path=pdf_path,
            out_dir=out_dir,
            dpi=250,
            lang="en,ms",
            gpu=True,
            write_json_each_page=False,
            progress_callback=cb,
            save_content_and_annotated=False,
        )
        # Save page_text.json and input.pdf to recents folder with original filename
        original_filename_path = os.path.join(out_dir, "original_filename.txt")
        if os.path.isfile(original_filename_path):
            try:
                with open(original_filename_path, "r", encoding="utf-8") as f:
                    original_filename = f.read().strip()
                if original_filename:
                    recents_dir = os.path.join(ROOT, "recents")
                    os.makedirs(recents_dir, exist_ok=True)
                    # Create safe filename: OGPDFNAME_page_text.json (preserve extension and basic chars)
                    base_name = os.path.splitext(original_filename)[0]
                    safe_name = re.sub(r"[^\w\-\.]", "_", base_name)
                    recent_page_text_path = os.path.join(recents_dir, f"{safe_name}_page_text.json")
                    recent_pdf_path = os.path.join(recents_dir, f"{safe_name}_input.pdf")
                    page_text_path = os.path.join(out_dir, "page_text.json")
                    pdf_path = os.path.join(out_dir, "input.pdf")
                    import shutil
                    if os.path.isfile(page_text_path):
                        shutil.copy2(page_text_path, recent_page_text_path)
                    if os.path.isfile(pdf_path):
                        shutil.copy2(pdf_path, recent_pdf_path)
            except Exception:
                pass
        with progress_lock:
            progress_store[job_id]["ocr"] = {"status": "done", "progress": 100, "total": 0, "current": 0}
    except Exception as e:
        with progress_lock:
            progress_store[job_id]["ocr"] = {
                "status": "error",
                "progress": 0,
                "total": 0,
                "current": 0,
                "error": str(e),
            }


def run_split_job(job_id: str):
    out_dir = os.path.join(UPLOAD_DIR, job_id)
    pdf_path = os.path.join(out_dir, "input.pdf")
    page_text_path = os.path.join(out_dir, "page_text.json")
    splits_dir = os.path.join(out_dir, "splits")
    # Verify required files exist
    if not os.path.isfile(pdf_path):
        with progress_lock:
            progress_store[job_id]["split"] = {
                "status": "error",
                "progress": 0,
                "total": 0,
                "current": 0,
                "error": f"Input PDF file not found: {pdf_path}",
            }
        return
    if not os.path.isfile(page_text_path):
        with progress_lock:
            progress_store[job_id]["split"] = {
                "status": "error",
                "progress": 0,
                "total": 0,
                "current": 0,
                "error": f"page_text.json not found: {page_text_path}",
            }
        return
    # Read GUID from stored file
    guid = ""
    guid_path = os.path.join(out_dir, "guid.txt")
    if os.path.isfile(guid_path):
        try:
            with open(guid_path, "r", encoding="utf-8") as f:
                guid = f.read().strip()
        except Exception:
            pass
    with progress_lock:
        progress_store[job_id]["split"] = {"status": "running", "progress": 0, "total": 0, "current": 0}
    try:
        from segment_vlm_pdf import run_segmentation

        def cb(current: int, total: int):
            pct = int(100 * (current + 1) / total) if total else 0
            with progress_lock:
                progress_store[job_id]["split"].update(
                    status="running", progress=min(100, pct), total=total, current=current + 1
                )

        run_segmentation(
            pdf_path=pdf_path,
            page_text_json_path=page_text_path,
            output_dir=splits_dir,
            progress_callback=cb,
            conf=0.70,
            dpi=200,
            guid=guid,
        )
        with progress_lock:
            progress_store[job_id]["split"] = {"status": "done", "progress": 100, "total": 0, "current": 0}
    except Exception as e:
        with progress_lock:
            progress_store[job_id]["split"] = {
                "status": "error",
                "progress": 0,
                "total": 0,
                "current": 0,
                "error": str(e),
            }


def extract_guid_from_filename(filename: str) -> str:
    """Extract GUID/number from PDF filename. Returns first sequence of digits found, or first part before underscore."""
    if not filename:
        return ""
    base = os.path.splitext(filename)[0]
    # Try to find a sequence of digits (GUID/number)
    match = re.search(r'\d+', base)
    if match:
        return match.group(0)
    # If no digits, use first part before underscore
    parts = base.split('_')
    return parts[0] if parts else base[:8]


@app.route("/api/upload-pdf", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    if not f.filename or not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Invalid or missing PDF"}), 400
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    pdf_path = os.path.join(job_dir, "input.pdf")
    f.save(pdf_path)
    # Store original filename and extract GUID
    original_filename = f.filename
    guid = extract_guid_from_filename(original_filename)
    with open(os.path.join(job_dir, "original_filename.txt"), "w", encoding="utf-8") as meta_file:
        meta_file.write(original_filename)
    with open(os.path.join(job_dir, "guid.txt"), "w", encoding="utf-8") as guid_file:
        guid_file.write(guid)
    with progress_lock:
        progress_store[job_id] = {"ocr": {"status": "pending"}, "split": {"status": "pending"}}
    t = threading.Thread(target=run_ocr_job, args=(job_id, pdf_path))
    t.daemon = True
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/api/ocr-progress/<job_id>")
def ocr_progress(job_id):
    with progress_lock:
        data = progress_store.get(job_id, {}).get("ocr", {"status": "unknown"})
    # If job doesn't exist in store or status is unknown, check if files exist
    if data.get("status") == "unknown" or not data:
        job_dir = os.path.join(UPLOAD_DIR, job_id)
        if not os.path.isdir(job_dir):
            return jsonify({"status": "not_found", "error": "Job not found"})
        page_text_path = os.path.join(job_dir, "page_text.json")
        if os.path.isfile(page_text_path):
            # Update progress_store with done status
            with progress_lock:
                if job_id not in progress_store:
                    progress_store[job_id] = {"ocr": {}, "split": {}}
                progress_store[job_id]["ocr"] = {"status": "done", "progress": 100, "total": 0, "current": 0}
            return jsonify({"status": "done", "progress": 100, "total": 0, "current": 0})
        # Check if OCR is still running (page_text.json doesn't exist yet but job dir exists)
        pdf_path = os.path.join(job_dir, "input.pdf")
        if os.path.isfile(pdf_path):
            # OCR might still be running
            return jsonify({"status": "running", "progress": 0, "total": 0, "current": 0})
        return jsonify({"status": "not_found", "error": "Job not found"})
    return jsonify(data)


@app.route("/api/split-with-ai", methods=["POST"])
def split_with_ai():
    body = request.get_json() or {}
    job_id = body.get("job_id")
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    job_dir = os.path.join(UPLOAD_DIR, job_id)
    page_text_path = os.path.join(job_dir, "page_text.json")
    pdf_path = os.path.join(job_dir, "input.pdf")
    if not os.path.isfile(page_text_path):
        return jsonify({"error": "OCR not done or missing page_text.json"}), 400
    if not os.path.isfile(pdf_path):
        return jsonify({"error": "Input PDF file not found"}), 404
    with progress_lock:
        if progress_store.get(job_id, {}).get("split", {}).get("status") == "running":
            return jsonify({"error": "Split already in progress"}), 409
        progress_store.setdefault(job_id, {})["split"] = {"status": "running", "progress": 0}
    t = threading.Thread(target=run_split_job, args=(job_id,))
    t.daemon = True
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/api/split-progress/<job_id>")
def split_progress(job_id):
    with progress_lock:
        data = progress_store.get(job_id, {}).get("split", {"status": "unknown"})
    # If job doesn't exist in store, check if files exist
    if data.get("status") == "unknown":
        job_dir = os.path.join(UPLOAD_DIR, job_id)
        if not os.path.isdir(job_dir):
            return jsonify({"status": "not_found", "error": "Job not found"})
        splits_dir = os.path.join(job_dir, "splits")
        seg_path = os.path.join(splits_dir, "segments.json")
        if os.path.isfile(seg_path):
            return jsonify({"status": "done", "progress": 100})
        return jsonify({"status": "not_found", "error": "Job not found"})
    return jsonify(data)


def _manual_splits_path(job_id):
    return os.path.join(UPLOAD_DIR, job_id, "splits", "manual_splits.json")


def _read_manual_splits(job_id):
    path = _manual_splits_path(job_id)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


@app.route("/api/page-count/<job_id>")
def get_page_count(job_id):
    path = os.path.join(UPLOAD_DIR, job_id, "page_text.json")
    if not os.path.isfile(path):
        return jsonify({"page_count": 0})
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pages = data.get("pages", [])
        return jsonify({"page_count": len(pages)})
    except Exception:
        return jsonify({"page_count": 0})


@app.route("/api/manual-splits/<job_id>", methods=["POST"])
def create_manual_split(job_id):
    body = request.get_json() or {}
    name = (body.get("name") or "").strip() or "Manual split"
    from_page = int(body.get("from_page", 1))
    to_page = int(body.get("to_page", 1))
    job_dir = os.path.join(UPLOAD_DIR, job_id)
    input_path = os.path.join(job_dir, "input.pdf")
    if not os.path.isfile(input_path):
        return jsonify({"error": "PDF not found"}), 404
    page_text_path = os.path.join(job_dir, "page_text.json")
    page_count = 0
    if os.path.isfile(page_text_path):
        try:
            with open(page_text_path, "r", encoding="utf-8") as f:
                page_count = len(json.load(f).get("pages", []))
        except Exception:
            pass
    if not page_count:
        import fitz
        with fitz.open(input_path) as doc:
            page_count = len(doc)
    if from_page < 1 or to_page < from_page or to_page > page_count:
        return jsonify({"error": f"Invalid page range (1–{page_count})"}), 400
    splits_dir = os.path.join(job_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    # Read GUID from stored file
    guid = ""
    guid_path = os.path.join(job_dir, "guid.txt")
    if os.path.isfile(guid_path):
        try:
            with open(guid_path, "r", encoding="utf-8") as f:
                guid = f.read().strip()
        except Exception:
            pass
    if not guid:
        guid = uuid.uuid4().hex[:8]
    # Convert name to camelCase (no spaces, no underscores)
    pdf_name = re.sub(r"[^\w\s\-]", "", name).strip()
    words = [w for w in re.split(r"[\s_]+", pdf_name) if w]
    if words:
        pdf_name = words[0].lower() + "".join(w.capitalize() for w in words[1:])
    else:
        pdf_name = "ManualSplit"
    pdf_name = pdf_name[:80]
    filename = f"{guid}_{pdf_name}.pdf"
    out_path = os.path.join(splits_dir, filename)
    import fitz
    with fitz.open(input_path) as src:
        out_doc = fitz.open()
        for p in range(from_page - 1, to_page):
            out_doc.insert_pdf(src, from_page=p, to_page=p)
        out_doc.save(out_path)
        out_doc.close()
    manual_list = _read_manual_splits(job_id)
    manual_list.append({"filename": filename, "name": name, "from_page": from_page, "to_page": to_page})
    with open(_manual_splits_path(job_id), "w", encoding="utf-8") as f:
        json.dump(manual_list, f, indent=2)
    return jsonify({"filename": filename})


@app.route("/api/splits/<job_id>")
def list_splits(job_id):
    splits_dir = os.path.join(UPLOAD_DIR, job_id, "splits")
    if not os.path.isdir(splits_dir):
        return jsonify({"files": [], "manual_filenames": []})
    manual_list = _read_manual_splits(job_id)
    manual_filenames = {m["filename"] for m in manual_list}
    seg_path = os.path.join(splits_dir, "segments.json")
    out_files = []
    if os.path.isfile(seg_path):
        try:
            with open(seg_path, "r", encoding="utf-8") as f:
                out_files = json.load(f).get("output_files", [])
        except Exception:
            pass
    existing = [f for f in os.listdir(splits_dir) if f.lower().endswith(".pdf")]
    ai_only = [f for f in out_files if f in existing]
    manual_only = [f for f in existing if f in manual_filenames]
    combined = ai_only + [f for f in manual_only if f not in ai_only]
    for f in existing:
        if f not in combined:
            combined.append(f)
    manual_meta = {m["filename"]: {"name": m.get("name", ""), "from_page": m.get("from_page"), "to_page": m.get("to_page")} for m in manual_list}
    return jsonify({"files": combined, "manual_filenames": list(manual_filenames), "manual_meta": manual_meta})


@app.route("/api/segments/<job_id>")
def get_segments(job_id):
    path = os.path.join(UPLOAD_DIR, job_id, "splits", "segments.json")
    if not os.path.isfile(path):
        return jsonify({"segments": [], "titles": [], "output_files": []})
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify({
        "segments": data.get("segments", []),
        "titles": data.get("titles", []),
        "output_files": data.get("output_files", [])
    })


@app.route("/api/page-text/<job_id>")
def get_page_text(job_id):
    path = os.path.join(UPLOAD_DIR, job_id, "page_text.json")
    if not os.path.isfile(path):
        return jsonify({"pages": []})
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify({"pages": data.get("pages", [])})


@app.route("/api/input-pdf/<job_id>")
def get_input_pdf(job_id):
    path = os.path.join(UPLOAD_DIR, job_id, "input.pdf")
    if not os.path.isfile(path) or not path.startswith(os.path.abspath(UPLOAD_DIR)):
        return "Not found", 404
    attachment = request.args.get("attachment", "true").lower() == "true"
    return send_file(path, as_attachment=attachment, download_name="input.pdf" if attachment else None)


@app.route("/api/splits/<job_id>/<filename>", methods=["GET", "DELETE"])
def download_or_delete_split(job_id, filename):
    splits_dir = os.path.join(UPLOAD_DIR, job_id, "splits")
    path = os.path.join(splits_dir, filename)
    if not os.path.isfile(path) or not path.startswith(os.path.abspath(splits_dir)):
        return "Not found", 404
    if request.method == "DELETE":
        manual_list = _read_manual_splits(job_id)
        manual_list = [m for m in manual_list if m.get("filename") != filename]
        with open(_manual_splits_path(job_id), "w", encoding="utf-8") as f:
            json.dump(manual_list, f, indent=2)
        os.remove(path)
        return jsonify({"ok": True})
    attachment = request.args.get("attachment", "true").lower() == "true"
    return send_file(path, as_attachment=attachment, download_name=filename if attachment else None)


@app.route("/api/splits/<job_id>", methods=["DELETE"])
def delete_all_splits(job_id):
    splits_dir = os.path.join(UPLOAD_DIR, job_id, "splits")
    if not os.path.isdir(splits_dir):
        return jsonify({"ok": True})
    for f in os.listdir(splits_dir):
        if f.lower().endswith(".pdf"):
            try:
                os.remove(os.path.join(splits_dir, f))
            except Exception:
                pass
    manual_path = _manual_splits_path(job_id)
    if os.path.isfile(manual_path):
        os.remove(manual_path)
    return jsonify({"ok": True})


RECENTS_DIR = os.path.join(ROOT, "recents")
os.makedirs(RECENTS_DIR, exist_ok=True)


@app.route("/api/recents")
def list_recents():
    """List all recent PDFs (page_text.json files in recents folder)."""
    if not os.path.isdir(RECENTS_DIR):
        return jsonify({"recents": []})
    recents = []
    for filename in os.listdir(RECENTS_DIR):
        if filename.endswith("_page_text.json"):
            # Extract original PDF name (remove _page_text.json suffix)
            original_name = filename.replace("_page_text.json", "")
            # Try to restore original filename - keep underscores that might be part of original name
            # Only replace multiple consecutive underscores with space
            original_name = re.sub(r"_+", " ", original_name).strip()
            if not original_name.endswith(".pdf"):
                original_name = original_name + ".pdf"
            filepath = os.path.join(RECENTS_DIR, filename)
            try:
                mtime = os.path.getmtime(filepath)
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    page_count = data.get("page_count", 0)
                recents.append({
                    "filename": filename,
                    "original_name": original_name,
                    "page_count": page_count,
                    "modified": mtime
                })
            except Exception:
                pass
    # Sort by modified time (newest first)
    recents.sort(key=lambda x: x["modified"], reverse=True)
    return jsonify({"recents": recents})


@app.route("/api/recents/<filename>")
def load_recent(filename):
    """Load a recent PDF's page_text.json and input.pdf and create a new job for it."""
    recent_path = os.path.join(RECENTS_DIR, filename)
    if not os.path.isfile(recent_path) or not filename.endswith("_page_text.json"):
        return jsonify({"error": "Recent file not found"}), 404
    # Create new job
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    # Copy page_text.json to new job directory
    page_text_path = os.path.join(job_dir, "page_text.json")
    import shutil
    shutil.copy2(recent_path, page_text_path)
    # Also copy input.pdf if it exists in recents
    base_name = filename.replace("_page_text.json", "")
    recent_pdf_path = os.path.join(RECENTS_DIR, f"{base_name}_input.pdf")
    pdf_path = os.path.join(job_dir, "input.pdf")
    if os.path.isfile(recent_pdf_path):
        shutil.copy2(recent_pdf_path, pdf_path)
    # Extract original filename and GUID (same logic as list_recents)
    original_name = base_name
    original_name = re.sub(r"_+", " ", original_name).strip()
    if not original_name.endswith(".pdf"):
        original_name = original_name + ".pdf"
    guid = extract_guid_from_filename(original_name)
    with open(os.path.join(job_dir, "original_filename.txt"), "w", encoding="utf-8") as f:
        f.write(original_name)
    with open(os.path.join(job_dir, "guid.txt"), "w", encoding="utf-8") as f:
        f.write(guid)
    # Mark OCR as done
    with progress_lock:
        progress_store[job_id] = {"ocr": {"status": "done", "progress": 100}, "split": {"status": "pending"}}
    return jsonify({"job_id": job_id, "original_name": original_name})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
