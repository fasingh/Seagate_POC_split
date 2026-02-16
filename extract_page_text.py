"""
Read content.json (from ocr.py) and output page-wise text as JSON.
Usage: python extract_page_text.py [content.json] [--out output.json]
"""
import json
import argparse
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_file", nargs="?", default="out_ocr/content.json", help="Path to content.json")
    ap.add_argument("--out", "-o", default=None, help="Output JSON file (default: <json_dir>/page_text.json)")
    args = ap.parse_args()

    with open(args.json_file, encoding="utf-8") as f:
        data = json.load(f)

    if args.out:
        out_path = args.out
    else:
        out_path = os.path.join(os.path.dirname(args.json_file), "page_text.json")

    pages = []
    for p in data.get("pages", []):
        pages.append({
            "page_index": p.get("page_index"),
            "text": ((p.get("ocr") or {}).get("full_text") or "").strip(),
        })

    out_json = {"source_pdf": data.get("source_pdf"), "page_count": len(pages), "pages": pages}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(pages)} pages to {out_path}")


if __name__ == "__main__":
    main()
