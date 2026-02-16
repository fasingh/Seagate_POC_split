"""
Run OCR on 717129_Redacted.pdf with incremental JSON (content.json updated after each page).
Open out_717129/content.json in another window to watch progress.
"""
import os
from ocr import ocr_pdf

PDF = "717129_Redacted.pdf"
OUT = "out_717129"

if __name__ == "__main__":
    if not os.path.isfile(PDF):
        print(f"Missing: {PDF}")
        exit(1)
    res = ocr_pdf(
        pdf_path=PDF,
        out_dir=OUT,
        dpi=250,
        lang="en,ms",
        gpu=False,
        write_json_each_page=True,
    )
    print("\nDone.")
    print("content.json:", res["json_path"])
    print("annotated PDF:", res["annotated_pdf_path"])
    print("pages:", res["page_count"])
