# PDF OCR + AI Split App

## Backend (Flask)

From the **project root** (Seagate folder):

```bash
pip install flask flask-cors
python backend/app.py
```

Runs at http://localhost:5000. Needs `.env` with `OPENAI_API_KEY` for the Split with AI step.

## Frontend (React)

```bash
cd frontend
npm install
npm run dev
```

Runs at http://localhost:5173 and proxies `/api` to the backend.

## Flow

1. **Upload PDF** — Upload a PDF; OCR runs in the background. Progress bar + fun facts. When done: "PDF has been processed, ready to split through AI."
2. **Split with AI** — Click "Start AI split". VLM detects document boundaries; progress bar. When done: "Splitting complete."
3. **Manual Reconciliation** — Preview or download each split PDF.
