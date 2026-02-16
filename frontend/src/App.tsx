import { useState, useRef, useEffect } from 'react'

const API = '/api'

type Page = 'upload' | 'preview' | 'manual'

const FUN_FACTS = [
  'PDFs can contain embedded fonts, images, and even JavaScript.',
  'The first PDF was created in 1993 by Adobe co-founder John Warnock.',
  'OCR stands for Optical Character Recognition.',
  'A single PDF can have different page sizes in the same file.',
  'Over 2.5 billion PDFs are opened every year.',
  'AI vision models can detect document boundaries from layout and headers.',
]

function App() {
  const [page, setPage] = useState<Page>('upload')
  const [jobId, setJobId] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null)
  const [ocrProgress, setOcrProgress] = useState(0)
  const [ocrCurrent, setOcrCurrent] = useState(0)
  const [ocrTotal, setOcrTotal] = useState(0)
  const [ocrStatus, setOcrStatus] = useState<'idle' | 'running' | 'done' | 'error'>('idle')
  const [ocrError, setOcrError] = useState<string | null>(null)
  const [splitProgress, setSplitProgress] = useState(0)
  const [splitStatus, setSplitStatus] = useState<'idle' | 'running' | 'done' | 'error'>('idle')
  const [splitError, setSplitError] = useState<string | null>(null)
  const [splitFiles, setSplitFiles] = useState<string[]>([])
  const [manualFilenames, setManualFilenames] = useState<Set<string>>(new Set())
  const [manualMeta, setManualMeta] = useState<Record<string, { name?: string; from_page?: number; to_page?: number }>>({})
  const [reconTabIndex, setReconTabIndex] = useState(0)
  const [reconViewMode, setReconViewMode] = useState<'text' | 'tab'>('tab')
  const [segments, setSegments] = useState<{ start_page: number; end_page: number; doc_index: number }[]>([])
  const [pageTextPages, setPageTextPages] = useState<{ page_index: number; text: string }[]>([])
  const [funFactIndex, setFunFactIndex] = useState(0)
  const [manualFormOpen, setManualFormOpen] = useState(false)
  const [manualName, setManualName] = useState('')
  const [manualFromPage, setManualFromPage] = useState(1)
  const [manualToPage, setManualToPage] = useState(1)
  const [pageCount, setPageCount] = useState(0)
  const [manualSaveMessage, setManualSaveMessage] = useState<string | null>(null)
  const [manualSaving, setManualSaving] = useState(false)
  const [manualError, setManualError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (!jobId || ocrStatus !== 'running') return
    const poll = async () => {
      try {
        const r = await fetch(`${API}/ocr-progress/${jobId}`)
        const d = await r.json()
        setOcrProgress(d.progress ?? 0)
        setOcrCurrent(d.current ?? 0)
        setOcrTotal(d.total ?? 0)
        if (d.status === 'done') setOcrStatus('done')
        if (d.status === 'error') {
          setOcrStatus('error')
          setOcrError(d.error ?? 'OCR failed')
        }
      } catch { /* ignore */ }
    }
    const t = setInterval(poll, 800)
    const onVisible = () => { if (document.visibilityState === 'visible') poll() }
    document.addEventListener('visibilitychange', onVisible)
    return () => { clearInterval(t); document.removeEventListener('visibilitychange', onVisible) }
  }, [jobId, ocrStatus])

  useEffect(() => {
    if (!jobId || splitStatus !== 'running') return
    const t = setInterval(async () => {
      try {
        const r = await fetch(`${API}/split-progress/${jobId}`)
        const d = await r.json()
        setSplitProgress(d.progress ?? 0)
        if (d.status === 'done') {
          setSplitStatus('done')
          const list = await fetch(`${API}/splits/${jobId}`)
          const listData = await list.json()
          setSplitFiles(listData.files ?? [])
        }
        if (d.status === 'error') {
          setSplitStatus('error')
          setSplitError(d.error ?? 'AI split failed')
        }
      } catch { /* ignore */ }
    }, 800)
    return () => clearInterval(t)
  }, [jobId, splitStatus])

  useEffect(() => {
    if (page !== 'upload') return
    const t = setInterval(() => setFunFactIndex((i) => (i + 1) % FUN_FACTS.length), 4000)
    return () => clearInterval(t)
  }, [page])

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || !file.name.toLowerCase().endsWith('.pdf')) return
    setSelectedFileName(file.name)
    setUploading(true)
    setOcrStatus('running')
    setOcrProgress(0)
    setOcrCurrent(0)
    setOcrTotal(0)
    setOcrError(null)
    const form = new FormData()
    form.append('file', file)
    try {
      const r = await fetch(`${API}/upload-pdf`, { method: 'POST', body: form })
      const d = await r.json()
      if (d.job_id) setJobId(d.job_id)
      else {
        setOcrStatus('error')
        setOcrError(d.error ?? 'Upload failed')
      }
    } catch (err) {
      setOcrStatus('error')
      setOcrError(String(err))
    }
    setUploading(false)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const startSplit = async () => {
    if (!jobId) return
    setSplitStatus('running')
    setSplitProgress(0)
    setSplitError(null)
    try {
      const r = await fetch(`${API}/split-with-ai`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: jobId }),
      })
      const d = await r.json()
      if (d.error) {
        setSplitStatus('error')
        setSplitError(d.error)
      }
    } catch (err) {
      setSplitStatus('error')
      setSplitError(String(err))
    }
  }

  const loadSplits = async () => {
    if (!jobId) return
    try {
      const [splitsRes, segRes, ptRes] = await Promise.all([
        fetch(`${API}/splits/${jobId}`),
        fetch(`${API}/segments/${jobId}`),
        fetch(`${API}/page-text/${jobId}`),
      ])
      const splitsData = await splitsRes.json()
      const segData = await segRes.json()
      const ptData = await ptRes.json()
      setSplitFiles(splitsData.files ?? [])
      setManualFilenames(new Set(splitsData.manual_filenames ?? []))
      setManualMeta(splitsData.manual_meta ?? {})
      setSegments(segData.segments ?? [])
      setPageTextPages(ptData.pages ?? [])
      setReconTabIndex(0)
    } catch { setSplitFiles([]); setSegments([]); setPageTextPages([]); setManualFilenames(new Set()); setManualMeta({}) }
  }

  useEffect(() => {
    if (page === 'preview' && jobId) loadSplits()
  }, [page, jobId])

  useEffect(() => {
    if (page === 'manual' && jobId) {
      fetch(`${API}/page-count/${jobId}`)
        .then((r) => r.json())
        .then((d) => setPageCount(d.page_count ?? 0))
        .catch(() => setPageCount(0))
    }
  }, [page, jobId])

  const saveManualSplit = async () => {
    if (!jobId) return
    setManualSaving(true)
    setManualError(null)
    setManualSaveMessage(null)
    try {
      const r = await fetch(`${API}/manual-splits/${jobId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: manualName || 'Manual split', from_page: manualFromPage, to_page: manualToPage }),
      })
      const d = await r.json()
      if (!r.ok) throw new Error(d.error || 'Failed')
      setManualSaveMessage('Split saved, please go to Preview to view your new split.')
      setManualName('')
      setManualFromPage(1)
      setManualToPage(1)
      loadSplits()
    } catch (e) {
      setManualError(e instanceof Error ? e.message : 'Failed to save split')
    } finally {
      setManualSaving(false)
    }
  }

  const deleteSplit = async (filename: string) => {
    if (!jobId) return
    try {
      await fetch(`${API}/splits/${jobId}/${encodeURIComponent(filename)}`, { method: 'DELETE' })
      loadSplits()
    } catch { /* ignore */ }
  }

  const deleteAllSplits = async () => {
    if (!jobId || !window.confirm('Delete all splits?')) return
    try {
      await fetch(`${API}/splits/${jobId}`, { method: 'DELETE' })
      loadSplits()
    } catch { /* ignore */ }
  }

  const displayName = (filename: string) => {
    if (manualMeta[filename]?.name) return manualMeta[filename].name!
    const base = filename.replace(/\.pdf$/i, '')
    const afterSplit = base.split('_split_')[1]
    const afterManual = base.startsWith('manual_') ? base.replace(/^manual_[a-f0-9]+_/, '') : null
    return (afterManual ?? afterSplit ?? base).replace(/_/g, ' ')
  }
  const isManualSplit = (filename: string) => manualFilenames.has(filename)
  const pageRangeFor = (i: number, filename?: string) => {
    if (filename && manualMeta[filename]) {
      const a = manualMeta[filename].from_page ?? 0
      const b = manualMeta[filename].to_page ?? 0
      return a === b ? `Page ${a}` : `Pages ${a}–${b}`
    }
    const seg = segments[i]
    if (!seg) return ''
    const a = seg.start_page + 1
    const b = seg.end_page + 1
    return a === b ? `Page ${a}` : `Pages ${a}–${b}`
  }
  const getTextForSegment = (docIndex: number): string => {
    const seg = segments.find((s) => s.doc_index === docIndex)
    if (!seg) return ''
    const byPage = Object.fromEntries(pageTextPages.map((p) => [p.page_index, p.text]))
    const lines: string[] = []
    for (let p = seg.start_page; p <= seg.end_page; p++) {
      lines.push(byPage[p] ?? '')
    }
    return lines.join('\n\n').trim()
  }

  const previewUrl = (filename: string) => `${API}/splits/${jobId}/${filename}?attachment=false`
  const downloadUrl = (filename: string) => `${API}/splits/${jobId}/${filename}`

  return (
    <div className="app">
      <header className="header">
        <img src="/company_logo.png" alt="" className="header-logo" />
        <span className="header-name">Auritas AI</span>
      </header>
      <div className="app-body">
        <nav className="sidebar">
          <button className={page === 'upload' ? 'active' : ''} onClick={() => setPage('upload')}>Upload & AI split</button>
          <button className={page === 'preview' ? 'active' : ''} onClick={() => setPage('preview')}>Preview</button>
          <button className={page === 'manual' ? 'active' : ''} onClick={() => setPage('manual')}>Manual Splits</button>
        </nav>
        <main className="main">
          <div className="main-content">
          {page === 'upload' && (
            <div className="page">
              <h1>Upload & AI split</h1>
              <label className="upload-zone">
                <input ref={fileInputRef} type="file" accept=".pdf" onChange={handleUpload} />
                <span className="label">{selectedFileName ? 'Choose another PDF' : 'Choose PDF file'}</span>
              </label>
              {(uploading || ocrStatus === 'running') && (
                <>
                  <p className="status-line">
                    {uploading ? (
                      <><strong>Uploading</strong> {selectedFileName}…</>
                    ) : (
                      <><strong>Processing</strong> {selectedFileName} — {ocrTotal ? `page ${ocrCurrent} of ${ocrTotal}` : 'starting OCR…'}</>
                    )}
                  </p>
                  <div className="progress-wrap">
                    <div className="progress-bar" style={{ width: `${ocrProgress}%` }} />
                  </div>
                  <p className="progress-label">{ocrProgress}%</p>
                  {jobId && ocrStatus === 'running' && (
                    <button type="button" className="btn" style={{ marginTop: 8 }} onClick={async () => {
                      try {
                        const r = await fetch(`${API}/ocr-progress/${jobId}`)
                        const d = await r.json()
                        setOcrProgress(d.progress ?? 0)
                        setOcrCurrent(d.current ?? 0)
                        setOcrTotal(d.total ?? 0)
                        if (d.status === 'done') setOcrStatus('done')
                        if (d.status === 'error') { setOcrStatus('error'); setOcrError(d.error ?? 'OCR failed') }
                      } catch { /* ignore */ }
                    }}>Check OCR status</button>
                  )}
                </>
              )}
              {ocrStatus === 'done' && (
                <>
                  <p className="success">PDF processed.</p>
                  {splitStatus === 'idle' && <button className="btn" onClick={startSplit} style={{ marginTop: 8 }}>Start AI split</button>}
                  {splitStatus === 'running' && (
                    <>
                      <p className="status-line"><strong>Analyzing</strong> page boundaries…</p>
                      <div className="progress-wrap">
                        <div className="progress-bar" style={{ width: `${splitProgress}%` }} />
                      </div>
                      <p className="progress-label">{splitProgress}%</p>
                    </>
                  )}
                  {splitStatus === 'done' && <p className="success">AI split complete. Go to Preview to view or download.</p>}
                  {splitStatus === 'error' && <p className="error">{splitError}</p>}
                </>
              )}
              {ocrStatus === 'error' && <p className="error">{ocrError}</p>}
            </div>
          )}

          {page === 'manual' && (
            <div className="page manual-splits-page">
              <h1>Manual Splits</h1>
              {!jobId ? (
                <p className="muted">Upload & AI split a PDF first.</p>
              ) : (
                <>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
                    <button type="button" className="btn add-split-btn" onClick={() => { setManualFormOpen(!manualFormOpen); setManualSaveMessage(null); setManualError(null); }}>
                      + Add split
                    </button>
                  </div>
                  <div className={manualFormOpen ? 'manual-splits-layout' : 'manual-splits-full'}>
                    <div className={manualFormOpen ? 'manual-pdf-left' : 'manual-pdf-full'}>
                      <div className="recon-preview-wrap">
                        <iframe title="Original PDF" src={`${API}/input-pdf/${jobId}?attachment=false`} />
                        <div className="recon-actions">
                          <a href={`${API}/input-pdf/${jobId}`} download="input.pdf" className="btn">Download</a>
                          <span className="muted" style={{ fontSize: 11 }}>Original PDF</span>
                        </div>
                      </div>
                    </div>
                    {manualFormOpen && (
                      <div className="manual-form-right">
                        <div className="manual-form">
                          <label>Name of the PDF</label>
                          <input type="text" value={manualName} onChange={(e) => setManualName(e.target.value)} placeholder="e.g. Chapter 1" />
                          <label>From page</label>
                          <input type="number" min={1} max={pageCount || 999} value={manualFromPage} onChange={(e) => setManualFromPage(parseInt(e.target.value, 10) || 1)} />
                          <label>To page</label>
                          <input type="number" min={1} max={pageCount || 999} value={manualToPage} onChange={(e) => setManualToPage(parseInt(e.target.value, 10) || 1)} />
                          <button type="button" className="btn" onClick={saveManualSplit} disabled={manualSaving}>Save</button>
                          {manualSaveMessage && <p className="success">{manualSaveMessage}</p>}
                          {manualError && <p className="error">{manualError}</p>}
                        </div>
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          )}

          {page === 'preview' && (
            <div className="page recon-page">
              <h1>Preview</h1>
              {!jobId ? (
                <p className="muted">Upload & AI split a PDF first.</p>
              ) : splitFiles.length === 0 ? (
                <>
                  <button className="btn" onClick={loadSplits} style={{ marginBottom: 12 }}>Refresh list</button>
                  <p className="muted">No split files yet. Complete Upload & AI split first.</p>
                  {splitError && <p className="error" style={{ marginTop: 8 }}>Last AI split error: {splitError}</p>}
                </>
              ) : (
                <>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12, flexWrap: 'wrap' }}>
                    <button className="btn" onClick={loadSplits}>Refresh list</button>
                    <button type="button" className="btn" style={{ background: 'var(--red)', color: '#fff' }} onClick={deleteAllSplits}>Delete all splits</button>
                  </div>
                  <div className="recon-view-toggle">
                    <span
                      className={reconViewMode === 'text' ? 'selected' : ''}
                      onClick={() => setReconViewMode('text')}
                      role="button"
                      tabIndex={0}
                      onKeyDown={(e) => e.key === 'Enter' && setReconViewMode('text')}
                    >
                      Text View
                    </span>
                    <span className="recon-view-sep">|</span>
                    <span
                      className={reconViewMode === 'tab' ? 'selected' : ''}
                      onClick={() => setReconViewMode('tab')}
                      role="button"
                      tabIndex={0}
                      onKeyDown={(e) => e.key === 'Enter' && setReconViewMode('tab')}
                    >
                      Tab View
                    </span>
                  </div>
                  {reconViewMode === 'tab' && (
                    <>
                      <div className="recon-tiles">
                        {splitFiles.map((f, i) => (
                          <button
                            key={f}
                            type="button"
                            className={`recon-tile ${i === reconTabIndex ? 'selected' : ''}`}
                            onClick={() => setReconTabIndex(i)}
                            title={`${f} (${pageRangeFor(i, f)})`}
                          >
                            {displayName(f)}
                            {isManualSplit(f) && <span className="tile-badge">Human</span>}
                          </button>
                        ))}
                      </div>
                      <div className="recon-content">
                        <div className="recon-preview-wrap">
                          <iframe key={splitFiles[reconTabIndex]} title="PDF preview" src={previewUrl(splitFiles[reconTabIndex] ?? '')} />
                          <div className="recon-actions">
                            {isManualSplit(splitFiles[reconTabIndex] ?? '') && <span className="human-badge">Human made split</span>}
                            <a href={downloadUrl(splitFiles[reconTabIndex] ?? '')} download={splitFiles[reconTabIndex]} className="btn">Download</a>
                            <button type="button" className="btn btn-danger" onClick={() => deleteSplit(splitFiles[reconTabIndex] ?? '')}>Delete</button>
                            <span className="muted" style={{ fontSize: 11 }}>{splitFiles[reconTabIndex]}</span>
                          </div>
                        </div>
                      </div>
                    </>
                  )}
                  {reconViewMode === 'text' && (
                    <ul className="recon-list">
                      {splitFiles.map((f, i) => (
                        <li key={f} className="recon-list-row">
                          <span className="recon-list-doc">{displayName(f)}</span>
                          {isManualSplit(f) && <span className="human-badge-inline">Human made split</span>}
                          <span className="recon-list-pages muted">{pageRangeFor(i, f)}</span>
                          <span className="recon-list-filename muted">{f}</span>
                          <a href={previewUrl(f)} target="_blank" rel="noopener noreferrer" className="btn btn-sm">Preview</a>
                          <a href={downloadUrl(f)} download={f} className="btn btn-sm">Download</a>
                          <button type="button" className="btn btn-sm btn-danger" onClick={() => deleteSplit(f)}>Delete</button>
                        </li>
                      ))}
                    </ul>
                  )}
                </>
              )}
            </div>
          )}
          </div>
          {page === 'upload' && (
            <div className="facts-strip">Fact: {FUN_FACTS[funFactIndex]}</div>
          )}
          <footer className="footer-made-by">Made by Auritas</footer>
        </main>
      </div>
    </div>
  )
}

export default App
