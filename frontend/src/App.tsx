import { useState, useRef, useEffect } from 'react'
import { FiUpload, FiSearch, FiScissors, FiClock, FiSettings, FiActivity, FiChevronDown, FiChevronRight } from 'react-icons/fi'

const API = '/api'

type Page = 'upload' | 'preview' | 'manual' | 'configurations' | 'monitoring'

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
  const [singlePdfOpen, setSinglePdfOpen] = useState(true)
  const [batchSplitsOpen, setBatchSplitsOpen] = useState(false)
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
  const [segmentTitles, setSegmentTitles] = useState<string[]>([])
  const [segmentFiles, setSegmentFiles] = useState<string[]>([])
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
  const [recents, setRecents] = useState<Array<{ filename: string; original_name: string; page_count: number; modified: number }>>([])
  const fileInputRef = useRef<HTMLInputElement>(null)
  
  // Configuration state
  const [batchSize, setBatchSize] = useState(100)
  const [gpuType, setGpuType] = useState('nvidia-t4')
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.70)
  const [awsRegion, setAwsRegion] = useState('us-east-1')
  const [instanceType, setInstanceType] = useState('g4dn.xlarge')
  const [maxConcurrentJobs, setMaxConcurrentJobs] = useState(10)
  const [ocrDpi, setOcrDpi] = useState(250)
  const [saveConfig, setSaveConfig] = useState(false)

  useEffect(() => {
    if (!jobId) return
    // Poll if running, or check once if idle (to catch completed OCR after refresh)
    if (ocrStatus !== 'running' && ocrStatus !== 'idle') return
    
    let intervalId: NodeJS.Timeout | null = null
    let isMounted = true
    const mainAbortController = new AbortController()
    
    const poll = async () => {
      if (!isMounted) return
      // Create a new abort controller for each request
      const requestAbortController = new AbortController()
      const timeoutId = setTimeout(() => {
        requestAbortController.abort()
      }, 10000) // 10 second timeout per request
      
      try {
        const r = await fetch(`${API}/ocr-progress/${jobId}`, {
          signal: requestAbortController.signal,
        })
        clearTimeout(timeoutId)
        if (!r.ok) return
        const d = await r.json()
        if (!isMounted) return
        
        if (d.status === 'not_found') {
          if (isMounted) {
            setOcrStatus('error')
            setOcrError('Job not found. Please upload a new PDF.')
            setJobId(null)
          }
          return
        }
        if (isMounted) {
          setOcrProgress(d.progress ?? 0)
          setOcrCurrent(d.current ?? 0)
          setOcrTotal(d.total ?? 0)
        }
        if (d.status === 'done') {
          if (isMounted) {
            setOcrStatus('done')
          }
          if (intervalId) {
            clearInterval(intervalId)
            intervalId = null
          }
          return
        }
        if (d.status === 'error') {
          if (isMounted) {
            setOcrStatus('error')
            setOcrError(d.error ?? 'OCR failed')
          }
          if (intervalId) {
            clearInterval(intervalId)
            intervalId = null
          }
          return
        }
        if (d.status === 'running' && ocrStatus === 'idle') {
          if (isMounted) {
            setOcrStatus('running') // Start polling if OCR is running
          }
        }
      } catch (err: any) {
        clearTimeout(timeoutId)
        // Ignore abort errors
        if (err?.name === 'AbortError') return
        // Ignore other errors silently to avoid console spam
      }
    }
    
    // If idle, check once immediately
    if (ocrStatus === 'idle') {
      poll()
      return () => {
        isMounted = false
        mainAbortController.abort()
      }
    }
    
    // If running, poll regularly (increased interval to reduce load)
    poll() // Poll immediately
    intervalId = setInterval(poll, 2000) // Increased from 800ms to 2000ms
    
    const onVisible = () => {
      if (document.visibilityState === 'visible' && isMounted) {
        poll()
      }
    }
    document.addEventListener('visibilitychange', onVisible)
    
    return () => {
      isMounted = false
      mainAbortController.abort()
      if (intervalId) {
        clearInterval(intervalId)
      }
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [jobId, ocrStatus])

  useEffect(() => {
    if (!jobId || splitStatus !== 'running') return
    
    let intervalId: NodeJS.Timeout | null = null
    let isMounted = true
    const mainAbortController = new AbortController()
    
    const poll = async () => {
      if (!isMounted) return
      // Create a new abort controller for each request
      const requestAbortController = new AbortController()
      const timeoutId = setTimeout(() => {
        requestAbortController.abort()
      }, 10000) // 10 second timeout per request
      
      try {
        const r = await fetch(`${API}/split-progress/${jobId}`, {
          signal: requestAbortController.signal,
        })
        clearTimeout(timeoutId)
        if (!r.ok) return
        const d = await r.json()
        if (!isMounted) return
        
        if (d.status === 'not_found') {
          if (isMounted) {
            setSplitStatus('error')
            setSplitError('Job not found. Please upload a new PDF.')
            setJobId(null)
          }
          if (intervalId) {
            clearInterval(intervalId)
            intervalId = null
          }
          return
        }
        if (isMounted) {
          setSplitProgress(d.progress ?? 0)
        }
        if (d.status === 'done') {
          if (isMounted) {
            setSplitStatus('done')
            try {
              const listAbortController = new AbortController()
              const listTimeoutId = setTimeout(() => listAbortController.abort(), 10000)
              const list = await fetch(`${API}/splits/${jobId}`, {
                signal: listAbortController.signal,
              })
              clearTimeout(listTimeoutId)
              if (list.ok && isMounted) {
                const listData = await list.json()
                setSplitFiles(listData.files ?? [])
              }
            } catch { /* ignore */ }
          }
          if (intervalId) {
            clearInterval(intervalId)
            intervalId = null
          }
          return
        }
        if (d.status === 'error') {
          if (isMounted) {
            setSplitStatus('error')
            setSplitError(d.error ?? 'AI split failed')
          }
          if (intervalId) {
            clearInterval(intervalId)
            intervalId = null
          }
          return
        }
      } catch (err: any) {
        clearTimeout(timeoutId)
        if (err?.name === 'AbortError') return
        // Ignore other errors
      }
    }
    
    poll() // Poll immediately
    intervalId = setInterval(poll, 2000) // Increased from 800ms to 2000ms
    
    return () => {
      isMounted = false
      mainAbortController.abort()
      if (intervalId) {
        clearInterval(intervalId)
      }
    }
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
    // Reset all state for new upload
    setOcrStatus('running')
    setOcrProgress(0)
    setOcrCurrent(0)
    setOcrTotal(0)
    setOcrError(null)
    setSplitStatus('idle')
    setSplitProgress(0)
    setSplitError(null)
    setSplitFiles([])
    setSegments([])
    setSegmentTitles([])
    setSegmentFiles([])
    setPageTextPages([])
    setManualFilenames(new Set())
    setManualMeta({})
    const form = new FormData()
    form.append('file', file)
    try {
      const r = await fetch(`${API}/upload-pdf`, { method: 'POST', body: form })
      const d = await r.json()
      if (d.job_id) {
        setJobId(d.job_id)
        loadRecents() // Refresh recents after new upload
      } else {
        setOcrStatus('error')
        setOcrError(d.error ?? 'Upload failed')
        setJobId(null) // Clear jobId on error
      }
    } catch (err) {
      setOcrStatus('error')
      setOcrError(String(err))
      setJobId(null) // Clear jobId on error
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
    if (!jobId) {
      // Clear split state if no jobId
      setSplitFiles([])
      setSegments([])
      setSegmentTitles([])
      setSegmentFiles([])
      setPageTextPages([])
      setManualFilenames(new Set())
      setManualMeta({})
      return
    }
    try {
      const [splitsRes, segRes, ptRes] = await Promise.all([
        fetch(`${API}/splits/${jobId}`),
        fetch(`${API}/segments/${jobId}`),
        fetch(`${API}/page-text/${jobId}`),
      ])
      // Check if any request failed (e.g., 404)
      if (!splitsRes.ok || !segRes.ok || !ptRes.ok) {
        // Job might not exist or not ready yet
        setSplitFiles([])
        setSegments([])
        setSegmentTitles([])
        setSegmentFiles([])
        setPageTextPages([])
        setManualFilenames(new Set())
        setManualMeta({})
        return
      }
      const splitsData = await splitsRes.json()
      const segData = await segRes.json()
      const ptData = await ptRes.json()
      setSplitFiles(splitsData.files ?? [])
      setManualFilenames(new Set(splitsData.manual_filenames ?? []))
      setManualMeta(splitsData.manual_meta ?? {})
      setSegments(segData.segments ?? [])
      setSegmentTitles(segData.titles ?? [])
      setSegmentFiles(segData.output_files ?? [])
      setPageTextPages(ptData.pages ?? [])
      setReconTabIndex(0)
    } catch {
      // On error, clear split state
      setSplitFiles([])
      setSegments([])
      setSegmentTitles([])
      setSegmentFiles([])
      setPageTextPages([])
      setManualFilenames(new Set())
      setManualMeta({})
    }
  }

  useEffect(() => {
    if (page === 'preview' && jobId) loadSplits()
  }, [page, jobId])

  useEffect(() => {
    loadRecents()
  }, [])

  const loadRecents = async () => {
    try {
      const r = await fetch(`${API}/recents`)
      const d = await r.json()
      setRecents(d.recents ?? [])
    } catch { setRecents([]) }
  }

  const loadRecent = async (filename: string) => {
    try {
      const r = await fetch(`${API}/recents/${encodeURIComponent(filename)}`)
      const d = await r.json()
      if (d.job_id) {
        setJobId(d.job_id)
        setOcrStatus('done')
        setOcrProgress(100)
        setSelectedFileName(d.original_name)
        setSplitStatus('idle')
        setPage('upload')
        loadRecents() // Refresh recents list
      }
    } catch (err) {
      alert('Failed to load recent PDF')
    }
  }

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
    // Find title from segments by matching filename
    const fileIndex = segmentFiles.indexOf(filename)
    if (fileIndex >= 0 && segmentTitles[fileIndex]) {
      // Replace underscores with spaces for display
      return segmentTitles[fileIndex].replace(/_/g, ' ')
    }
    // Fallback: extract from filename (remove GUID and extension, convert camelCase to Title Case)
    const base = filename.replace(/\.pdf$/i, '')
    const parts = base.split('_')
    if (parts.length > 1) {
      const namePart = parts.slice(1).join('_')
      // Convert camelCase to Title Case
      return namePart.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase()).trim()
    }
    return base
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
          <div className="sidebar-section">
            <button className="sidebar-dropdown-header" onClick={() => setSinglePdfOpen(!singlePdfOpen)}>
              {singlePdfOpen ? <FiChevronDown style={{ marginRight: 8, fontSize: 14 }} /> : <FiChevronRight style={{ marginRight: 8, fontSize: 14 }} />}
              <span>Single PDF Splits</span>
            </button>
            {singlePdfOpen && (
              <div className="sidebar-dropdown-content">
                <button className={page === 'upload' ? 'active' : ''} onClick={() => setPage('upload')}>
                  <FiUpload style={{ marginRight: 8, fontSize: 16 }} />
                  Upload & AI split
                </button>
                <button className={page === 'preview' ? 'active' : ''} onClick={() => setPage('preview')}>
                  <FiSearch style={{ marginRight: 8, fontSize: 16 }} />
                  Preview
                </button>
                <button className={page === 'manual' ? 'active' : ''} onClick={() => setPage('manual')}>
                  <FiScissors style={{ marginRight: 8, fontSize: 16 }} />
                  Manual Splits
                </button>
              </div>
            )}
          </div>
          <div className="sidebar-section">
            <button className="sidebar-dropdown-header" onClick={() => setBatchSplitsOpen(!batchSplitsOpen)}>
              {batchSplitsOpen ? <FiChevronDown style={{ marginRight: 8, fontSize: 14 }} /> : <FiChevronRight style={{ marginRight: 8, fontSize: 14 }} />}
              <span>Batch Splits</span>
            </button>
            {batchSplitsOpen && (
              <div className="sidebar-dropdown-content">
                <button className={page === 'configurations' ? 'active' : ''} onClick={() => setPage('configurations' as Page)}>
                  <FiSettings style={{ marginRight: 8, fontSize: 16 }} />
                  Configurations
                </button>
                <button className={page === 'monitoring' ? 'active' : ''} onClick={() => setPage('monitoring' as Page)}>
                  <FiActivity style={{ marginRight: 8, fontSize: 16 }} />
                  Monitoring
                </button>
              </div>
            )}
          </div>
        </nav>
        <main className="main">
          <div className="main-content-with-recents">
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
              <div className="preview-header">
                <h1 className="preview-title">Preview</h1>
                {jobId && splitFiles.length > 0 && (
                  <div className="preview-header-actions">
                    <button className="btn" onClick={loadSplits}>Refresh list</button>
                    <button type="button" className="btn" style={{ background: 'var(--red)', color: '#fff' }} onClick={deleteAllSplits}>Delete all splits</button>
                  </div>
                )}
              </div>
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
                  <div className="recon-view-toggle">
                    <span
                      className={reconViewMode === 'text' ? 'selected' : ''}
                      onClick={() => setReconViewMode('text')}
                      role="button"
                      tabIndex={0}
                      onKeyDown={(e) => e.key === 'Enter' && setReconViewMode('text')}
                    >
                      List View
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
                          <a href={previewUrl(f)} target="_blank" rel="noopener noreferrer" className="recon-list-link">Preview</a>
                          <a href={downloadUrl(f)} download={f} className="recon-list-link">Download</a>
                          <button type="button" className="recon-list-link recon-list-link-danger" onClick={() => deleteSplit(f)}>Delete</button>
                        </li>
                      ))}
                    </ul>
                  )}
                </>
              )}
            </div>
          )}

          {page === 'configurations' && (
            <div className="page config-page">
              <h1>Batch Processing Configurations</h1>
              <p className="muted" style={{ marginBottom: 24 }}>
                Configure settings for processing large batches of PDFs (e.g., 30K PDFs). 
                These settings will be applied to all batch jobs.
              </p>
              
              <div className="config-form">
                <div className="config-section">
                  <h2>Processing Settings</h2>
                  <div className="form-group">
                    <label>Batch Size</label>
                    <input 
                      type="number" 
                      min="1" 
                      max="1000" 
                      value={batchSize} 
                      onChange={(e) => setBatchSize(parseInt(e.target.value) || 100)}
                      placeholder="Number of PDFs per batch"
                    />
                    <span className="form-hint">Number of PDFs to process in each batch (recommended: 50-200)</span>
                  </div>
                  
                  <div className="form-group">
                    <label>Max Concurrent Jobs</label>
                    <input 
                      type="number" 
                      min="1" 
                      max="50" 
                      value={maxConcurrentJobs} 
                      onChange={(e) => setMaxConcurrentJobs(parseInt(e.target.value) || 10)}
                      placeholder="Maximum parallel jobs"
                    />
                    <span className="form-hint">Number of jobs to run simultaneously</span>
                  </div>
                  
                  <div className="form-group">
                    <label>Confidence Threshold</label>
                    <input 
                      type="number" 
                      min="0" 
                      max="1" 
                      step="0.01"
                      value={confidenceThreshold} 
                      onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value) || 0.70)}
                      placeholder="0.70"
                    />
                    <span className="form-hint">AI confidence score threshold for document boundary detection (0.0 - 1.0)</span>
                  </div>
                  
                  <div className="form-group">
                    <label>OCR DPI</label>
                    <input 
                      type="number" 
                      min="150" 
                      max="300" 
                      step="50"
                      value={ocrDpi} 
                      onChange={(e) => setOcrDpi(parseInt(e.target.value) || 250)}
                      placeholder="250"
                    />
                    <span className="form-hint">Resolution for OCR processing (higher = better quality, slower)</span>
                  </div>
                </div>
                
                <div className="config-section">
                  <h2>AWS Infrastructure</h2>
                  <div className="form-group">
                    <label>AWS Region</label>
                    <select value={awsRegion} onChange={(e) => setAwsRegion(e.target.value)}>
                      <option value="us-east-1">US East (N. Virginia) - us-east-1</option>
                      <option value="us-west-2">US West (Oregon) - us-west-2</option>
                      <option value="eu-west-1">Europe (Ireland) - eu-west-1</option>
                      <option value="ap-southeast-1">Asia Pacific (Singapore) - ap-southeast-1</option>
                    </select>
                    <span className="form-hint">AWS region for batch processing</span>
                  </div>
                  
                  <div className="form-group">
                    <label>EC2 Instance Type</label>
                    <select value={instanceType} onChange={(e) => setInstanceType(e.target.value)}>
                      <option value="g4dn.xlarge">g4dn.xlarge (1x T4 GPU, 4 vCPU, 16GB RAM)</option>
                      <option value="g4dn.2xlarge">g4dn.2xlarge (1x T4 GPU, 8 vCPU, 32GB RAM)</option>
                      <option value="g5.xlarge">g5.xlarge (1x A10G GPU, 4 vCPU, 16GB RAM)</option>
                      <option value="g5.2xlarge">g5.2xlarge (1x A10G GPU, 8 vCPU, 32GB RAM)</option>
                      <option value="p3.2xlarge">p3.2xlarge (1x V100 GPU, 8 vCPU, 61GB RAM)</option>
                    </select>
                    <span className="form-hint">EC2 instance type with GPU support</span>
                  </div>
                  
                  <div className="form-group">
                    <label>GPU Type</label>
                    <select value={gpuType} onChange={(e) => setGpuType(e.target.value)}>
                      <option value="nvidia-t4">NVIDIA T4</option>
                      <option value="nvidia-a10g">NVIDIA A10G</option>
                      <option value="nvidia-v100">NVIDIA V100</option>
                      <option value="nvidia-a100">NVIDIA A100</option>
                    </select>
                    <span className="form-hint">GPU type for CUDA processing</span>
                  </div>
                </div>
                
                <div className="config-section">
                  <h2>Storage & Output</h2>
                  <div className="form-group">
                    <label>S3 Bucket for Input PDFs</label>
                    <input 
                      type="text" 
                      value="seagate-pdf-inputs" 
                      readOnly
                      placeholder="S3 bucket name"
                    />
                    <span className="form-hint">S3 bucket where input PDFs are stored</span>
                  </div>
                  
                  <div className="form-group">
                    <label>S3 Bucket for Output Splits</label>
                    <input 
                      type="text" 
                      value="seagate-pdf-splits" 
                      readOnly
                      placeholder="S3 bucket name"
                    />
                    <span className="form-hint">S3 bucket where split PDFs will be saved</span>
                  </div>
                  
                  <div className="form-group">
                    <label>Output Format</label>
                    <select defaultValue="guid_camelcase">
                      <option value="guid_camelcase">GUID_CamelCase.pdf</option>
                      <option value="guid_original">GUID_OriginalName.pdf</option>
                      <option value="camelcase_only">CamelCase.pdf</option>
                    </select>
                    <span className="form-hint">Naming convention for output files</span>
                  </div>
                </div>
                
                <div className="config-actions">
                  <button className="btn" onClick={() => setSaveConfig(true)}>
                    Save Configuration
                  </button>
                  {saveConfig && (
                    <span className="success" style={{ marginLeft: 12 }}>
                      Configuration saved successfully!
                    </span>
                  )}
                </div>
              </div>
            </div>
          )}

          {page === 'monitoring' && (
            <div className="page monitoring-page">
              <h1>Batch Job Monitoring</h1>
              <p className="muted" style={{ marginBottom: 24 }}>
                Monitor the status and progress of batch PDF processing jobs.
              </p>
              
              <div className="monitoring-stats">
                <div className="stat-card">
                  <div className="stat-value">0</div>
                  <div className="stat-label">Active Jobs</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">0</div>
                  <div className="stat-label">Queued Jobs</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">0</div>
                  <div className="stat-label">Completed</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">0</div>
                  <div className="stat-label">Failed</div>
                </div>
              </div>
              
              <div className="monitoring-jobs">
                <h2>Recent Batch Jobs</h2>
                <div className="jobs-list">
                  <div className="job-item">
                    <div className="job-info">
                      <div className="job-name">No batch jobs yet</div>
                      <div className="job-meta">Upload PDFs to start a batch job</div>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="monitoring-actions">
                <button className="btn" onClick={() => alert('Batch job creation will be implemented')}>
                  Create New Batch Job
                </button>
                <button className="btn" style={{ background: 'var(--dark-blue)', marginLeft: 12 }}>
                  Refresh Status
                </button>
              </div>
            </div>
          )}
          </div>
          {page === 'upload' && (
            <aside className="recents-panel">
              <h2>
                <FiClock style={{ marginRight: 8, fontSize: 18, verticalAlign: 'middle' }} />
                Recents
              </h2>
              {recents.length === 0 ? (
                <p className="muted" style={{ fontSize: 12 }}>No recent PDFs</p>
              ) : (
                <ul className="recents-list">
                  {recents.map((r) => (
                    <li key={r.filename} className="recent-item" onClick={() => loadRecent(r.filename)}>
                      <div className="recent-name">{r.original_name}</div>
                      <div className="recent-meta">{r.page_count} pages</div>
                    </li>
                  ))}
                </ul>
              )}
            </aside>
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
