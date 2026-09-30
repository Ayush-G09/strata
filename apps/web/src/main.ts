// In production, set VITE_API_URL at build time to the deployed server's URL (e.g. on Vercel's project settings).
// Locally, with no env var set, this falls back to the dev server's default port on whatever host the page itself
// is served from — unchanged from before.
const API = import.meta.env.VITE_API_URL || `http://${location.hostname}:8890`

const statusEl = document.getElementById('status')!
const statusText = document.getElementById('statusText')!
const docIdInput = document.getElementById('docId') as HTMLInputElement
const docTextInput = document.getElementById('docText') as HTMLTextAreaElement
const chunkStrategySelect = document.getElementById('chunkStrategy') as HTMLSelectElement
const docFileInput = document.getElementById('docFile') as HTMLInputElement
const fileHint = document.getElementById('fileHint')!
const addDocBtn = document.getElementById('addDocBtn') as HTMLButtonElement
const docList = document.getElementById('docList')!
const questionInput = document.getElementById('questionInput') as HTMLInputElement
const askBtn = document.getElementById('askBtn') as HTMLButtonElement
const answerEl = document.getElementById('answer')!
const chunkList = document.getElementById('chunkList')!
const runEvalBtn = document.getElementById('runEvalBtn') as HTMLButtonElement
const evalResults = document.getElementById('evalResults')!

interface RetrievedChunk {
  chunkId: string
  docId: string
  text: string
  fusedScore: number
  vectorRank: number | null
  bm25Rank: number | null
}

let lastRetrieved: RetrievedChunk[] = []
let docCount = 0

async function checkHealth() {
  try {
    const res = await fetch(`${API}/health`)
    if (!res.ok) throw new Error('bad response')
    statusEl.classList.remove('offline')
    statusText.textContent = 'connected'
  } catch {
    statusEl.classList.add('offline')
    statusText.textContent = 'server offline — run `npm run dev -w apps/server`'
  }
}
checkHealth()
setInterval(checkHealth, 5000)

const MAX_UPLOAD_BYTES = 2 * 1024 * 1024 // 2 MB — generous for plain text, keeps a mis-picked large/binary file from hanging the tab

docFileInput.addEventListener('change', async () => {
  const file = docFileInput.files?.[0]
  if (!file) return
  fileHint.textContent = ''
  if (file.size > MAX_UPLOAD_BYTES) {
    fileHint.textContent = `File too large (${(file.size / 1024 / 1024).toFixed(1)} MB) — keep it under 2 MB of plain text.`
    docFileInput.value = ''
    return
  }
  try {
    const text = await file.text()
    docTextInput.value = text
    if (!docIdInput.value.trim()) docIdInput.value = file.name.replace(/\.[^.]+$/, '')
    fileHint.textContent = `Loaded "${file.name}" — ${text.length.toLocaleString()} characters. Edit below if needed, then Add document.`
  } catch (err) {
    fileHint.textContent = `Couldn't read that file: ${err instanceof Error ? err.message : String(err)}`
  } finally {
    docFileInput.value = '' // allow re-selecting the same file again later without a no-op "change" event
  }
})

function setBusy(button: HTMLButtonElement, busy: boolean, label: string) {
  button.disabled = busy
  button.innerHTML = busy ? `<span class="spinner"></span>${label}` : label
}

addDocBtn.addEventListener('click', async () => {
  const id = docIdInput.value.trim() || `doc-${++docCount}`
  const text = docTextInput.value.trim()
  if (!text) return
  setBusy(addDocBtn, true, 'Adding…')
  try {
    const chunking = chunkStrategySelect.value === 'fixed'
      ? { strategy: 'fixed', size: 400, overlap: 60 }
      : { strategy: 'structural', size: 400, overlap: 0 }
    const res = await fetch(`${API}/documents`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ id, text, chunking }),
    })
    const data = await res.json()
    if (!res.ok) throw new Error(data.error ?? 'failed to add document')
    const item = document.createElement('div')
    item.className = 'doc-item'
    item.innerHTML = `<svg class="check" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg><span><b>${escapeHtml(id)}</b> — ${data.chunksAdded} chunk(s)</span>`
    item.style.display = 'flex'
    item.style.alignItems = 'center'
    docList.appendChild(item)
    docTextInput.value = ''
    docIdInput.value = ''
    fileHint.textContent = ''
  } catch (err) {
    alert(err instanceof Error ? err.message : String(err))
  } finally {
    setBusy(addDocBtn, false, 'Add document')
  }
})

async function ask() {
  const question = questionInput.value.trim()
  if (!question) return
  setBusy(askBtn, true, 'Asking…')
  answerEl.innerHTML = '<div class="empty-state"><span class="spinner" style="border-color: rgba(94,198,255,0.25); border-top-color: var(--accent); width:18px; height:18px;"></span><span>Retrieving context and generating an answer…</span></div>'
  try {
    const res = await fetch(`${API}/query`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ question, k: 5 }),
    })
    const data = await res.json()
    if (!res.ok) throw new Error(data.error ?? 'query failed')
    lastRetrieved = data.retrieved as RetrievedChunk[]
    renderAnswer(data.answer.text as string, data.answer.invalidCitationNumbers as number[])
    renderChunks(lastRetrieved)
  } catch (err) {
    answerEl.innerHTML = `<div class="empty-state" style="color: var(--bad);">${escapeHtml(err instanceof Error ? err.message : String(err))}</div>`
  } finally {
    setBusy(askBtn, false, 'Ask')
  }
}
askBtn.addEventListener('click', ask)
questionInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') ask() })

function renderAnswer(text: string, invalidNumbers: number[]) {
  const html = escapeHtml(text).replace(/\[(\d+)\]/g, (_m, nStr) => {
    const n = Number(nStr)
    const chunk = lastRetrieved[n - 1]
    if (!chunk || invalidNumbers.includes(n)) return `<span class="invalid-cite" title="not a real retrieved source">[${n}]</span>`
    return `<span class="cite" data-chunk-id="${chunk.chunkId}">[${n}]</span>`
  })
  answerEl.innerHTML = html
  answerEl.querySelectorAll<HTMLElement>('.cite').forEach((el) => {
    el.addEventListener('click', () => highlightChunk(el.dataset.chunkId!))
  })
}

function renderChunks(chunks: RetrievedChunk[]) {
  chunkList.innerHTML = ''
  chunks.forEach((c) => {
    const card = document.createElement('div')
    card.className = 'chunk-card'
    card.id = `chunk-${cssEscape(c.chunkId)}`
    card.innerHTML = `
      <div class="meta">
        <span class="doc-badge">${escapeHtml(c.docId)}</span>
        <span>fused <b>${c.fusedScore.toFixed(3)}</b></span>
        <span>vec #<b>${c.vectorRank ?? '—'}</b></span>
        <span>bm25 #<b>${c.bm25Rank ?? '—'}</b></span>
      </div>
      <div class="chunk-text">${escapeHtml(c.text)}</div>
    `
    chunkList.appendChild(card)
  })
}

function highlightChunk(chunkId: string) {
  document.querySelectorAll('.chunk-card.hot').forEach((el) => el.classList.remove('hot'))
  document.querySelectorAll('.cite.hot').forEach((el) => el.classList.remove('hot'))
  const card = document.getElementById(`chunk-${cssEscape(chunkId)}`)
  card?.classList.add('hot')
  card?.scrollIntoView({ behavior: 'smooth', block: 'center' })
}

const METRIC_LABELS: Record<string, string> = {
  faithfulness: 'Faithfulness', answerRelevancy: 'Answer relevancy',
  contextPrecision: 'Context precision', contextRecall: 'Context recall', overall: 'Overall',
}

runEvalBtn.addEventListener('click', async () => {
  setBusy(runEvalBtn, true, 'Running…')
  evalResults.innerHTML = '<div class="empty-state"><span class="spinner" style="border-color: rgba(94,198,255,0.25); border-top-color: var(--accent);"></span><span>Running 2 real cases through the connected model— calls a real LLM per claim/chunk judged, so it can take a little while.</span></div>'
  try {
    const res = await fetch(`${API}/eval-sample`)
    const data = await res.json()
    if (!res.ok) throw new Error(data.error ?? 'eval failed')
    const rows = (['faithfulness', 'answerRelevancy', 'contextPrecision', 'contextRecall', 'overall'] as const)
      .map((key) => `<div class="eval-row"><span class="metric-name">${METRIC_LABELS[key]}</span><span class="scores"><span class="hybrid">${pct(data.hybrid.average[key])}</span><span class="vs">vs</span><span class="naive">${pct(data.naive.average[key])}</span></span></div>`)
      .join('')
    evalResults.innerHTML = `
      <div class="eval-legend">
        <span><span class="swatch" style="background:var(--good);"></span>Strata hybrid</span>
        <span><span class="swatch" style="background:var(--text-faint);"></span>Naive baseline</span>
      </div>
      ${rows}`
  } catch (err) {
    evalResults.innerHTML = `<div class="empty-state" style="color: var(--bad);">${escapeHtml(err instanceof Error ? err.message : String(err))}</div>`
  } finally {
    setBusy(runEvalBtn, false, 'Run sample evaluation')
  }
})

function pct(n: number): string {
  return `${(n * 100).toFixed(0)}%`
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function cssEscape(s: string): string {
  return s.replace(/[^a-zA-Z0-9_-]/g, '_')
}
