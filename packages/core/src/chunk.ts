import type { Chunk, ChunkingOptions, Document } from './types'

/** Splits a document into fixed-size, overlapping windows measured in characters. Simple and predictable —
 * the right choice when the content has no reliable structure (raw text dumps, transcripts). */
function chunkFixed(doc: Document, size: number, overlap: number): Chunk[] {
  if (size <= 0) throw new RangeError('size must be positive')
  if (overlap < 0 || overlap >= size) throw new RangeError('overlap must be >= 0 and < size')

  const chunks: Chunk[] = []
  const step = size - overlap
  const text = doc.text
  let start = 0
  let index = 0
  while (start < text.length) {
    const end = Math.min(start + size, text.length)
    chunks.push({ id: `${doc.id}:${index}`, docId: doc.id, text: text.slice(start, end), start, end, index })
    index++
    if (end === text.length) break
    start += step
  }
  return chunks
}

/** Finds paragraphs (text separated by one or more blank lines) as contiguous, non-overlapping spans of the
 * ORIGINAL text — the separators themselves belong to no paragraph, but every character of actual content is
 * accounted for in exactly one span. */
function findParagraphs(text: string): { start: number; end: number }[] {
  const spans: { start: number; end: number }[] = []
  const sepRe = /\n{2,}/g
  let cursor = 0
  let m: RegExpExecArray | null
  while ((m = sepRe.exec(text))) {
    if (m.index > cursor) spans.push({ start: cursor, end: m.index })
    cursor = m.index + m[0].length
  }
  if (cursor < text.length) spans.push({ start: cursor, end: text.length })
  return spans
}

/** Greedily merges consecutive paragraphs up to `size` characters, so a chunk boundary lands on a paragraph
 * break whenever possible instead of splitting a sentence in half. A single paragraph longer than `size` is
 * itself split with `chunkFixed`-style fixed windows (no overlap) — it can't be merged with anything, but it
 * still shouldn't produce one enormous, unsearchable chunk. */
function chunkStructural(doc: Document, size: number): Chunk[] {
  if (size <= 0) throw new RangeError('size must be positive')
  const paragraphs = findParagraphs(doc.text)
  const chunks: Chunk[] = []
  let index = 0

  const flush = (start: number, end: number) => {
    if (end <= start) return
    chunks.push({ id: `${doc.id}:${index}`, docId: doc.id, text: doc.text.slice(start, end), start, end, index })
    index++
  }

  let groupStart: number | null = null
  let groupEnd = 0

  for (const p of paragraphs) {
    const paragraphLen = p.end - p.start
    if (paragraphLen > size) {
      // flush whatever was accumulating, then hard-split this oversized paragraph on its own
      if (groupStart !== null) { flush(groupStart, groupEnd); groupStart = null }
      let s = p.start
      while (s < p.end) {
        const e = Math.min(s + size, p.end)
        flush(s, e)
        s = e
      }
      continue
    }
    if (groupStart === null) {
      groupStart = p.start
      groupEnd = p.end
      continue
    }
    if (p.end - groupStart <= size) {
      groupEnd = p.end // still fits, merge it in
    } else {
      flush(groupStart, groupEnd)
      groupStart = p.start
      groupEnd = p.end
    }
  }
  if (groupStart !== null) flush(groupStart, groupEnd)
  return chunks
}

export function chunkDocument(doc: Document, options: ChunkingOptions): Chunk[] {
  if (doc.text.length === 0) return []
  return options.strategy === 'fixed'
    ? chunkFixed(doc, options.size, options.overlap)
    : chunkStructural(doc, options.size)
}
