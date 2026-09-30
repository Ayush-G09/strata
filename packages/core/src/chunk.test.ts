import { describe, expect, it } from 'vitest'
import { chunkDocument } from './chunk'
import type { Document } from './types'

function mulberry32(seed: number) {
  let a = seed
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function randomText(rng: () => number, paragraphs: number): string {
  const words = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog', 'lorem', 'ipsum', 'strata', 'chunk']
  const parts: string[] = []
  for (let p = 0; p < paragraphs; p++) {
    const sentences = 1 + Math.floor(rng() * 5)
    const sentenceParts: string[] = []
    for (let s = 0; s < sentences; s++) {
      const len = 3 + Math.floor(rng() * 10)
      const w: string[] = []
      for (let i = 0; i < len; i++) w.push(words[Math.floor(rng() * words.length)])
      sentenceParts.push(w.join(' ') + '.')
    }
    parts.push(sentenceParts.join(' '))
  }
  // join with a randomised number of blank lines, so paragraph detection has to handle 1..3 newlines' worth of gap
  return parts.map((p, i) => (i === 0 ? p : '\n'.repeat(1 + Math.floor(rng() * 3)) + p)).join('')
}

describe('chunkDocument — fixed strategy', () => {
  it('every chunk exactly matches the original text at its recorded offsets', () => {
    const doc: Document = { id: 'd1', text: 'abcdefghijklmnopqrstuvwxyz' }
    const chunks = chunkDocument(doc, { strategy: 'fixed', size: 10, overlap: 3 })
    for (const c of chunks) expect(c.text).toBe(doc.text.slice(c.start, c.end))
  })

  it('consecutive chunks overlap by exactly the configured amount', () => {
    const doc: Document = { id: 'd1', text: 'x'.repeat(97) }
    const chunks = chunkDocument(doc, { strategy: 'fixed', size: 20, overlap: 5 })
    for (let i = 1; i < chunks.length; i++) {
      expect(chunks[i].start).toBe(chunks[i - 1].end - 5)
    }
  })

  it('covers the entire document with no gap, for many random sizes/overlaps', () => {
    const rng = mulberry32(7)
    for (let trial = 0; trial < 40; trial++) {
      const text = randomText(rng, 1 + Math.floor(rng() * 5))
      const size = 20 + Math.floor(rng() * 200)
      const overlap = Math.floor(rng() * Math.min(size - 1, 50))
      const doc: Document = { id: 'd', text }
      const chunks = chunkDocument(doc, { strategy: 'fixed', size, overlap })
      expect(chunks[0].start).toBe(0)
      expect(chunks[chunks.length - 1].end).toBe(text.length)
      for (let i = 1; i < chunks.length; i++) expect(chunks[i].start).toBeLessThanOrEqual(chunks[i - 1].end)
    }
  })

  it('rejects overlap >= size and non-positive size, with messages naming the problem', () => {
    const doc: Document = { id: 'd', text: 'hello world' }
    expect(() => chunkDocument(doc, { strategy: 'fixed', size: 10, overlap: 10 })).toThrow(/overlap/)
    expect(() => chunkDocument(doc, { strategy: 'fixed', size: 10, overlap: 11 })).toThrow(/overlap/) // overlap > size too
    expect(() => chunkDocument(doc, { strategy: 'fixed', size: 0, overlap: 0 })).toThrow(/size/) // exactly the boundary, size === 0
    expect(() => chunkDocument(doc, { strategy: 'fixed', size: -5, overlap: 0 })).toThrow(/size/)
  })

  it('an empty document produces no chunks', () => {
    expect(chunkDocument({ id: 'd', text: '' }, { strategy: 'fixed', size: 10, overlap: 0 })).toEqual([])
  })

  it('assigns each chunk an id containing the document id and its own index, and never stops one short of the end', () => {
    const doc: Document = { id: 'doc7', text: 'x'.repeat(25) }
    const chunks = chunkDocument(doc, { strategy: 'fixed', size: 10, overlap: 0 })
    expect(chunks.map((c) => c.id)).toEqual(['doc7:0', 'doc7:1', 'doc7:2'])
    expect(chunks[chunks.length - 1].end).toBe(25) // the loop must not stop one step early and leave a tail uncovered
  })
})

describe('chunkDocument — structural strategy', () => {
  it('every chunk exactly matches the original text at its recorded offsets', () => {
    const doc: Document = { id: 'd1', text: 'First paragraph here.\n\nSecond paragraph, a bit longer than the first one.\n\n\nThird.' }
    const chunks = chunkDocument(doc, { strategy: 'structural', size: 40, overlap: 0 })
    for (const c of chunks) expect(c.text).toBe(doc.text.slice(c.start, c.end))
  })

  it('never produces a chunk larger than the configured size, for many random documents', () => {
    const rng = mulberry32(99)
    for (let trial = 0; trial < 40; trial++) {
      const text = randomText(rng, 1 + Math.floor(rng() * 10))
      const size = 30 + Math.floor(rng() * 150)
      const chunks = chunkDocument({ id: 'd', text }, { strategy: 'structural', size, overlap: 0 })
      for (const c of chunks) expect(c.end - c.start).toBeLessThanOrEqual(size)
    }
  })

  it('merges short consecutive paragraphs into one chunk when they fit', () => {
    const doc: Document = { id: 'd', text: 'One.\n\nTwo.\n\nThree.' }
    const chunks = chunkDocument(doc, { strategy: 'structural', size: 100, overlap: 0 })
    expect(chunks.length).toBe(1)
    expect(chunks[0].text).toBe(doc.text)
  })

  it('requires two or more consecutive newlines to start a new paragraph — a single newline stays mid-paragraph', () => {
    const doc: Document = { id: 'd', text: 'First line.\nStill the same paragraph.\n\nA genuinely new paragraph.' }
    const chunks = chunkDocument(doc, { strategy: 'structural', size: 500, overlap: 0 })
    // with a generous size limit everything merges into one chunk regardless, so instead force a size that can
    // only hold ONE paragraph at a time, which reveals how many paragraphs were actually detected
    const tight = chunkDocument(doc, { strategy: 'structural', size: 39, overlap: 0 })
    expect(tight.length).toBe(2) // exactly two paragraphs: the single-newline text is NOT itself a paragraph break
    expect(tight[0].text).toBe('First line.\nStill the same paragraph.')
    expect(chunks.length).toBe(1)
  })

  it('a paragraph exactly at the size limit is kept whole, not split (boundary is inclusive)', () => {
    const text = 'x'.repeat(50)
    const chunks = chunkDocument({ id: 'd', text }, { strategy: 'structural', size: 50, overlap: 0 })
    expect(chunks.length).toBe(1)
    expect(chunks[0].text).toBe(text)
  })

  it('a paragraph exactly one character over the size limit IS split', () => {
    const text = 'x'.repeat(51)
    const chunks = chunkDocument({ id: 'd', text }, { strategy: 'structural', size: 50, overlap: 0 })
    expect(chunks.length).toBeGreaterThan(1)
  })

  it('two consecutive paragraphs that together exactly fill the size limit are merged, not split', () => {
    const doc: Document = { id: 'd', text: `${'a'.repeat(20)}\n\n${'b'.repeat(22)}` } // 20 + 2 (blank line) + 22 = 44
    const chunks = chunkDocument(doc, { strategy: 'structural', size: 44, overlap: 0 })
    expect(chunks.length).toBe(1)
  })

  it('a document starting with a blank line produces no empty leading chunk', () => {
    const doc: Document = { id: 'd', text: '\n\nActual content starts here.' }
    const chunks = chunkDocument(doc, { strategy: 'structural', size: 100, overlap: 0 })
    expect(chunks.length).toBe(1)
    expect(chunks[0].text).toBe('Actual content starts here.')
  })

  it('a document ending exactly at a blank-line separator produces no empty trailing chunk', () => {
    const doc: Document = { id: 'd', text: 'Content here.\n\n' }
    const chunks = chunkDocument(doc, { strategy: 'structural', size: 100, overlap: 0 })
    expect(chunks.length).toBe(1)
    expect(chunks[0].text).toBe('Content here.')
  })

  it('rejects a non-positive size, with a message naming the problem', () => {
    expect(() => chunkDocument({ id: 'd', text: 'x' }, { strategy: 'structural', size: 0, overlap: 0 })).toThrow(/size/)
  })

  it('gives every chunk an id containing the document id and its own sequential index', () => {
    const doc: Document = { id: 'report', text: 'a.\n\nb.\n\nc.' }
    const chunks = chunkDocument(doc, { strategy: 'structural', size: 3, overlap: 0 })
    expect(chunks.map((c) => c.id)).toEqual(chunks.map((c, i) => `report:${i}`))
  })

  it('splits a paragraph that alone exceeds the size limit, instead of producing one giant chunk', () => {
    const longParagraph = 'word '.repeat(200).trim()
    const doc: Document = { id: 'd', text: longParagraph }
    const chunks = chunkDocument(doc, { strategy: 'structural', size: 50, overlap: 0 })
    expect(chunks.length).toBeGreaterThan(1)
    for (const c of chunks) expect(c.end - c.start).toBeLessThanOrEqual(50)
  })

  it('does not lose or duplicate any non-whitespace content of the original document', () => {
    // Merged chunks retain their internal blank-line separators (they're part of the span between the first and
    // last paragraph in the group), so an exact whitespace-preserving comparison isn't meaningful here — only a
    // dropped separator BETWEEN chunks is expected. Stripping all whitespace from both sides still catches the
    // failure mode that actually matters: losing or duplicating real content.
    const rng = mulberry32(2024)
    for (let trial = 0; trial < 40; trial++) {
      const text = randomText(rng, 1 + Math.floor(rng() * 8))
      const chunks = chunkDocument({ id: 'd', text }, { strategy: 'structural', size: 20 + Math.floor(rng() * 100), overlap: 0 })
      const reconstructed = chunks.map((c) => c.text).join('')
      expect(reconstructed.replace(/\s+/g, '')).toBe(text.replace(/\s+/g, ''))
    }
  })

  it('assigns each chunk a sequential index starting at 0', () => {
    const doc: Document = { id: 'd', text: 'a.\n\nb.\n\nc.\n\nd.' }
    const chunks = chunkDocument(doc, { strategy: 'structural', size: 3, overlap: 0 })
    chunks.forEach((c, i) => expect(c.index).toBe(i))
  })
})
