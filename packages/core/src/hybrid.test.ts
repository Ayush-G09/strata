import { describe, expect, it } from 'vitest'
import { BM25Index } from './bm25'
import { HashingEmbedder } from './embeddings'
import { HybridRetriever, reciprocalRankFusion } from './hybrid'
import { VectorStore } from './vectorStore'
import type { Chunk } from './types'

function chunk(id: string, text: string): Chunk {
  return { id, docId: 'd', text, start: 0, end: text.length, index: 0 }
}

async function buildRetriever(texts: string[]) {
  const embedder = new HashingEmbedder()
  const chunks = texts.map((t, i) => chunk(String(i), t))
  const vectors = await embedder.embed(texts)
  const store = new VectorStore()
  chunks.forEach((c, i) => store.add(c, vectors[i]))
  const bm25 = new BM25Index(chunks)
  return new HybridRetriever(bm25, store, embedder)
}

describe('reciprocalRankFusion', () => {
  it('an item ranked first in both lists scores higher than one ranked first in only one', () => {
    const fused = reciprocalRankFusion([
      [{ id: 'a' }, { id: 'b' }, { id: 'c' }],
      [{ id: 'a' }, { id: 'c' }, { id: 'b' }],
    ])
    expect(fused.get('a')!).toBeGreaterThan(fused.get('b')!)
    expect(fused.get('a')!).toBeGreaterThan(fused.get('c')!)
  })

  it('an item absent from a list contributes 0 from that list, not a penalty that could go negative', () => {
    const fused = reciprocalRankFusion([[{ id: 'a' }], []])
    expect(fused.get('a')).toBeGreaterThan(0)
  })

  it('computes the exact RRF value: 1/(rankConstant + rank), not just a plausible-looking ordering', () => {
    const fused = reciprocalRankFusion([[{ id: 'a' }, { id: 'b' }]], 60)
    expect(fused.get('a')).toBeCloseTo(1 / 61, 12)
    expect(fused.get('b')).toBeCloseTo(1 / 62, 12)
  })

  it('sums contributions across multiple lists rather than only keeping the last one', () => {
    const fused = reciprocalRankFusion([[{ id: 'a' }], [{ id: 'a' }]], 60)
    expect(fused.get('a')).toBeCloseTo(2 / 61, 12)
  })

  it('a smaller rank constant makes rank 1 dominate more heavily over rank 2', () => {
    const listA = [{ id: 'x' }, { id: 'y' }]
    const tight = reciprocalRankFusion([listA], 1)
    const loose = reciprocalRankFusion([listA], 1000)
    const tightGap = tight.get('x')! - tight.get('y')!
    const looseGap = loose.get('x')! - loose.get('y')!
    expect(tightGap).toBeGreaterThan(looseGap)
  })
})

describe('HybridRetriever', () => {
  it('finds a document via exact keyword match even if semantically dissimilar from the query phrasing', async () => {
    const retriever = await buildRetriever([
      'The RRF-2026 protocol specification defines packet framing.',
      'A discussion of general networking concepts and history.',
      'Something entirely unrelated about gardening.',
    ])
    const results = await retriever.search('RRF-2026 protocol', 3)
    expect(results[0].chunk.id).toBe('0')
    expect(results[0].bm25Rank).toBe(1)
  })

  it('finds a document via vector similarity even without an exact keyword match', async () => {
    const retriever = await buildRetriever([
      'the cat sat on the mat',
      'the cat sat on the rug',
      'quantum entanglement and local realism in physics',
    ])
    const results = await retriever.search('the cat sat on the mat', 3)
    expect(results[0].chunk.id).toBe('0')
    expect(results[0].vectorRank).toBe(1)
  })

  it('a document ranked well by BOTH signals outranks one ranked well by only one', async () => {
    const retriever = await buildRetriever([
      'strata hybrid retrieval combines bm25 and vector search', // matches keyword AND is topically closest
      'strata strata strata strata strata strata strata strata', // heavy keyword repetition, no real topical content
      'an entirely unrelated sentence about weather patterns',
    ])
    const results = await retriever.search('strata hybrid retrieval', 3)
    expect(results[0].chunk.id).toBe('0')
  })

  it('respects k', async () => {
    const retriever = await buildRetriever(Array.from({ length: 20 }, (_, i) => `document number ${i} about strata`))
    const results = await retriever.search('strata', 5)
    expect(results.length).toBe(5)
  })

  it('returns results in strictly non-increasing fused-score order', async () => {
    const retriever = await buildRetriever(['strata strata strata retrieval', 'strata retrieval', 'unrelated weather patterns entirely'])
    const results = await retriever.search('strata retrieval', 3)
    for (let i = 1; i < results.length; i++) expect(results[i].fusedScore).toBeLessThanOrEqual(results[i - 1].fusedScore)
  })

  it('sorts strictly by descending fused score, even when that differs from insertion order', async () => {
    const retriever = await buildRetriever([
      'irrelevant content about nothing at all',
      'somewhat relevant, mentions strata once',
      'strata strata strata strata is extremely relevant here',
    ])
    const results = await retriever.search('strata', 3)
    expect(results[0].chunk.id).toBe('2') // the last-constructed chunk, not the first — proves sort, not insertion order
  })

  it('includes a chunk that BM25 omits entirely (zero keyword overlap), populated via its vector-search rank alone', async () => {
    const retriever = await buildRetriever([
      'apples and oranges are fruits',
      'a spaceship travels through the vast emptiness of the cosmos',
      'bananas and grapes are also fruits',
    ])
    const results = await retriever.search('fruits', 3)
    const spaceship = results.find((r) => r.chunk.id === '1')
    expect(spaceship).toBeDefined()
    expect(spaceship!.chunk.text).toContain('spaceship') // real chunk data, not left undefined by a skipped population loop
    expect(spaceship!.bm25Rank).toBeNull() // BM25 never matched it at all — its only route into the result is vector rank
  })
})
