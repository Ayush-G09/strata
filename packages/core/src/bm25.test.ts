import { describe, expect, it } from 'vitest'
import { BM25Index } from './bm25'
import type { Chunk } from './types'

function chunk(id: string, text: string): Chunk {
  return { id, docId: 'd', text, start: 0, end: text.length, index: 0 }
}

describe('BM25Index', () => {
  it('ranks a document mentioning the exact query term above one that does not', () => {
    const idx = new BM25Index([
      chunk('a', 'the reciprocal rank fusion algorithm combines two ranked lists'),
      chunk('b', 'the weather today is sunny with a light breeze'),
    ])
    const results = idx.search('reciprocal rank fusion', 5)
    expect(results[0].chunk.id).toBe('a')
  })

  it('omits documents matching none of the query terms entirely', () => {
    const idx = new BM25Index([chunk('a', 'apples and oranges'), chunk('b', 'quantum mechanics')])
    const results = idx.search('bananas', 5)
    expect(results).toEqual([])
  })

  it('a rarer matched term contributes more score than a common one (IDF weighting)', () => {
    // "the" appears in every document (no discriminating power); "reciprocal" appears in only one
    const idx = new BM25Index([
      chunk('a', 'the reciprocal fusion of ranks'),
      chunk('b', 'the common ordinary ranks of things'),
      chunk('c', 'the usual ordinary ranks of things'),
    ])
    const results = idx.search('the reciprocal', 5)
    expect(results[0].chunk.id).toBe('a')
  })

  it('rewards a higher term frequency, all else equal', () => {
    const idx = new BM25Index([
      chunk('a', 'strata strata strata is a retrieval engine'),
      chunk('b', 'strata is a retrieval engine'),
    ])
    const results = idx.search('strata', 5)
    expect(results[0].chunk.id).toBe('a')
  })

  it('an empty index returns no results without throwing', () => {
    const idx = new BM25Index([])
    expect(idx.search('anything', 5)).toEqual([])
  })

  it('respects k', () => {
    const chunks = Array.from({ length: 10 }, (_, i) => chunk(String(i), `document about topic ${i % 2 === 0 ? 'alpha' : 'beta'}`))
    const idx = new BM25Index(chunks)
    expect(idx.search('alpha', 3).length).toBe(3)
  })

  it('reports its size', () => {
    expect(new BM25Index([chunk('a', 'x'), chunk('b', 'y')]).size).toBe(2)
  })

  it('sorts strictly by descending score, even when that differs from insertion order', () => {
    // The highest-scoring document here is the LAST one constructed, so a broken or no-op sort would still
    // coincidentally pass an assertion based on the natural (insertion) order — the earlier tests in this file are
    // all vulnerable to exactly that. This one is not.
    const idx = new BM25Index([
      chunk('a', 'irrelevant content about nothing to do with the query'),
      chunk('b', 'somewhat relevant, mentions strata just once'),
      chunk('c', 'strata strata strata strata is extremely relevant here'),
    ])
    const results = idx.search('strata', 3)
    expect(results[0].chunk.id).toBe('c')
  })

  it('a document with no real tokens (e.g. pure punctuation) contributes zero length, not a phantom token, to the corpus average', () => {
    // If tokenize ever fell back to a placeholder value instead of an empty array on no match, this document would
    // silently count as length 1 instead of 0, skewing avgDocLength for every other document's score.
    const idx = new BM25Index([chunk('empty', '!!! --- ...'), chunk('real', 'cat cat dog')])
    const results = idx.search('cat', 2)
    function referenceBM25(termFreq: number, docLen: number, avgDocLen: number, df: number, N: number, k1 = 1.5, b = 0.75): number {
      const idf = Math.log((N - df + 0.5) / (df + 0.5) + 1)
      const denom = termFreq + k1 * (1 - b + b * (docLen / avgDocLen))
      return idf * ((termFreq * (k1 + 1)) / denom)
    }
    const avgDocLen = (0 + 3) / 2 // the punctuation-only doc must contribute length 0, not 1
    const expected = referenceBM25(2, 3, avgDocLen, 1, 2)
    expect(results[0].score).toBeCloseTo(expected, 10)
  })

  it('matches an independently-written reference implementation of the BM25 formula exactly, not just in relative ranking', () => {
    // Written separately from bm25.ts's own formula, so a wrong constant or flipped operator in EITHER place would
    // very likely disagree — pinning an exact numeric value catches arithmetic mistakes that a purely
    // relative-ordering assertion (does "a" rank above "b"?) can miss when a bug happens to preserve the ranking
    // anyway (a real failure mode: several mutation survivors in this file were exactly that).
    function referenceBM25(termFreq: number, docLen: number, avgDocLen: number, df: number, N: number, k1 = 1.5, b = 0.75): number {
      const idf = Math.log((N - df + 0.5) / (df + 0.5) + 1)
      const denom = termFreq + k1 * (1 - b + b * (docLen / avgDocLen))
      return idf * ((termFreq * (k1 + 1)) / denom)
    }

    const docs = [chunk('a', 'cat cat dog'), chunk('b', 'cat bird bird bird bird')]
    const idx = new BM25Index(docs)
    const avgDocLen = (3 + 5) / 2

    const single = idx.search('cat', 2)
    const expectedCatA = referenceBM25(2, 3, avgDocLen, 2, 2)
    const expectedCatB = referenceBM25(1, 5, avgDocLen, 2, 2)
    expect(single.find((r) => r.chunk.id === 'a')!.score).toBeCloseTo(expectedCatA, 10)
    expect(single.find((r) => r.chunk.id === 'b')!.score).toBeCloseTo(expectedCatB, 10)
    expect(single[0].chunk.id).toBe('a') // higher term freq + shorter doc: correctly ranks first

    // a two-term query forces score ACCUMULATION across terms for doc 'a' (matches both "cat" and "dog") — this is
    // what actually catches a broken `scores.get(chunkIndex) ?? 0` that silently discards an earlier term's score
    // instead of adding to it, which a single-term query can never exercise
    const multi = idx.search('cat dog', 2)
    const expectedDogA = referenceBM25(1, 3, avgDocLen, 1, 2)
    const expectedTotalA = expectedCatA + expectedDogA
    expect(multi.find((r) => r.chunk.id === 'a')!.score).toBeCloseTo(expectedTotalA, 10)
  })
})
