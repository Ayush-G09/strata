import { describe, expect, it } from 'vitest'
import type { Chunk } from '@strata/core'
import { naiveRetrieve } from './baselineRetriever'

function chunk(id: string, text: string): Chunk {
  return { id, docId: 'd', text, start: 0, end: text.length, index: 0 }
}

describe('naiveRetrieve', () => {
  it('ranks by number of distinct query terms present, ignoring how many times each appears', async () => {
    const chunks = [
      chunk('a', 'strata retrieval engine with hybrid search'), // matches 3 distinct query terms
      chunk('b', 'strata strata strata strata strata'), // matches only 1 distinct term, however many times
    ]
    const results = await naiveRetrieve('strata retrieval hybrid', chunks, 2)
    expect(results[0].chunk.id).toBe('a')
  })

  it('breaks ties by original chunk order, not alphabetically or by content', async () => {
    const chunks = [chunk('z', 'no overlap here'), chunk('a', 'strata'), chunk('m', 'strata')]
    const results = await naiveRetrieve('strata', chunks, 3)
    expect(results.map((r) => r.chunk.id)).toEqual(['a', 'm', 'z']) // 'a' and 'm' tie on score, original order preserved; 'z' last (0 matches)
  })

  it('respects k', async () => {
    const chunks = Array.from({ length: 10 }, (_, i) => chunk(String(i), 'strata'))
    const results = await naiveRetrieve('strata', chunks, 3)
    expect(results.length).toBe(3)
  })

  it('returns fusedScore equal to the raw distinct-term match count, with null ranks (it has no vector/BM25 concept)', async () => {
    const results = await naiveRetrieve('alpha beta gamma', [chunk('a', 'alpha beta')], 1)
    expect(results[0].fusedScore).toBe(2)
    expect(results[0].vectorRank).toBeNull()
    expect(results[0].bm25Rank).toBeNull()
  })

  it('a chunk matching zero query terms still appears in the results (unlike BM25, nothing is omitted)', async () => {
    const results = await naiveRetrieve('strata', [chunk('a', 'completely unrelated text')], 5)
    expect(results.length).toBe(1)
    expect(results[0].fusedScore).toBe(0)
  })

  it('an empty chunk list returns no results without throwing', async () => {
    expect(await naiveRetrieve('anything', [], 5)).toEqual([])
  })
})
