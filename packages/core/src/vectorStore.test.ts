import { describe, expect, it } from 'vitest'
import { VectorStore } from './vectorStore'
import { HashingEmbedder } from './embeddings'
import type { Chunk } from './types'

function chunk(id: string, text: string): Chunk {
  return { id, docId: 'd', text, start: 0, end: text.length, index: 0 }
}

describe('VectorStore', () => {
  it('returns results ordered by descending similarity', async () => {
    const e = new HashingEmbedder()
    const store = new VectorStore()
    const docs = [
      chunk('a', 'the cat sat on the mat'),
      chunk('b', 'quantum entanglement and local realism'),
      chunk('c', 'the cat sat on the rug'),
    ]
    const vectors = await e.embed(docs.map((d) => d.text))
    docs.forEach((d, i) => store.add(d, vectors[i]))

    const [q] = await e.embed(['the cat sat on the mat'])
    const results = store.search(q, 3)
    expect(results[0].chunk.id).toBe('a') // exact match
    expect(results[1].chunk.id).toBe('c') // shares vocabulary
    expect(results[2].chunk.id).toBe('b') // unrelated
    for (let i = 1; i < results.length; i++) expect(results[i].score).toBeLessThanOrEqual(results[i - 1].score)
  })

  it('respects k, returning no more than requested even with a larger corpus', async () => {
    const e = new HashingEmbedder()
    const store = new VectorStore()
    const texts = Array.from({ length: 10 }, (_, i) => `document number ${i} about topic ${i % 3}`)
    const vectors = await e.embed(texts)
    texts.forEach((t, i) => store.add(chunk(String(i), t), vectors[i]))
    const [q] = await e.embed(['topic 1'])
    expect(store.search(q, 4).length).toBe(4)
  })

  it('an empty store returns no results', async () => {
    const store = new VectorStore()
    expect(store.search([1, 0], 5)).toEqual([])
  })

  it('reports its size', async () => {
    const store = new VectorStore()
    expect(store.size).toBe(0)
    store.add(chunk('a', 'x'), [1])
    expect(store.size).toBe(1)
  })
})
