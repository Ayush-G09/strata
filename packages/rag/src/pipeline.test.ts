import { describe, expect, it } from 'vitest'
import { HashingEmbedder } from '@strata/core'
import { FakeLlmClient } from './llmClient'
import { StrataIndex } from './pipeline'

describe('StrataIndex (end to end, fully offline)', () => {
  it('answers a query with a citation that resolves to the actually-retrieved, relevant chunk', async () => {
    const llm = new FakeLlmClient((prompt) => {
      // a stand-in "model": cites whichever numbered source contains the word "capital", proving the plumbing
      // (chunk -> embed -> retrieve -> prompt -> parse) carries the real retrieved text through end to end
      const lines = prompt.split('\n').filter((l) => /^\[\d+\]/.test(l))
      const match = lines.find((l) => l.toLowerCase().includes('capital'))
      const n = match ? lines.indexOf(match) + 1 : 1
      return `The capital of France is Paris [${n}].`
    })
    const index = new StrataIndex(new HashingEmbedder(), llm)
    await index.addDocument(
      { id: 'geo', text: 'Paris is the capital of France.\n\nBerlin is the capital of Germany.\n\nBananas are yellow.' },
      { strategy: 'structural', size: 200, overlap: 0 },
    )

    const result = await index.query('What is the capital of France?', 3)
    expect(result.answer.citedChunkIds.length).toBe(1)
    const citedChunk = result.retrieved.find((r) => r.chunk.id === result.answer.citedChunkIds[0])!.chunk
    expect(citedChunk.text).toContain('capital of France')
    expect(result.answer.invalidCitationNumbers).toEqual([])
  })

  it('tracks chunk count as documents are added', async () => {
    const index = new StrataIndex(new HashingEmbedder(), new FakeLlmClient(() => 'answer [1].'))
    expect(index.chunkCount).toBe(0)
    const added = await index.addDocument({ id: 'd1', text: 'a'.repeat(500) }, { strategy: 'fixed', size: 100, overlap: 0 })
    expect(index.chunkCount).toBe(added.length)
    expect(index.chunkCount).toBeGreaterThan(1)
  })

  it('exposes the exact same chunks added via addDocument through the chunks getter', async () => {
    const index = new StrataIndex(new HashingEmbedder(), new FakeLlmClient(() => 'x [1].'))
    const added = await index.addDocument({ id: 'd1', text: 'a'.repeat(250) }, { strategy: 'fixed', size: 100, overlap: 0 })
    expect(index.chunks).toEqual(added)
  })

  it('an empty document adds no chunks (and returns an empty array, not a placeholder) and does not break subsequent queries', async () => {
    const index = new StrataIndex(new HashingEmbedder(), new FakeLlmClient(() => 'no info available.'))
    const added = await index.addDocument({ id: 'empty', text: '' }, { strategy: 'fixed', size: 10, overlap: 0 })
    expect(added).toEqual([])
    expect(index.chunkCount).toBe(0)
    const result = await index.query('anything?', 3)
    expect(result.retrieved).toEqual([])
  })

  it('actually populates the vector store when a document is added, not just the BM25 index', async () => {
    const index = new StrataIndex(new HashingEmbedder(), new FakeLlmClient(() => 'x [1].'))
    await index.addDocument({ id: 'd1', text: 'strata combines bm25 keyword search with vector similarity search.' }, { strategy: 'structural', size: 200, overlap: 0 })
    const result = await index.query('vector similarity search', 3)
    // if the vector store were never populated, every result's vectorRank would be null — this is only true if
    // the per-chunk embed-and-add loop actually ran, not just the BM25 index rebuild
    expect(result.retrieved.some((r) => r.vectorRank !== null)).toBe(true)
  })

  it('retrieval sees content from multiple added documents, not just the most recent one', async () => {
    const index = new StrataIndex(new HashingEmbedder(), new FakeLlmClient(() => 'x [1].'))
    await index.addDocument({ id: 'd1', text: 'strata uses BM25 for keyword search.' }, { strategy: 'structural', size: 200, overlap: 0 })
    await index.addDocument({ id: 'd2', text: 'strata uses reciprocal rank fusion to combine rankings.' }, { strategy: 'structural', size: 200, overlap: 0 })
    const result = await index.query('reciprocal rank fusion', 5)
    expect(result.retrieved.some((r) => r.chunk.docId === 'd2')).toBe(true)
  })

  it('surfaces an invalid citation from the model rather than silently dropping evidence of it', async () => {
    const llm = new FakeLlmClient(() => 'This cites a source that does not exist [99].')
    const index = new StrataIndex(new HashingEmbedder(), llm)
    await index.addDocument({ id: 'd1', text: 'some real content here about strata.' }, { strategy: 'structural', size: 200, overlap: 0 })
    const result = await index.query('anything', 3)
    expect(result.answer.invalidCitationNumbers).toEqual([99])
  })
})
