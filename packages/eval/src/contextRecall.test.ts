import { describe, expect, it } from 'vitest'
import { FakeLlmClient } from '@strata/rag'
import type { Chunk } from '@strata/core'
import { contextRecall } from './contextRecall'

function chunk(id: string, text: string): Chunk {
  return { id, docId: 'd', text, start: 0, end: text.length, index: 0 }
}

describe('contextRecall', () => {
  it('scores 1.0 when every ground-truth claim is attributable to the retrieved context', async () => {
    const llm = new FakeLlmClient((prompt) => (prompt.startsWith('List each distinct factual claim') ? 'Claim A.\nClaim B.' : 'yes'))
    const result = await contextRecall('ground truth answer', [chunk('a', 'context covering everything')], llm)
    expect(result.score).toBe(1)
  })

  it('scores partially when the retrieved context is missing something the ground truth needed', async () => {
    const llm = new FakeLlmClient((prompt) => {
      if (prompt.startsWith('List each distinct factual claim')) return 'Claim A.\nClaim B.'
      return prompt.includes('"Claim A."') ? 'yes' : 'no' // context only covers claim A
    })
    const result = await contextRecall('ground truth', [chunk('a', 'partial context')], llm)
    expect(result.score).toBe(0.5)
  })

  it('an empty ground-truth answer (no claims) scores 1.0 by convention, not 0', async () => {
    const llm = new FakeLlmClient((prompt) => (prompt.startsWith('List each distinct factual claim') ? '' : 'no'))
    const result = await contextRecall('', [chunk('a', 'x')], llm)
    expect(result.score).toBe(1)
  })

  it('sends a judgment prompt containing the exact context (chunks joined by a blank line) and the exact claim', async () => {
    const captured: string[] = []
    const llm = new FakeLlmClient((prompt) => {
      if (prompt.startsWith('List each distinct factual claim')) return 'The only claim.'
      captured.push(prompt)
      return 'yes'
    })
    await contextRecall('ground truth', [chunk('a', 'first chunk'), chunk('b', 'second chunk')], llm)
    expect(captured[0]).toContain('Context:\nfirst chunk\n\nsecond chunk')
    expect(captured[0]).toContain('Claim: "The only claim."')
    expect(captured[0].toLowerCase()).toContain('attributed to')
  })

  it('scores 0 when the retrieved context supports none of the ground-truth claims', async () => {
    const llm = new FakeLlmClient((prompt) => (prompt.startsWith('List each distinct factual claim') ? 'Claim A.' : 'no'))
    const result = await contextRecall('ground truth', [chunk('a', 'completely unrelated context')], llm)
    expect(result.score).toBe(0)
  })
})
