import { describe, expect, it } from 'vitest'
import { FakeLlmClient } from '@strata/rag'
import type { Chunk } from '@strata/core'
import { contextPrecision } from './contextPrecision'

function chunk(id: string, text: string): Chunk {
  return { id, docId: 'd', text, start: 0, end: text.length, index: 0 }
}

describe('contextPrecision', () => {
  it('scores 1.0 when every retrieved chunk, in order, is judged relevant', async () => {
    const llm = new FakeLlmClient(() => 'yes')
    const chunks = [chunk('a', 'relevant one'), chunk('b', 'relevant two')]
    const result = await contextPrecision('q', chunks, llm)
    expect(result.score).toBe(1)
  })

  it('penalises a relevant chunk ranked behind an irrelevant one more than the reverse order', async () => {
    const relevantFirst = [chunk('a', 'RELEVANT'), chunk('b', 'irrelevant')]
    const relevantSecond = [chunk('a', 'irrelevant'), chunk('b', 'RELEVANT')]
    const llm = new FakeLlmClient((prompt) => (prompt.includes('"RELEVANT"') ? 'yes' : 'no'))
    const first = await contextPrecision('q', relevantFirst, llm)
    const second = await contextPrecision('q', relevantSecond, llm)
    expect(first.score).toBeGreaterThan(second.score)
  })

  it('scores 0 when no retrieved chunk is judged relevant', async () => {
    const llm = new FakeLlmClient(() => 'no')
    const result = await contextPrecision('q', [chunk('a', 'x'), chunk('b', 'y')], llm)
    expect(result.score).toBe(0)
  })

  it('scores 0 for an empty retrieval, without dividing by zero', async () => {
    const result = await contextPrecision('q', [], new FakeLlmClient(() => 'yes'))
    expect(result.score).toBe(0)
    expect(result.relevance).toEqual([])
  })

  it('sends a judgment prompt containing the exact question and the exact chunk text', async () => {
    let captured = ''
    const llm = new FakeLlmClient((prompt) => { captured = prompt; return 'yes' })
    await contextPrecision('what is the capital?', [chunk('a', 'Paris is the capital of France.')], llm)
    expect(captured).toContain('Question: what is the capital?')
    expect(captured).toContain('Chunk: "Paris is the capital of France."')
    expect(captured.toLowerCase()).toContain('answer with a single word: yes or no')
  })

  it('records the per-chunk relevance judgments in retrieval order', async () => {
    const llm = new FakeLlmClient((prompt) => (prompt.includes('"first"') ? 'yes' : 'no'))
    const result = await contextPrecision('q', [chunk('a', 'first'), chunk('b', 'second')], llm)
    expect(result.relevance).toEqual([true, false])
  })
})
