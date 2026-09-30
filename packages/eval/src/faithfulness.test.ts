import { describe, expect, it } from 'vitest'
import { FakeLlmClient } from '@strata/rag'
import type { Chunk } from '@strata/core'
import { faithfulness } from './faithfulness'

function chunk(id: string, text: string): Chunk {
  return { id, docId: 'd', text, start: 0, end: text.length, index: 0 }
}

describe('faithfulness', () => {
  it('scores 1.0 when every decomposed claim is judged supported by the context', async () => {
    const llm = new FakeLlmClient((prompt) => {
      if (prompt.startsWith('List each distinct factual claim')) return 'Paris is the capital of France.\nFrance is in Europe.'
      return 'yes' // every judgment call says supported
    })
    const context = [chunk('a', 'Paris is the capital of France, located in Europe.')]
    const result = await faithfulness('Paris is the capital of France, in Europe.', context, llm)
    expect(result.score).toBe(1)
    expect(result.claims.length).toBe(2)
  })

  it('scores partially when only some claims are supported', async () => {
    const llm = new FakeLlmClient((prompt) => {
      if (prompt.startsWith('List each distinct factual claim')) return 'Claim one.\nClaim two.\nClaim three.\nClaim four.'
      // support "one" and "three" only
      if (prompt.includes('"Claim one."') || prompt.includes('"Claim three."')) return 'yes'
      return 'no'
    })
    const result = await faithfulness('irrelevant, claims come from the fake decomposer', [chunk('a', 'x')], llm)
    expect(result.score).toBe(0.5)
    expect(result.claims.filter((c) => c.supported).map((c) => c.claim)).toEqual(['Claim one.', 'Claim three.'])
  })

  it('scores 0 when no claim is supported', async () => {
    const llm = new FakeLlmClient((prompt) => {
      if (prompt.startsWith('List each distinct factual claim')) return 'A fabricated claim.'
      return 'no'
    })
    const result = await faithfulness('answer', [chunk('a', 'unrelated context')], llm)
    expect(result.score).toBe(0)
  })

  it('an answer with no verifiable claims (e.g. "I don’t know") is scored fully faithful, not 0', async () => {
    const llm = new FakeLlmClient((prompt) => {
      if (prompt.startsWith('List each distinct factual claim')) return ''
      return 'no'
    })
    const result = await faithfulness("I don't know based on the given sources.", [chunk('a', 'x')], llm)
    expect(result.score).toBe(1)
    expect(result.claims).toEqual([])
  })

  it('sends a judgment prompt containing the exact context (chunks joined by a blank line) and the exact claim', async () => {
    const captured: string[] = []
    const llm = new FakeLlmClient((prompt) => {
      if (prompt.startsWith('List each distinct factual claim')) return 'The only claim.'
      captured.push(prompt)
      return 'yes'
    })
    await faithfulness('answer text', [chunk('a', 'first chunk'), chunk('b', 'second chunk')], llm)
    expect(captured[0]).toContain('Context:\nfirst chunk\n\nsecond chunk')
    expect(captured[0]).toContain('Claim: "The only claim."')
    expect(captured[0].toLowerCase()).toContain('answer with a single word: yes or no')
  })

  it('tolerates numbered/bulleted claim lists even though the prompt asks for plain lines', async () => {
    const llm = new FakeLlmClient((prompt) => {
      if (prompt.startsWith('List each distinct factual claim')) return '1. First claim.\n2. Second claim.\n- Third claim.'
      return 'yes'
    })
    const result = await faithfulness('answer', [chunk('a', 'x')], llm)
    expect(result.claims.map((c) => c.claim)).toEqual(['First claim.', 'Second claim.', 'Third claim.'])
  })
})
