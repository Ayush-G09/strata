import { describe, expect, it } from 'vitest'
import type { Chunk } from '@strata/core'
import { buildPrompt } from './prompt'

function chunk(id: string, text: string): Chunk {
  return { id, docId: 'd', text, start: 0, end: text.length, index: 0 }
}

describe('buildPrompt', () => {
  it('produces the exact expected prompt structure for a known question and sources', () => {
    const prompt = buildPrompt('what is strata?', [chunk('a', 'first source text'), chunk('b', 'second source text')])
    const expected = [
      'Answer the question using ONLY the numbered sources below. Cite the source number inline like [1] for every',
      'claim you make. If the sources do not contain enough information to answer, say so plainly instead of guessing.',
      '',
      'Sources:',
      '[1] first source text\n\n[2] second source text',
      '',
      'Question: what is strata?',
      '',
      'Answer:',
    ].join('\n')
    expect(prompt).toBe(expected)
  })

  it('numbers each source starting at 1, in the order given', () => {
    const prompt = buildPrompt('what is strata?', [chunk('a', 'first source text'), chunk('b', 'second source text')])
    expect(prompt).toContain('[1] first source text')
    expect(prompt).toContain('[2] second source text')
  })

  it('includes the literal question text', () => {
    const prompt = buildPrompt('how does BM25 work?', [chunk('a', 'x')])
    expect(prompt).toContain('how does BM25 work?')
  })

  it('instructs the model to answer only from the given sources, and to admit when it can’t', () => {
    const prompt = buildPrompt('q', [chunk('a', 'x')])
    expect(prompt.toLowerCase()).toContain('only')
    expect(prompt.toLowerCase()).toContain('say so plainly')
  })

  it('handles zero retrieved sources without crashing, and the Sources section is simply empty', () => {
    const prompt = buildPrompt('q', [])
    expect(() => prompt).not.toThrow()
    expect(prompt).toContain('Sources:\n\n\nQuestion: q')
  })
})
