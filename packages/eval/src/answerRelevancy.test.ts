import { describe, expect, it } from 'vitest'
import { HashingEmbedder } from '@strata/core'
import { FakeLlmClient } from '@strata/rag'
import { answerRelevancy } from './answerRelevancy'

describe('answerRelevancy', () => {
  it('scores highly when the reverse-generated questions closely resemble the real question', async () => {
    const llm = new FakeLlmClient(() => 'What is the capital of France?\nWhich city is the capital of France?\nName the capital of France.')
    const result = await answerRelevancy('What is the capital of France?', 'Paris is the capital of France.', new HashingEmbedder(), llm, 3)
    expect(result.generatedQuestions.length).toBe(3)
    expect(result.score).toBeGreaterThan(0.5)
  })

  it('scores lower when the answer is vague enough to generate unrelated questions', async () => {
    const onTopic = new FakeLlmClient(() => 'What is the capital of France?\nWhich city is the capital of France?')
    const offTopic = new FakeLlmClient(() => 'What is the weather like today?\nHow do birds migrate?')
    const embedder = new HashingEmbedder()
    const good = await answerRelevancy('What is the capital of France?', 'Paris.', embedder, onTopic, 2)
    const bad = await answerRelevancy('What is the capital of France?', 'Something vague and unrelated.', embedder, offTopic, 2)
    expect(good.score).toBeGreaterThan(bad.score)
  })

  it('scores 0 and returns no questions if the model generates nothing usable', async () => {
    const llm = new FakeLlmClient(() => '')
    const result = await answerRelevancy('q', 'a', new HashingEmbedder(), llm)
    expect(result.score).toBe(0)
    expect(result.generatedQuestions).toEqual([])
  })

  it('sends a generation prompt containing the exact answer text and requested count', async () => {
    let captured = ''
    const llm = new FakeLlmClient((prompt) => { captured = prompt; return 'Q1?\nQ2?' })
    await answerRelevancy('q', 'The exact answer text.', new HashingEmbedder(), llm, 2)
    expect(captured).toContain('write 2 different questions')
    expect(captured).toContain('Answer:\nThe exact answer text.')
  })

  it('respects n, the requested number of generated questions', async () => {
    const llm = new FakeLlmClient(() => 'Q1?\nQ2?\nQ3?\nQ4?\nQ5?')
    const result = await answerRelevancy('q', 'a', new HashingEmbedder(), llm, 2)
    expect(result.generatedQuestions.length).toBe(2)
  })
})
