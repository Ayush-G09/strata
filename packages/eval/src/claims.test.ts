import { describe, expect, it } from 'vitest'
import { FakeLlmClient } from '@strata/rag'
import { decomposeClaims } from './claims'

describe('decomposeClaims', () => {
  it('sends a prompt containing the exact text to decompose', async () => {
    let captured = ''
    const llm = new FakeLlmClient((prompt) => { captured = prompt; return 'A claim.' })
    await decomposeClaims('The text to decompose.', llm)
    expect(captured).toContain('Text:\nThe text to decompose.')
    expect(captured.toLowerCase()).toContain('one claim per line')
    expect(captured.toLowerCase()).toContain('no numbering')
  })

  it('returns the model’s response split into individual claims', async () => {
    const llm = new FakeLlmClient(() => 'First claim.\nSecond claim.')
    expect(await decomposeClaims('irrelevant', llm)).toEqual(['First claim.', 'Second claim.'])
  })

  it('returns an empty array when the model finds nothing to decompose', async () => {
    const llm = new FakeLlmClient(() => '')
    expect(await decomposeClaims('irrelevant', llm)).toEqual([])
  })
})
