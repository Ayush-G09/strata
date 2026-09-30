import { describe, expect, it } from 'vitest'
import type { Chunk } from '@strata/core'
import { parseAnswer } from './answer'

function chunk(id: string): Chunk {
  return { id, docId: 'd', text: `text of ${id}`, start: 0, end: 1, index: 0 }
}

describe('parseAnswer', () => {
  it('resolves a citation marker to the chunk at that 1-indexed position', () => {
    const sources = [chunk('a'), chunk('b'), chunk('c')]
    const result = parseAnswer('The sky is blue [2].', sources)
    expect(result.citedChunkIds).toEqual(['b'])
    expect(result.invalidCitationNumbers).toEqual([])
  })

  it('de-duplicates repeated citations of the same source', () => {
    const sources = [chunk('a'), chunk('b')]
    const result = parseAnswer('First point [1]. Related point [1] again. Second point [2].', sources)
    expect(result.citedChunkIds).toEqual(['a', 'b'])
  })

  it('preserves citation order as first-cited, not numeric order', () => {
    const sources = [chunk('a'), chunk('b'), chunk('c')]
    const result = parseAnswer('Point one [3]. Point two [1].', sources)
    expect(result.citedChunkIds).toEqual(['c', 'a'])
  })

  it('flags a citation number beyond the given sources as invalid, without crashing', () => {
    const sources = [chunk('a')]
    const result = parseAnswer('This cites a source that was never given [7].', sources)
    expect(result.citedChunkIds).toEqual([])
    expect(result.invalidCitationNumbers).toEqual([7])
  })

  it('flags citation number 0 as invalid (sources are 1-indexed)', () => {
    const result = parseAnswer('Says [0].', [chunk('a')])
    expect(result.invalidCitationNumbers).toEqual([0])
  })

  it('an answer with no citations at all is valid but empty', () => {
    const result = parseAnswer('No sources needed for this.', [chunk('a')])
    expect(result.citedChunkIds).toEqual([])
    expect(result.invalidCitationNumbers).toEqual([])
  })

  it('does not duplicate the same invalid number reported twice', () => {
    const result = parseAnswer('[9] and again [9].', [chunk('a')])
    expect(result.invalidCitationNumbers).toEqual([9])
  })
})
