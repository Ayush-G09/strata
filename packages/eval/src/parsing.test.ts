import { describe, expect, it } from 'vitest'
import { parseLines, parseYesNo } from './parsing'

describe('parseLines', () => {
  it('splits on newlines and trims each line', () => {
    expect(parseLines('one\ntwo\nthree')).toEqual(['one', 'two', 'three'])
  })

  it('strips leading numbering, bullets, and dashes', () => {
    expect(parseLines('1. first\n2) second\n- third\n* fourth\n• fifth')).toEqual(['first', 'second', 'third', 'fourth', 'fifth'])
  })

  it('drops empty lines', () => {
    expect(parseLines('one\n\n\ntwo\n')).toEqual(['one', 'two'])
  })

  it('returns an empty array for empty or whitespace-only input', () => {
    expect(parseLines('')).toEqual([])
    expect(parseLines('   \n  \n')).toEqual([])
  })
})

describe('parseYesNo', () => {
  it('is true for a response starting with "yes", case-insensitively', () => {
    expect(parseYesNo('yes')).toBe(true)
    expect(parseYesNo('Yes.')).toBe(true)
    expect(parseYesNo('YES, definitely.')).toBe(true)
  })

  it('is false for a response starting with "no"', () => {
    expect(parseYesNo('no')).toBe(false)
    expect(parseYesNo('No, it does not.')).toBe(false)
  })

  it('requires "yes" at the START of the response, not merely present anywhere in it', () => {
    // this is what actually distinguishes a real judgment from a model saying e.g. "No, yes-like claims aren't
    // present here" — the anchor is what makes that correctly read as a "no"
    expect(parseYesNo("No, yes-like claims aren't present here.")).toBe(false)
  })

  it('tolerates leading whitespace before "yes"', () => {
    expect(parseYesNo('   yes')).toBe(true)
  })

  it('requires a word boundary after "yes" (does not match "yesterday")', () => {
    expect(parseYesNo('yesterday it was true')).toBe(false)
  })

  it('is false for an empty or unrelated response', () => {
    expect(parseYesNo('')).toBe(false)
    expect(parseYesNo('unclear')).toBe(false)
  })
})
