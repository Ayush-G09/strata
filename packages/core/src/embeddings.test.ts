import { describe, expect, it } from 'vitest'
import { cosineSimilarity, HashingEmbedder } from './embeddings'

describe('HashingEmbedder', () => {
  it('is deterministic: the same text always embeds to the exact same vector', async () => {
    const e = new HashingEmbedder()
    const [a] = await e.embed(['the quick brown fox'])
    const [b] = await e.embed(['the quick brown fox'])
    expect(a).toEqual(b)
  })

  it('produces a vector of the configured dimensionality', async () => {
    const e = new HashingEmbedder(64)
    const [v] = await e.embed(['hello world'])
    expect(v.length).toBe(64)
  })

  it('produces a unit-length vector for any non-empty text', async () => {
    const e = new HashingEmbedder()
    const [v] = await e.embed(['strata is a retrieval engine with a built-in evaluation harness'])
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0))
    expect(norm).toBeCloseTo(1, 5)
  })

  it('embeds empty text to the zero vector without throwing', async () => {
    const e = new HashingEmbedder()
    const [v] = await e.embed([''])
    expect(v.every((x) => x === 0)).toBe(true)
  })

  it('two documents sharing more vocabulary score a higher cosine similarity than two sharing none', async () => {
    const e = new HashingEmbedder()
    const [a, b, c] = await e.embed([
      'the cat sat on the mat',
      'the cat sat on the rug',
      'quantum entanglement violates local realism',
    ])
    expect(cosineSimilarity(a, b)).toBeGreaterThan(cosineSimilarity(a, c))
  })

  it('a document is maximally similar to itself', async () => {
    const e = new HashingEmbedder()
    const [a] = await e.embed(['strata retrieval evaluation harness'])
    expect(cosineSimilarity(a, a)).toBeCloseTo(1, 8)
  })

  it('two tokens hashing into the same bucket with opposite sign partially cancel, not reinforce', async () => {
    // With dims=1 every token shares the single bucket, so the combined text's sign is fully determined by
    // whether the two tokens' individual signs agree (reinforce to ±1) or disagree (cancel to exactly 0) — this
    // pins down the actual sign/accumulation logic instead of only checking downstream similarity ordering, which
    // a broken sign computation can still coincidentally preserve.
    const e = new HashingEmbedder(1)
    const words = Array.from({ length: 60 }, (_, i) => `word${i}`)
    let sawCancellation = false
    let sawReinforcement = false
    for (const a of words) {
      for (const b of words) {
        if (a === b) continue
        const [va] = await e.embed([a])
        const [vb] = await e.embed([b])
        const [vab] = await e.embed([`${a} ${b}`])
        const agree = Math.sign(va[0]) === Math.sign(vb[0])
        if (agree) {
          expect(Math.sign(vab[0])).toBe(Math.sign(va[0]))
          sawReinforcement = true
        } else {
          expect(vab[0]).toBe(0)
          sawCancellation = true
        }
      }
    }
    // sanity: this word list must actually exercise both cases, or the test would vacuously pass no matter the logic
    expect(sawCancellation).toBe(true)
    expect(sawReinforcement).toBe(true)
  })
})

describe('cosineSimilarity', () => {
  it('rejects vectors of mismatched dimension, with a message naming the problem', () => {
    expect(() => cosineSimilarity([1, 0], [1, 0, 0])).toThrow(/same dimension/)
  })

  it('computes the exact cosine value for vectors of different magnitude but the same direction', () => {
    // dot=2, |a|=1, |b|=2 -> 2/(1*2) = 1 exactly. Pins the division (not multiplication, not a swapped ratio).
    expect(cosineSimilarity([1, 0], [2, 0])).toBe(1)
  })

  it('computes a known non-trivial exact value', () => {
    // a=(3,4) |a|=5, b=(4,3) |b|=5, dot=3*4+4*3=24 -> 24/25 = 0.96 exactly
    expect(cosineSimilarity([3, 4], [4, 3])).toBeCloseTo(0.96, 12)
  })

  it('is 0 for orthogonal vectors', () => {
    expect(cosineSimilarity([1, 0], [0, 1])).toBe(0)
  })

  it('is -1 for opposite vectors', () => {
    expect(cosineSimilarity([1, 0], [-1, 0])).toBe(-1)
  })

  it('is 0, not NaN, when either vector is all zeros', () => {
    expect(cosineSimilarity([0, 0], [1, 1])).toBe(0)
    expect(cosineSimilarity([1, 1], [0, 0])).toBe(0) // the zero vector on the OTHER side too
    expect(cosineSimilarity([0, 0], [0, 0])).toBe(0)
  })
})
