import { describe, expect, it } from 'vitest'
import { HashingEmbedder } from '@strata/core'
import { FakeLlmClient } from '@strata/rag'
import { runGoldenSet } from './harness'
import { GOLDEN_SET } from './goldenSet'

const STOPWORDS = new Set(['the', 'a', 'an', 'is', 'are', 'to', 'of', 'and', 'or', 'what', 'which', 'how', 'many', 'per', 'this', 'that', 'with', 'do', 'i', 'does', 'can', 'be'])

function keywords(text: string): Set<string> {
  return new Set((text.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter((w) => w.length > 2 && !STOPWORDS.has(w)))
}

function overlapCount(a: string, b: string): number {
  const kb = keywords(b)
  let n = 0
  for (const w of keywords(a)) if (kb.has(w)) n++
  return n
}

/** A fully offline "simulated judge": not a claim about a real model's quality (that's what actually running the
 * harness against Gemini/Ollama is for), but a deterministic, keyword-overlap-based stand-in that behaves
 * consistently enough across every prompt shape this pipeline produces to prove the harness ITSELF is wired
 * correctly end to end, and that hybrid retrieval measurably beats the naive baseline on identical inputs — the
 * one thing that changed between the two runs below is which retriever found the context. */
function simulatedJudge(prompt: string): string {
  if (prompt.includes('Sources:') && prompt.includes('Answer the question using ONLY')) {
    const question = /Question: (.+)/.exec(prompt)?.[1] ?? ''
    const sources = [...prompt.matchAll(/\[(\d+)\] ([^\n]+)/g)]
    if (sources.length === 0) return "I don't know based on the given sources."
    let best = sources[0]
    let bestScore = -1
    for (const s of sources) {
      const score = overlapCount(question, s[2])
      if (score > bestScore) { bestScore = score; best = s }
    }
    return `${best[2]} [${best[1]}]`
  }
  if (prompt.startsWith('List each distinct factual claim')) {
    const text = prompt.split('\n\nText:\n')[1] ?? prompt
    return text.split(/(?<=[.!?])\s+/).map((s) => s.trim()).filter(Boolean).join('\n')
  }
  if (prompt.includes('write') && prompt.includes('different questions')) {
    const answer = prompt.split('\n\nAnswer:\n')[1] ?? ''
    return [1, 2, 3].map((i) => `What is the answer to ${answer.slice(0, 40)}? (${i})`).join('\n')
  }
  if (prompt.includes('Is this claim directly supported') || prompt.includes('Can this claim be attributed')) {
    const context = /Context:\n([\s\S]*?)\n\nClaim/.exec(prompt)?.[1] ?? ''
    const claim = /Claim: "(.+)"/.exec(prompt)?.[1] ?? ''
    return overlapCount(claim, context) >= 2 ? 'yes' : 'no'
  }
  if (prompt.includes('Is this chunk relevant')) {
    const question = /Question: (.+)/.exec(prompt)?.[1] ?? ''
    const chunkText = /Chunk: "(.+)"/.exec(prompt)?.[1] ?? ''
    return overlapCount(question, chunkText) >= 1 ? 'yes' : 'no'
  }
  return 'no'
}

describe('runGoldenSet (harness plumbing, offline simulated judge)', () => {
  it('produces a score for every case in the golden set, in range [0, 1]', async () => {
    const result = await runGoldenSet(new FakeLlmClient(simulatedJudge), new HashingEmbedder(), { retrieval: 'hybrid' })
    expect(result.perCase.length).toBe(GOLDEN_SET.length)
    for (const c of result.perCase) {
      for (const key of ['faithfulness', 'answerRelevancy', 'contextPrecision', 'contextRecall', 'overall'] as const) {
        expect(c[key]).toBeGreaterThanOrEqual(0)
        expect(c[key]).toBeLessThanOrEqual(1)
      }
    }
  })

  it('hybrid retrieval is never worse than the naive keyword-overlap baseline, same judge, same cases', async () => {
    // Not asserting a STRICT win here: this golden set's documents are short, single-topic, and non-adversarial,
    // and the offline HashingEmbedder is bag-of-words, not truly semantic — so BM25's real advantages (IDF
    // weighting, term-frequency saturation) and vector search's real advantage (catching a genuine paraphrase)
    // aren't reliably exercised by this particular combination. Those advantages ARE proven directly, at the
    // retrieval-primitive level, in bm25.test.ts (rarer-term weighting, term-frequency reward) and hybrid.test.ts
    // (exact keyword match found with no vector signal, semantic-ish similarity found with no keyword signal,
    // agreement between both signals outranking either alone). This test's job is narrower: prove hybrid is never
    // WORSE end to end, which is the honest claim this setup can actually support.
    const embedder = new HashingEmbedder()
    const hybrid = await runGoldenSet(new FakeLlmClient(simulatedJudge), embedder, { retrieval: 'hybrid' })
    const naive = await runGoldenSet(new FakeLlmClient(simulatedJudge), embedder, { retrieval: 'naive' })
    expect(hybrid.average.contextPrecision).toBeGreaterThanOrEqual(naive.average.contextPrecision)
    expect(hybrid.average.overall).toBeGreaterThanOrEqual(naive.average.overall)
  })

  it('a subset of cases can be run instead of the full golden set', async () => {
    const result = await runGoldenSet(new FakeLlmClient(simulatedJudge), new HashingEmbedder(), { cases: [GOLDEN_SET[0]] })
    expect(result.perCase.length).toBe(1)
    expect(result.perCase[0].id).toBe(GOLDEN_SET[0].id)
  })

  it('records which retrieval mode was used', async () => {
    const result = await runGoldenSet(new FakeLlmClient(simulatedJudge), new HashingEmbedder(), { retrieval: 'naive', cases: [GOLDEN_SET[0]] })
    expect(result.retrieval).toBe('naive')
  })

  it('defaults to hybrid retrieval when no mode is specified', async () => {
    const result = await runGoldenSet(new FakeLlmClient(simulatedJudge), new HashingEmbedder(), { cases: [GOLDEN_SET[0]] })
    expect(result.retrieval).toBe('hybrid')
  })

  it('overall is the exact arithmetic mean of all four metrics, not a different weighting', async () => {
    const result = await runGoldenSet(new FakeLlmClient(simulatedJudge), new HashingEmbedder(), { cases: [GOLDEN_SET[0]] })
    const c = result.perCase[0]
    expect(c.overall).toBeCloseTo((c.faithfulness + c.answerRelevancy + c.contextPrecision + c.contextRecall) / 4, 10)
  })

  it('hybrid and naive retrieval actually surface a DIFFERENT top chunk (not just reordered), changing context recall', async () => {
    // naiveRetrieve counts only DISTINCT query terms, so a single-word query can't distinguish a document
    // mentioning it once from one mentioning it five times — it falls back to original insertion order, keeping
    // "low" first. BM25 (inside hybrid) rewards the higher term frequency and correctly ranks "high" first
    // (proven directly, with insertion order controlled for, in bm25.test.ts). Requesting only the single top
    // chunk (k=1) means the two modes don't just reorder the same set, they retrieve genuinely DIFFERENT context —
    // and only "high" actually contains the fact the ground truth needs, so context recall must differ too.
    const adversarialCase = {
      id: 'adversarial',
      corpus: [
        { id: 'low', text: 'Strata is one part of a larger system.' },
        { id: 'high', text: 'Strata strata strata strata strata handles hybrid retrieval by combining BM25 and vector search.' },
      ],
      chunking: { strategy: 'structural' as const, size: 300, overlap: 0 },
      question: 'strata',
      groundTruthAnswer: 'Strata handles hybrid retrieval by combining BM25 and vector search.',
    }
    const embedder = new HashingEmbedder()
    const hybrid = await runGoldenSet(new FakeLlmClient(simulatedJudge), embedder, { retrieval: 'hybrid', k: 1, cases: [adversarialCase] })
    const naive = await runGoldenSet(new FakeLlmClient(simulatedJudge), embedder, { retrieval: 'naive', k: 1, cases: [adversarialCase] })
    expect(hybrid.perCase[0].contextRecall).toBeGreaterThan(naive.perCase[0].contextRecall)
  })
})
