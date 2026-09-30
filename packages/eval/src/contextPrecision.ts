import type { Chunk } from '@strata/core'
import type { LlmClient } from '@strata/rag'
import { parseYesNo } from './parsing'

export interface ContextPrecisionResult {
  score: number
  relevance: boolean[]
}

/** Are the retrieved chunks actually useful, and are the useful ones ranked near the top? Each chunk is judged
 * relevant or not independently, then combined as mean average precision over the RANKED list — precision at
 * each position where a relevant chunk appears, averaged over just those positions. A relevant chunk buried at
 * rank 10 behind nine irrelevant ones scores far worse than the same chunk at rank 1, which a plain "how many of
 * the k are relevant" fraction would not capture at all. */
export async function contextPrecision(
  question: string, rankedChunks: Chunk[], llm: LlmClient,
): Promise<ContextPrecisionResult> {
  if (rankedChunks.length === 0) return { score: 0, relevance: [] }

  const relevance = await Promise.all(
    rankedChunks.map(async (chunk) => {
      const prompt = [
        `Question: ${question}`,
        '',
        `Chunk: "${chunk.text}"`,
        '',
        'Is this chunk relevant to answering the question? Answer with a single word: yes or no.',
      ].join('\n')
      return parseYesNo(await llm.generate(prompt))
    }),
  )

  let relevantSoFar = 0
  let precisionSum = 0
  relevance.forEach((isRelevant, i) => {
    if (isRelevant) {
      relevantSoFar++
      precisionSum += relevantSoFar / (i + 1)
    }
  })

  const score = relevantSoFar > 0 ? precisionSum / relevantSoFar : 0
  return { score, relevance }
}
