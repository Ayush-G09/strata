import type { Chunk } from '@strata/core'
import type { LlmClient } from '@strata/rag'
import { decomposeClaims } from './claims'
import { parseYesNo } from './parsing'

export interface ContextRecallResult {
  score: number
  claims: { claim: string; attributable: boolean }[]
}

/** Does the retrieved context contain everything needed to produce the correct answer — even if the model's own
 * generated answer happened to miss something? This is checked against a known-correct ground-truth answer, not
 * the model's own output (that's what `faithfulness` is for): the ground truth is decomposed into claims, and each
 * is checked against the retrieved context. A missing chunk that held necessary information shows up here as a
 * claim with no supporting context, regardless of how the generated answer turned out. */
export async function contextRecall(
  groundTruthAnswer: string, retrievedChunks: Chunk[], llm: LlmClient,
): Promise<ContextRecallResult> {
  const claims = await decomposeClaims(groundTruthAnswer, llm)
  if (claims.length === 0) return { score: 1, claims: [] }

  const contextText = retrievedChunks.map((c) => c.text).join('\n\n')
  const judged = await Promise.all(
    claims.map(async (claim) => {
      const prompt = [
        `Context:\n${contextText}`,
        '',
        `Claim: "${claim}"`,
        '',
        'Can this claim be attributed to (found in) the context above? Answer with a single word: yes or no.',
      ].join('\n')
      return { claim, attributable: parseYesNo(await llm.generate(prompt)) }
    }),
  )

  const score = judged.filter((j) => j.attributable).length / judged.length
  return { score, claims: judged }
}
