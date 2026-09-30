import type { Chunk } from '@strata/core'
import type { LlmClient } from '@strata/rag'
import { decomposeClaims } from './claims'
import { parseYesNo } from './parsing'

export interface FaithfulnessResult {
  score: number
  claims: { claim: string; supported: boolean }[]
}

/** Does the generated answer stick to what the retrieved context actually supports? The answer is decomposed into
 * its individual claims, and each is checked against the context independently — one unsupported claim in an
 * otherwise-good answer lowers the score proportionally, rather than a single overall "does this look right"
 * judgment that could miss one hallucinated sentence buried in an otherwise faithful paragraph. */
export async function faithfulness(answerText: string, context: Chunk[], llm: LlmClient): Promise<FaithfulnessResult> {
  const claims = await decomposeClaims(answerText, llm)
  // no verifiable factual claims (e.g. "I don't know") means there is nothing to be unfaithful about — scored as
  // fully faithful by convention, the same way an empty test suite is not scored as 0% passing
  if (claims.length === 0) return { score: 1, claims: [] }

  const contextText = context.map((c) => c.text).join('\n\n')
  const judged = await Promise.all(
    claims.map(async (claim) => {
      const prompt = [
        `Context:\n${contextText}`,
        '',
        `Claim: "${claim}"`,
        '',
        'Is this claim directly supported by the context above? Answer with a single word: yes or no.',
      ].join('\n')
      const raw = await llm.generate(prompt)
      return { claim, supported: parseYesNo(raw) }
    }),
  )

  const score = judged.filter((j) => j.supported).length / judged.length
  return { score, claims: judged }
}
