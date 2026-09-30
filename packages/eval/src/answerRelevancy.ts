import { cosineSimilarity } from '@strata/core'
import type { Embedder } from '@strata/core'
import type { LlmClient } from '@strata/rag'
import { parseLines } from './parsing'

export interface AnswerRelevancyResult {
  score: number
  generatedQuestions: string[]
}

/** Does the answer actually address the question asked? Measured indirectly, the way Ragas does it: ask the model
 * to reverse-engineer questions this answer would be a good response to, then compare those generated questions
 * to the real one by embedding similarity. An answer that drifts off-topic tends to generate questions that don't
 * resemble the original — a vague or evasive answer generates vague, dissimilar questions too. */
export async function answerRelevancy(
  question: string, answerText: string, embedder: Embedder, llm: LlmClient, n = 3,
): Promise<AnswerRelevancyResult> {
  const prompt = [
    `Given the following answer, write ${n} different questions that this answer would be a good, direct response`,
    'to. One question per line, no numbering, no commentary.',
    '',
    'Answer:',
    answerText,
  ].join('\n')
  const raw = await llm.generate(prompt)
  const generatedQuestions = parseLines(raw).slice(0, n)
  if (generatedQuestions.length === 0) return { score: 0, generatedQuestions: [] }

  const [questionVector, ...generatedVectors] = await embedder.embed([question, ...generatedQuestions])
  const similarities = generatedVectors.map((v) => cosineSimilarity(questionVector, v))
  const score = similarities.reduce((sum, s) => sum + s, 0) / similarities.length
  return { score, generatedQuestions }
}
