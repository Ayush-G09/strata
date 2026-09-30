import { StrataIndex } from '@strata/rag'
import type { LlmClient } from '@strata/rag'
import type { Embedder, RankedResult } from '@strata/core'
import { naiveRetrieve } from './baselineRetriever'
import { faithfulness } from './faithfulness'
import { answerRelevancy } from './answerRelevancy'
import { contextPrecision } from './contextPrecision'
import { contextRecall } from './contextRecall'
import { GOLDEN_SET } from './goldenSet'
import type { GoldenCase } from './goldenSet'

export interface CaseResult {
  id: string
  faithfulness: number
  answerRelevancy: number
  contextPrecision: number
  contextRecall: number
  overall: number
}

export interface HarnessResult {
  retrieval: 'hybrid' | 'naive'
  perCase: CaseResult[]
  average: Omit<CaseResult, 'id'>
}

const METRIC_KEYS = ['faithfulness', 'answerRelevancy', 'contextPrecision', 'contextRecall', 'overall'] as const

/** Runs every case in the golden set (or a supplied subset) end to end — build a fresh index, retrieve, generate,
 * then score all four metrics — and averages the result. `retrieval: 'naive'` swaps in the keyword-overlap
 * baseline instead of Strata's real hybrid retriever, everything else identical, which is what makes the
 * comparison mean anything: the only variable that changed is the one being measured. */
export async function runGoldenSet(
  llm: LlmClient, embedder: Embedder,
  options: { retrieval?: 'hybrid' | 'naive'; k?: number; cases?: GoldenCase[] } = {},
): Promise<HarnessResult> {
  const { retrieval = 'hybrid', k = 4, cases = GOLDEN_SET } = options
  const perCase: CaseResult[] = []

  for (const c of cases) {
    const index = new StrataIndex(embedder, llm)
    for (const doc of c.corpus) await index.addDocument(doc, c.chunking)

    const retrieveOverride = retrieval === 'naive'
      ? (q: string, kk: number): Promise<RankedResult[]> => naiveRetrieve(q, index.chunks, kk)
      : undefined

    const { answer, retrieved } = retrieveOverride
      ? await index.query(c.question, k, retrieveOverride)
      : await index.query(c.question, k)

    const chunks = retrieved.map((r) => r.chunk)
    const [faith, relevancy, precision, recall] = await Promise.all([
      faithfulness(answer.text, chunks, llm),
      answerRelevancy(c.question, answer.text, embedder, llm),
      contextPrecision(c.question, chunks, llm),
      contextRecall(c.groundTruthAnswer, chunks, llm),
    ])

    const overall = (faith.score + relevancy.score + precision.score + recall.score) / 4
    perCase.push({
      id: c.id, faithfulness: faith.score, answerRelevancy: relevancy.score,
      contextPrecision: precision.score, contextRecall: recall.score, overall,
    })
  }

  const average = Object.fromEntries(
    METRIC_KEYS.map((key) => [key, perCase.reduce((sum, r) => sum + r[key], 0) / perCase.length]),
  ) as Omit<CaseResult, 'id'>

  return { retrieval, perCase, average }
}
