import type { Chunk } from '@strata/core'

/** Builds a grounding prompt: each retrieved chunk is shown numbered, and the model is told explicitly to answer
 * only from these sources, citing the number inline, and to say so plainly if they don't contain the answer —
 * the same "don't let the model invent an answer with nowhere to point back to" discipline as Redline's
 * line-verified review comments, applied here to citations instead of diff lines. */
export function buildPrompt(query: string, chunks: Chunk[]): string {
  const sources = chunks.map((c, i) => `[${i + 1}] ${c.text}`).join('\n\n')
  return [
    'Answer the question using ONLY the numbered sources below. Cite the source number inline like [1] for every',
    'claim you make. If the sources do not contain enough information to answer, say so plainly instead of guessing.',
    '',
    'Sources:',
    sources,
    '',
    `Question: ${query}`,
    '',
    'Answer:',
  ].join('\n')
}
