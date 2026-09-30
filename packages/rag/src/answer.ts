import type { Chunk } from '@strata/core'

export interface ParsedAnswer {
  text: string
  /** chunk ids actually cited, in the order first cited, de-duplicated */
  citedChunkIds: string[]
  /** citation markers like [7] that don't correspond to any numbered source actually given to the model —
   * a model citing a source that was never shown to it, whether from confusion or invention, either way something
   * a caller should know about rather than silently trust */
  invalidCitationNumbers: number[]
}

/** Extracts and validates `[n]`-style citation markers against the exact chunk list the prompt was built from
 * (same order, 1-indexed) — this is the citation-grounding equivalent of Redline's "a comment must point at a
 * line the diff actually contains" check. A model can still say something false, but it cannot cite a source that
 * was never given to it without that being caught here. */
export function parseAnswer(rawText: string, sourceChunks: Chunk[]): ParsedAnswer {
  const citedChunkIds: string[] = []
  const invalidCitationNumbers: number[] = []
  const seen = new Set<string>()

  for (const match of rawText.matchAll(/\[(\d+)\]/g)) {
    const n = Number(match[1])
    const chunk = sourceChunks[n - 1]
    if (!chunk) {
      if (!invalidCitationNumbers.includes(n)) invalidCitationNumbers.push(n)
      continue
    }
    if (!seen.has(chunk.id)) {
      seen.add(chunk.id)
      citedChunkIds.push(chunk.id)
    }
  }

  return { text: rawText, citedChunkIds, invalidCitationNumbers }
}
