import type { Chunk, RankedResult } from '@strata/core'

function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[a-z0-9]+/g) ?? []
}

/** The "number to beat": ranks chunks purely by how many distinct query terms appear anywhere in the chunk's
 * text, with no IDF term-rarity weighting and no semantic matching at all — ties broken by original chunk order.
 * This is deliberately what naive keyword overlap looks like without either of hybrid retrieval's two real
 * ingredients, so there is a concrete, measured baseline to actually beat instead of an assumed one. */
export async function naiveRetrieve(query: string, chunks: Chunk[], k: number): Promise<RankedResult[]> {
  const queryTerms = new Set(tokenize(query))
  return chunks
    .map((chunk, originalIndex) => {
      const chunkTerms = new Set(tokenize(chunk.text))
      let matches = 0
      for (const t of queryTerms) if (chunkTerms.has(t)) matches++
      return { chunk, fusedScore: matches, vectorRank: null, bm25Rank: null, originalIndex }
    })
    .sort((a, b) => b.fusedScore - a.fusedScore || a.originalIndex - b.originalIndex)
    .slice(0, k)
    .map(({ chunk, fusedScore, vectorRank, bm25Rank }): RankedResult => ({ chunk, fusedScore, vectorRank, bm25Rank }))
}
