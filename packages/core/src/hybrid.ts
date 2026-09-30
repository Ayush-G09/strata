import { BM25Index } from './bm25'
import type { Embedder } from './embeddings'
import { VectorStore } from './vectorStore'
import type { Chunk } from './types'

export interface RankedResult {
  chunk: Chunk
  fusedScore: number
  vectorRank: number | null
  bm25Rank: number | null
}

/** Reciprocal Rank Fusion: combines two independently-ranked lists into one, using only each item's RANK in each
 * list, never the raw scores — which is exactly what makes it possible to combine BM25 scores and cosine
 * similarities at all, since the two are not on any comparable scale. An item's fused score is the sum of
 * 1/(rankConstant + rank) across every list it appears in (rank starting at 1); absence from a list contributes 0,
 * not a penalty. `rankConstant` (60 is the standard default from the original RRF paper) softens the
 * difference between e.g. rank 1 and rank 2 versus rank 50 and rank 51. */
export function reciprocalRankFusion(
  rankedLists: { id: string; }[][],
  rankConstant = 60,
): Map<string, number> {
  const fused = new Map<string, number>()
  for (const list of rankedLists) {
    list.forEach((item, i) => {
      const rank = i + 1
      fused.set(item.id, (fused.get(item.id) ?? 0) + 1 / (rankConstant + rank))
    })
  }
  return fused
}

/** Combines a BM25 keyword index and a vector store's semantic search into one ranking via RRF — RAGFlow's own
 * documented approach (vector search + BM25 + rerank together), implemented from scratch here rather than reused,
 * so it can be tested and reasoned about directly. */
export class HybridRetriever {
  constructor(private bm25: BM25Index, private vectorStore: VectorStore, private embedder: Embedder) {}

  async search(query: string, k: number, candidatePoolSize = Math.max(k * 4, 20)): Promise<RankedResult[]> {
    const [queryVector] = await this.embedder.embed([query])
    const vectorResults = this.vectorStore.search(queryVector, candidatePoolSize)
    const bm25Results = this.bm25.search(query, candidatePoolSize)

    const vectorRank = new Map(vectorResults.map((r, i) => [r.chunk.id, i + 1]))
    const bm25Rank = new Map(bm25Results.map((r, i) => [r.chunk.id, i + 1]))
    const chunkById = new Map<string, Chunk>()
    for (const r of vectorResults) chunkById.set(r.chunk.id, r.chunk)
    for (const r of bm25Results) chunkById.set(r.chunk.id, r.chunk)

    const fused = reciprocalRankFusion([
      vectorResults.map((r) => ({ id: r.chunk.id })),
      bm25Results.map((r) => ({ id: r.chunk.id })),
    ])

    return [...fused.entries()]
      .map(([id, fusedScore]) => ({
        chunk: chunkById.get(id)!,
        fusedScore,
        vectorRank: vectorRank.get(id) ?? null,
        bm25Rank: bm25Rank.get(id) ?? null,
      }))
      .sort((a, b) => b.fusedScore - a.fusedScore)
      .slice(0, k)
  }
}
