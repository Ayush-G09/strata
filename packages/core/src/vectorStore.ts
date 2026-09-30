import { cosineSimilarity } from './embeddings'
import type { Chunk } from './types'

export interface ScoredChunk {
  chunk: Chunk
  score: number
}

/** An in-memory vector index: linear-scan cosine similarity. Deliberately simple — correctness and
 * traceability matter more here than raw throughput, and a linear scan is trivial to verify against a brute-force
 * reference (it effectively IS the brute-force reference). Swapping in an ANN index later would be an
 * implementation detail behind this same interface, not a change to anything that depends on it. */
export class VectorStore {
  private entries: { chunk: Chunk; vector: number[] }[] = []

  add(chunk: Chunk, vector: number[]): void {
    this.entries.push({ chunk, vector })
  }

  get size(): number {
    return this.entries.length
  }

  search(queryVector: number[], k: number): ScoredChunk[] {
    return this.entries
      .map(({ chunk, vector }) => ({ chunk, score: cosineSimilarity(queryVector, vector) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
  }
}
