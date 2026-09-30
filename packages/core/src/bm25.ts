import type { Chunk } from './types'

function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[a-z0-9]+/g) ?? []
}

interface Posting { chunkIndex: number; termFreq: number }

/** Classic BM25 keyword ranking over a fixed corpus of chunks. Complements the vector store: BM25 is exact-term
 * matching (a document mentioning "RRF" is found by the literal string, no matter what an embedding might do with
 * an unfamiliar acronym), where an embedding is approximate/semantic. Standard formula, k1=1.5, b=0.75 —
 * unremarkable, well-documented constants, not tuned here. */
export class BM25Index {
  private chunks: Chunk[] = []
  private docLengths: number[] = []
  private avgDocLength = 0
  private postings = new Map<string, Posting[]>()
  private readonly k1 = 1.5
  private readonly b = 0.75

  constructor(chunks: Chunk[]) {
    this.chunks = chunks
    let totalLength = 0
    chunks.forEach((chunk, chunkIndex) => {
      const tokens = tokenize(chunk.text)
      this.docLengths[chunkIndex] = tokens.length
      totalLength += tokens.length
      const freq = new Map<string, number>()
      for (const t of tokens) freq.set(t, (freq.get(t) ?? 0) + 1)
      for (const [term, termFreq] of freq) {
        if (!this.postings.has(term)) this.postings.set(term, [])
        this.postings.get(term)!.push({ chunkIndex, termFreq })
      }
    })
    this.avgDocLength = chunks.length > 0 ? totalLength / chunks.length : 0
  }

  get size(): number {
    return this.chunks.length
  }

  /** BM25 score for every chunk containing at least one query term, sorted descending. A chunk matching none of
   * the query terms is omitted entirely (score 0 is not the same as "somewhat relevant"). */
  search(query: string, k: number): { chunk: Chunk; score: number }[] {
    const N = this.chunks.length
    if (N === 0) return []
    const terms = tokenize(query)
    const scores = new Map<number, number>()

    for (const term of terms) {
      const postingList = this.postings.get(term)
      if (!postingList) continue
      const df = postingList.length
      const idf = Math.log((N - df + 0.5) / (df + 0.5) + 1) // BM25's IDF, always non-negative (unlike classic IDF)
      for (const { chunkIndex, termFreq } of postingList) {
        const docLen = this.docLengths[chunkIndex]
        const denom = termFreq + this.k1 * (1 - this.b + (this.b * docLen) / (this.avgDocLength || 1))
        const score = idf * ((termFreq * (this.k1 + 1)) / denom)
        scores.set(chunkIndex, (scores.get(chunkIndex) ?? 0) + score)
      }
    }

    return [...scores.entries()]
      .map(([chunkIndex, score]) => ({ chunk: this.chunks[chunkIndex], score }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
  }
}
