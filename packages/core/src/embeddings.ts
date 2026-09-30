/** Anything that turns text into vectors — a real provider (Gemini, a local model) in production, or the
 * deterministic hashing embedder below for tests and for running with no network access and no API key at all. */
export interface Embedder {
  embed(texts: string[]): Promise<number[][]>
  readonly dims: number
}

function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[a-z0-9]+/g) ?? []
}

/** A simple, fast string hash (djb2) — deterministic across runs and processes, which is what lets two calls to
 * `embed` with the same text always produce the exact same vector, with no model and no network. */
function djb2(s: string): number {
  let h = 5381
  for (let i = 0; i < s.length; i++) h = (h * 33) ^ s.charCodeAt(i)
  return h >>> 0
}

/** Feature-hashing bag-of-words embedder: each token hashes into one of `dims` buckets, with its sign also
 * derived from the hash (the standard "hashing trick") so unrelated tokens partially cancel instead of only ever
 * adding up. Not a semantic embedding — two paraphrases with different words score low — but it is deterministic,
 * needs no API key or network, and is genuinely useful as a keyword-ish fallback and as a fast, reproducible
 * embedder for tests that must not depend on an external model's specific output.
 */
export class HashingEmbedder implements Embedder {
  constructor(readonly dims = 256) {}

  async embed(texts: string[]): Promise<number[][]> {
    return texts.map((text) => {
      const vec = new Array(this.dims).fill(0)
      for (const token of tokenize(text)) {
        const h = djb2(token)
        const bucket = h % this.dims
        const sign = (h & 1) === 0 ? 1 : -1
        vec[bucket] += sign
      }
      const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0))
      return norm > 0 ? vec.map((v) => v / norm) : vec
    })
  }
}

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) throw new RangeError('vectors must have the same dimension')
  let dot = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  if (na === 0 || nb === 0) return 0
  return dot / (Math.sqrt(na) * Math.sqrt(nb))
}
