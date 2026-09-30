import {
  BM25Index, chunkDocument, HybridRetriever, VectorStore,
} from '@strata/core'
import type { Chunk, ChunkingOptions, Document, Embedder, RankedResult } from '@strata/core'
import { parseAnswer } from './answer'
import type { ParsedAnswer } from './answer'
import type { LlmClient } from './llmClient'
import { buildPrompt } from './prompt'

export interface QueryResult {
  answer: ParsedAnswer
  retrieved: RankedResult[]
}

/** An in-memory Strata index: chunks a set of documents once, embeds them, and answers queries against them via
 * hybrid retrieval + grounded generation. Everything needed to run the whole pipeline end to end with no external
 * services beyond whichever `LlmClient` and `Embedder` are supplied — a `FakeLlmClient` and `HashingEmbedder` make
 * the whole thing runnable, and testable, completely offline. */
export class StrataIndex {
  private allChunks: Chunk[] = []
  private vectorStore = new VectorStore()
  private bm25: BM25Index = new BM25Index([])

  constructor(private embedder: Embedder, private llm: LlmClient) {}

  async addDocument(doc: Document, options: ChunkingOptions): Promise<Chunk[]> {
    const chunks = chunkDocument(doc, options)
    if (chunks.length === 0) return []
    const vectors = await this.embedder.embed(chunks.map((c) => c.text))
    chunks.forEach((c, i) => this.vectorStore.add(c, vectors[i]))
    this.allChunks.push(...chunks)
    this.bm25 = new BM25Index(this.allChunks) // rebuilt on each add: simplicity over incremental-index complexity,
    // appropriate for the corpus sizes this project is scoped to (see Honest limits)
    return chunks
  }

  get chunkCount(): number {
    return this.allChunks.length
  }

  /** `retrieve` overrides how candidate chunks are found, while reusing the same prompting/generation/citation
   * pipeline — this is what lets the evaluation harness compare hybrid retrieval against a naive baseline without
   * duplicating the generation half of the pipeline for each. Defaults to the real hybrid retriever. */
  async query(
    question: string, k = 5,
    retrieve: (q: string, k: number) => Promise<RankedResult[]> = (q, kk) =>
      new HybridRetriever(this.bm25, this.vectorStore, this.embedder).search(q, kk),
  ): Promise<QueryResult> {
    const retrieved = await retrieve(question, k)
    const prompt = buildPrompt(question, retrieved.map((r) => r.chunk))
    const raw = await this.llm.generate(prompt)
    const answer = parseAnswer(raw, retrieved.map((r) => r.chunk))
    return { answer, retrieved }
  }

  get chunks(): Chunk[] {
    return this.allChunks
  }
}
