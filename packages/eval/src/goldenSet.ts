import type { ChunkingOptions, Document } from '@strata/core'

export interface GoldenCase {
  id: string
  corpus: Document[]
  chunking: ChunkingOptions
  question: string
  groundTruthAnswer: string
}

const structural: ChunkingOptions = { strategy: 'structural', size: 300, overlap: 0 }

const handbook: Document[] = [
  { id: 'vacation', text: 'Vacation policy: Employees accrue 1.5 days of paid time off per month, up to a maximum of 20 days per year. Unused days roll over up to a cap of 5 days into the next year.' },
  { id: 'remote', text: 'Remote work policy: Employees may work remotely up to 3 days per week with manager approval. Fully remote arrangements require VP approval.' },
  { id: 'expenses', text: 'Expense policy: Expenses under $50 do not require a receipt. Expenses over $50 require an itemized receipt submitted within 30 days.' },
]

const product: Document[] = [
  { id: 'retrieval', text: "Strata's hybrid retriever combines BM25 keyword search with vector similarity search using Reciprocal Rank Fusion." },
  { id: 'chunking', text: "Strata's structural chunker merges consecutive paragraphs up to a configured size limit, splitting any single paragraph that alone exceeds it." },
  { id: 'providers', text: 'Strata supports two LLM providers out of the box: Gemini, a free hosted model, and Ollama, which runs fully locally with no API key.' },
]

/** Six real questions against two small synthetic corpora, each with distractor documents that are topically
 * related but not the answer — the same "planted, verifiable" discipline as Redline's golden set. A retriever
 * that just grabs the first document, or one that can't tell "remote work" from "vacation," fails these visibly,
 * not just in aggregate. */
export const GOLDEN_SET: GoldenCase[] = [
  { id: 'pto-accrual', corpus: handbook, chunking: structural, question: 'How many paid time off days do employees accrue per month?', groundTruthAnswer: 'Employees accrue 1.5 days of paid time off per month.' },
  { id: 'remote-days', corpus: handbook, chunking: structural, question: 'How many days per week can an employee work remotely with just manager approval?', groundTruthAnswer: 'Up to 3 days per week with manager approval; fully remote requires VP approval.' },
  { id: 'receipt-threshold', corpus: handbook, chunking: structural, question: 'Do I need a receipt for a $30 expense?', groundTruthAnswer: 'No, expenses under $50 do not require a receipt.' },
  { id: 'pto-rollover', corpus: handbook, chunking: structural, question: 'How many unused vacation days roll over to the next year?', groundTruthAnswer: 'Up to 5 days of unused vacation days roll over to the next year.' },
  { id: 'llm-providers', corpus: product, chunking: structural, question: 'Which two LLM providers does Strata support out of the box?', groundTruthAnswer: 'Strata supports Gemini, a free hosted model, and Ollama, which runs fully locally with no API key.' },
  { id: 'fusion-algorithm', corpus: product, chunking: structural, question: "What algorithm does Strata's hybrid retriever use to combine BM25 and vector search?", groundTruthAnswer: 'It uses Reciprocal Rank Fusion (RRF) to combine BM25 keyword search with vector similarity search.' },
]
