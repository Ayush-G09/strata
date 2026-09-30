/** A chunk always carries its exact source offsets — the whole point is that a retrieved answer can be traced
 * back to the exact span of the original document that produced it, never just "somewhere in the text." */
export interface Chunk {
  id: string
  docId: string
  text: string
  /** inclusive-exclusive character offsets into the ORIGINAL document text, before any chunking */
  start: number
  end: number
  index: number
}

export interface Document {
  id: string
  text: string
}

export type ChunkingStrategy = 'fixed' | 'structural'

export interface ChunkingOptions {
  strategy: ChunkingStrategy
  /** target chunk size in characters (approximate for 'structural', which prefers not to split paragraphs) */
  size: number
  /** how many characters of overlap between consecutive chunks (ignored by 'structural') */
  overlap: number
}
