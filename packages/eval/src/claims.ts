import type { LlmClient } from '@strata/rag'
import { parseLines } from './parsing'

/** Shared by faithfulness (decomposing the generated answer) and context recall (decomposing the ground-truth
 * answer) — both metrics work the same way underneath: break a piece of text into its individual factual claims,
 * then judge each claim against some context, one call at a time. */
export async function decomposeClaims(text: string, llm: LlmClient): Promise<string[]> {
  const prompt = [
    'List each distinct factual claim made in the following text, one claim per line, with no numbering, no',
    'bullets, and no extra commentary — just the claims themselves, each a short, standalone sentence.',
    '',
    'Text:',
    text,
  ].join('\n')
  const raw = await llm.generate(prompt)
  return parseLines(raw)
}
