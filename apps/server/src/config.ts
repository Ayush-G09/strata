import { GeminiClient, OllamaClient } from '@strata/rag'
import type { LlmClient } from '@strata/rag'

/** Same provider-selection shape as Redline: default to the free hosted model if a key is present, fall back to
 * a fully local model with no key at all, or let an env var force one explicitly. */
export function createLlmClient(): LlmClient {
  const forced = process.env.STRATA_PROVIDER
  if (forced === 'ollama') return new OllamaClient(process.env.OLLAMA_MODEL)
  if (forced === 'gemini' || process.env.GEMINI_API_KEY) {
    if (!process.env.GEMINI_API_KEY) throw new Error('STRATA_PROVIDER=gemini requires GEMINI_API_KEY to be set')
    return new GeminiClient(process.env.GEMINI_API_KEY)
  }
  return new OllamaClient(process.env.OLLAMA_MODEL)
}

export const PORT = Number(process.env.PORT ?? 8890)
