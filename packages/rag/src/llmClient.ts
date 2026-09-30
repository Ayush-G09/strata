/** One small interface every model provider implements — the same shape Redline used, for the same reason: it
 * lets the rest of the pipeline (prompting, citation-checking, evaluation) be written and tested once, against a
 * fake, deterministic implementation, with zero changes needed when a real provider is swapped in. */
export interface LlmClient {
  generate(prompt: string): Promise<string>
}

/** A fully deterministic, network-free client for tests: returns a canned response, or runs a caller-supplied
 * function of the prompt. Never makes this pipeline's correctness depend on a real model's non-determinism. */
export class FakeLlmClient implements LlmClient {
  constructor(private respond: (prompt: string) => string) {}
  async generate(prompt: string): Promise<string> {
    return this.respond(prompt)
  }
}

/** Google's free-tier Gemini API — the same "runs for free" provider Redline uses, plain `fetch`, no SDK. */
export class GeminiClient implements LlmClient {
  constructor(private apiKey: string, private model = 'gemini-flash-lite-latest') {}

  async generate(prompt: string): Promise<string> {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${this.model}:generateContent?key=${this.apiKey}`
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }] }),
    })
    if (!res.ok) throw new Error(`Gemini request failed: ${res.status} ${await res.text()}`)
    const data = (await res.json()) as { candidates?: { content?: { parts?: { text?: string }[] } }[] }
    return data.candidates?.[0]?.content?.parts?.[0]?.text ?? ''
  }
}

/** A fully local model via Ollama — no key, no network beyond localhost, nothing leaving the machine. */
export class OllamaClient implements LlmClient {
  constructor(private model = 'qwen2.5-coder:3b', private baseUrl = 'http://localhost:11434') {}

  async generate(prompt: string): Promise<string> {
    const res = await fetch(`${this.baseUrl}/api/generate`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: this.model, prompt, stream: false }),
    })
    if (!res.ok) throw new Error(`Ollama request failed: ${res.status} ${await res.text()}`)
    const data = (await res.json()) as { response?: string }
    return data.response ?? ''
  }
}
