/**
 * Runs the golden set against a REAL configured model (Gemini or Ollama, whichever `createLlmClient` resolves to)
 * and prints real, measured scores — both for the real hybrid retriever and the naive baseline, so the number
 * hybrid retrieval actually has to beat is also a real, printed number, not an assumed one. The same discipline
 * Redline's `npm run eval` uses.
 */
import { HashingEmbedder } from '@strata/core'
import { runGoldenSet } from '@strata/eval'
import { createLlmClient } from '../src/config'

function fmt(n: number): string {
  return `${(n * 100).toFixed(1)}%`
}

async function main() {
  const llm = createLlmClient()
  const embedder = new HashingEmbedder()

  console.log('Running golden set with the naive keyword-overlap baseline...')
  const naive = await runGoldenSet(llm, embedder, { retrieval: 'naive' })

  console.log('Running golden set with Strata\'s real hybrid retriever...')
  const hybrid = await runGoldenSet(llm, embedder, { retrieval: 'hybrid' })

  console.log('\n=== Per-case scores (hybrid) ===')
  for (const c of hybrid.perCase) {
    console.log(
      `  ${c.id.padEnd(20)} faithfulness=${fmt(c.faithfulness)} relevancy=${fmt(c.answerRelevancy)} `
      + `precision=${fmt(c.contextPrecision)} recall=${fmt(c.contextRecall)} overall=${fmt(c.overall)}`,
    )
  }

  console.log('\n=== Averages ===')
  console.log('  naive baseline :', Object.entries(naive.average).map(([k, v]) => `${k}=${fmt(v)}`).join('  '))
  console.log('  hybrid (real)  :', Object.entries(hybrid.average).map(([k, v]) => `${k}=${fmt(v)}`).join('  '))
}

main().catch((err) => {
  console.error('[runEval] FAILED:', err)
  process.exitCode = 1
})
