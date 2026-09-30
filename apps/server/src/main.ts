import { createServer } from 'node:http'
import { HashingEmbedder } from '@strata/core'
import { StrataIndex } from '@strata/rag'
import type { ChunkingOptions } from '@strata/core'
import { GOLDEN_SET, runGoldenSet } from '@strata/eval'
import { createLlmClient, PORT } from './config'

const embedder = new HashingEmbedder()
const llm = createLlmClient()
const index = new StrataIndex(embedder, llm)

function send(res: import('node:http').ServerResponse, status: number, body: unknown) {
  const json = JSON.stringify(body)
  res.writeHead(status, {
    'content-type': 'application/json',
    'access-control-allow-origin': '*',
    'access-control-allow-headers': 'content-type',
    'access-control-allow-methods': 'GET, POST, OPTIONS',
  })
  res.end(json)
}

async function readBody(req: import('node:http').IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = []
  for await (const chunk of req) chunks.push(chunk as Buffer)
  const raw = Buffer.concat(chunks).toString('utf8')
  return raw ? JSON.parse(raw) : {}
}

const server = createServer(async (req, res) => {
  try {
    if (req.method === 'OPTIONS') { send(res, 204, {}); return }

    if (req.method === 'GET' && req.url === '/health') {
      send(res, 200, { ok: true, chunkCount: index.chunkCount })
      return
    }

    if (req.method === 'POST' && req.url === '/documents') {
      const body = (await readBody(req)) as { id?: string; text?: string; chunking?: ChunkingOptions }
      if (!body.id || !body.text) { send(res, 400, { error: 'id and text are required' }); return }
      const chunking: ChunkingOptions = body.chunking ?? { strategy: 'structural', size: 500, overlap: 0 }
      const chunks = await index.addDocument({ id: body.id, text: body.text }, chunking)
      send(res, 200, { chunksAdded: chunks.length, totalChunks: index.chunkCount })
      return
    }

    if (req.method === 'POST' && req.url === '/query') {
      const body = (await readBody(req)) as { question?: string; k?: number }
      if (!body.question) { send(res, 400, { error: 'question is required' }); return }
      const result = await index.query(body.question, body.k ?? 5)
      send(res, 200, {
        answer: result.answer,
        retrieved: result.retrieved.map((r) => ({
          chunkId: r.chunk.id, docId: r.chunk.docId, text: r.chunk.text,
          fusedScore: r.fusedScore, vectorRank: r.vectorRank, bm25Rank: r.bm25Rank,
        })),
      })
      return
    }

    if (req.method === 'GET' && req.url?.startsWith('/eval-sample')) {
      // Runs 2 real golden-set cases, both retrieval modes, through the ACTUAL configured LLM — the same
      // `runGoldenSet` the CLI script uses, just scoped small enough to answer within a UI request instead of
      // requiring the full `npm run eval` run. Real numbers on demand, not a canned demo response.
      const cases = GOLDEN_SET.slice(0, 2)
      const [hybrid, naive] = await Promise.all([
        runGoldenSet(llm, embedder, { retrieval: 'hybrid', cases }),
        runGoldenSet(llm, embedder, { retrieval: 'naive', cases }),
      ])
      send(res, 200, { hybrid, naive })
      return
    }

    send(res, 404, { error: 'not found' })
  } catch (err) {
    send(res, 500, { error: err instanceof Error ? err.message : String(err) })
  }
})

server.listen(PORT, () => {
  console.log(`Strata server listening on http://localhost:${PORT}`)
})
