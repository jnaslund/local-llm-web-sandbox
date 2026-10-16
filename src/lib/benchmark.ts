import { useSession, type BenchmarkResult } from '../state/session'
import type { GenerationSettings, WireMessage } from '../worker/protocol'
import { generate } from './llm'

const LONG_PROMPT = 'Write a detailed, 400-word explanation of how a bicycle gear system works.'
const FOLLOWUP_PROMPT = 'Now summarize that explanation in two sentences.'
const greedy = { temperature: 0, topK: 1, enableThinking: false }
// Same as long-greedy: a kept conversation is only continued with unchanged settings (maxOutputTokens is fixed per
// conversation). The two-sentence answer stops well before the limit.
const followupSettings: GenerationSettings = { ...greedy, maxNewTokens: 256 }

/**
 * Fixed prompts so results are comparable across devices. The two long runs differ only in decoding strategy,
 * which isolates the cost of sampling (done on the CPU over the full vocabulary) from the forward pass.
 */
export const BENCHMARK_PROMPTS: { name: string; prompt: string; settings: GenerationSettings }[] = [
  { name: 'short-answer', prompt: 'What is the capital of France? Answer in one sentence.', settings: { ...greedy, maxNewTokens: 64 } },
  { name: 'long-greedy', prompt: LONG_PROMPT, settings: { ...greedy, maxNewTokens: 256 } },
  {
    name: 'long-sampled',
    prompt: LONG_PROMPT,
    settings: { temperature: 1, topK: 64, maxNewTokens: 256, enableThinking: false },
  },
]

export async function runBenchmark() {
  const session = useSession.getState()
  if (!session.loadedModelId || session.generating || session.benchmarking) return
  session.set({ benchmarking: true, generating: true })

  const results: BenchmarkResult[] = []
  try {
    let longReply = ''
    for (const b of BENCHMARK_PROMPTS) {
      let reply = ''
      const metrics = await generate([{ role: 'user', text: b.prompt }], b.settings, (t) => (reply += t))
      if (b.name === 'long-greedy') longReply = reply
      results.push({ name: b.name, metrics })
    }

    // Second turn after the long reply (~300-token history). `followup-stateless` re-prefills the whole history, as
    // every chat turn does on Transformers.js. `followup-stateful` (LiteRT-LM) continues a kept conversation, so only
    // the new message is prefilled. It is a runtime capability, so it is reported separately rather than as raw speed.
    const followup = (reply: string): WireMessage[] => [
      { role: 'user', text: LONG_PROMPT },
      { role: 'assistant', text: reply },
      { role: 'user', text: FOLLOWUP_PROMPT },
    ]
    if (session.loadMetrics?.runtime === 'litert') {
      const id = `benchmark-${crypto.randomUUID()}`
      let reply = ''
      await generate([{ role: 'user', text: LONG_PROMPT }], followupSettings, (t) => (reply += t), id)
      results.push({ name: 'followup-stateful', metrics: await generate(followup(reply), followupSettings, undefined, id) })
    }
    results.push({ name: 'followup-stateless', metrics: await generate(followup(longReply), followupSettings) })
    const s = useSession.getState()
    s.set({
      benchmarks: [
        ...s.benchmarks,
        {
          runtime: session.loadMetrics?.runtime ?? 'transformers',
          modelId: session.loadedModelId,
          perf: session.loadMetrics?.perf,
          startedAt: new Date().toISOString(),
          results,
        },
      ],
    })
  } finally {
    useSession.getState().set({ benchmarking: false, generating: false })
  }
}
