/// <reference lib="webworker" />
import {
  AutoProcessor,
  Gemma4ForConditionalGeneration,
  InterruptableStoppingCriteria,
  ModelRegistry,
  RawImage,
  TextStreamer,
  env,
  type Message,
  type ProgressInfo,
} from '@huggingface/transformers'
import { DTYPE, findModel } from '../config/models'
import type {
  AdapterInfo,
  DecodeProfile,
  GenerationSettings,
  PerfOptions,
  WireMessage,
  WorkerRequest,
  WorkerResponse,
} from './protocol'

declare const self: DedicatedWorkerGlobalScope

type Processor = Awaited<ReturnType<typeof AutoProcessor.from_pretrained>>
type Model = Awaited<ReturnType<typeof Gemma4ForConditionalGeneration.from_pretrained>>

type Tokenizer = NonNullable<Processor['tokenizer']>

/**
 * Gemma 4 wraps its reasoning in `<|channel>thought\n…<channel|>`. Both markers are special tokens, which TextStreamer
 * drops without flushing its word buffer, so text from either side would merge. This flushes at each marker and
 * reports every piece of text with the channel it belongs to (minus the `thought\n` header).
 */
class ChannelStreamer extends TextStreamer {
  private open: bigint
  private close: bigint
  private onText: (text: string, thought: boolean) => void
  private inThought = false
  /** Text seen since the channel opened, held back until it's clear whether it is the `thought\n` header. */
  private head: string | null = null

  constructor(
    tokenizer: Tokenizer,
    onText: (text: string, thought: boolean) => void,
    options: ConstructorParameters<typeof TextStreamer>[1],
  ) {
    super(tokenizer, options)
    this.open = BigInt(tokenizer.convert_tokens_to_ids('<|channel>'))
    this.close = BigInt(tokenizer.convert_tokens_to_ids('<channel|>'))
    this.onText = onText
  }

  put(value: bigint[][]) {
    const token = value[0]?.length === 1 ? value[0][0] : undefined
    if (token === this.open || token === this.close) {
      if (this.token_cache.length > 0) {
        const text = this.tokenizer.decode(this.token_cache, this.decode_kwargs)
        this.on_finalized_text(text.slice(this.print_len), false)
        this.token_cache = []
        this.print_len = 0
      }
      this.flushHead()
      this.inThought = token === this.open
      this.head = this.inThought ? '' : null
    }
    super.put(value)
  }

  on_finalized_text(text: string, streamEnd: boolean) {
    if (this.head !== null) {
      this.head += text
      if (this.head !== 'thought\n' && 'thought\n'.startsWith(this.head) && !streamEnd) return
      text = this.head.replace(/^thought\n/, '')
      this.head = null
    }
    if (text) this.onText(text, this.inThought)
  }

  private flushHead() {
    if (this.head) this.onText(this.head, this.inThought)
    this.head = null
  }
}

const DEVICE = 'webgpu'

let processor: Processor | null = null
let model: Model | null = null
const stopping = new InterruptableStoppingCriteria()

function post(msg: WorkerResponse) {
  self.postMessage(msg)
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}

function errorStack(err: unknown): string | undefined {
  return err instanceof Error ? err.stack : undefined
}

/** Which step of the current request is running, so errors can say where they happened. */
let stage = ''

/** Cumulative wall time spent in each ONNX session's run(), keyed by session name. */
const sessionTime: Record<string, number> = {}

type OrtTensor = {
  location: string
  type: string
  dims: readonly number[]
  getData: (releaseData?: boolean) => Promise<unknown>
  dispose: () => void
}
type RunnableSession = { run: (feeds: Record<string, OrtTensor>, ...rest: unknown[]) => Promise<Record<string, OrtTensor>> }

const EMBED_OUTPUTS = ['inputs_embeds', 'per_layer_inputs']

/** embed_tokens GPU outputs waiting to be consumed by the next decoder run, then freed. */
let pendingGpuTensors: OrtTensor[] = []

function releasePendingGpuTensors() {
  for (const t of pendingGpuTensors) t.dispose()
  pendingGpuTensors = []
}

/**
 * With `embedOutputsOnGpu`, embed_tokens returns GPU tensors. Decode steps pass them straight to the decoder (and
 * free them afterwards). The prefill pass may merge image/audio features into the embeddings on the CPU, so there
 * they are downloaded to CPU tensors first.
 */
async function settleEmbedOutputs(feeds: Record<string, OrtTensor>, outputs: Record<string, OrtTensor>) {
  const prefill = (feeds.input_ids?.dims.at(-1) ?? 1) > 1
  for (const name of EMBED_OUTPUTS) {
    const t = outputs[name]
    if (t?.location !== 'gpu-buffer') continue
    if (prefill) {
      const Ctor = t.constructor as new (type: string, data: unknown, dims: readonly number[]) => OrtTensor
      outputs[name] = new Ctor(t.type, await t.getData(true), t.dims)
    } else {
      pendingGpuTensors.push(t)
    }
  }
}

/** Wrap every session's run() so decode steps can be broken down per session. */
function instrumentSessions(m: Model) {
  for (const [name, session] of Object.entries(m.sessions as Record<string, RunnableSession>)) {
    const run = session.run.bind(session)
    session.run = async (feeds, ...rest) => {
      const t0 = performance.now()
      try {
        const outputs = await run(feeds, ...rest)
        if (name === 'embed_tokens') await settleEmbedOutputs(feeds, outputs)
        return outputs
      } finally {
        sessionTime[name] = (sessionTime[name] ?? 0) + performance.now() - t0
        if (name === 'decoder_model_merged') releasePendingGpuTensors()
      }
    }
  }
}

async function onnxAdapterInfo(): Promise<AdapterInfo | undefined> {
  try {
    const device = (await env.backends.onnx.webgpu?.device) as (GPUDevice & { adapterInfo?: GPUAdapterInfo }) | undefined
    const info = device?.adapterInfo ?? (env.backends.onnx.webgpu?.adapter as GPUAdapter | undefined)?.info
    if (!info) return undefined
    return { vendor: info.vendor || undefined, architecture: info.architecture || undefined, description: info.description || undefined }
  } catch {
    return undefined
  }
}

/** Settings store model ids; Transformers.js wants the HF repo. */
function repoFor(modelId: string): string {
  const repo = findModel(modelId)?.transformers.repo
  if (!repo) throw new Error(`Unknown model: ${modelId}`)
  return repo
}

async function inspect(modelId: string) {
  const repo = repoFor(modelId)
  const options = { dtype: DTYPE, device: DEVICE } as const
  const cached = await ModelRegistry.is_cached(repo, options)
  let totalBytes: number | null = null
  try {
    const files = await ModelRegistry.get_files(repo, options)
    const metas = await Promise.all(files.map((f) => ModelRegistry.get_file_metadata(repo, f)))
    totalBytes = metas.reduce((sum, m) => sum + (m.size ?? 0), 0)
  } catch {
    // Size is informational only.
  }
  post({ type: 'inspect-result', runtime: 'transformers', modelId, cached, totalBytes })
}

async function unload() {
  await model?.dispose()
  model = null
  processor = null
}

/** ORT session config entries; the WebGPU EP reads its options from keys with this prefix. */
function sessionConfig(perf: PerfOptions): Record<string, string> {
  const extra: Record<string, string> = {}
  if (perf.int64OnGpu) extra['ep.webgpuexecutionprovider.enableInt64'] = '1'
  if (perf.disableValidation) extra['ep.webgpuexecutionprovider.validationMode'] = 'disabled'
  return extra
}

async function load(modelId: string, perf: PerfOptions) {
  const repo = repoFor(modelId)
  stage = 'unload previous model'
  await unload()
  stage = 'cache check'
  const cached = await ModelRegistry.is_cached(repo, { dtype: DTYPE, device: DEVICE }).catch(() => false)

  const start = performance.now()
  let downloadEnd: number | null = null

  const progress_callback = (info: ProgressInfo) => {
    if (info.status !== 'progress_total') return
    const done = info.total > 0 && info.loaded >= info.total
    if (done && downloadEnd === null) downloadEnd = performance.now()
    post({
      type: 'load-progress',
      phase: done ? 'initializing' : 'downloading',
      loaded: info.loaded,
      total: info.total,
      files: info.files,
    })
  }

  stage = 'download + initialize'
  const [p, m] = await Promise.all([
    AutoProcessor.from_pretrained(repo),
    Gemma4ForConditionalGeneration.from_pretrained(repo, {
      dtype: DTYPE,
      device: DEVICE,
      progress_callback,
      session_options: {
        extra: sessionConfig(perf),
        // Matched per output name, so this only affects embed_tokens (Transformers.js sets the decoder's own map
        // for its KV cache, and the vision/audio encoders have different output names).
        ...(perf.embedOutputsOnGpu
          ? { preferredOutputLocation: Object.fromEntries(EMBED_OUTPUTS.map((n) => [n, 'gpu-buffer' as const])) }
          : {}),
      },
    }),
  ])
  processor = p
  model = m
  instrumentSessions(m)

  const end = performance.now()
  const dlEnd = downloadEnd ?? end
  post({
    type: 'load-done',
    metrics: {
      runtime: 'transformers',
      modelId,
      dtype: DTYPE,
      downloadMs: dlEnd - start,
      initMs: end - dlEnd,
      totalMs: end - start,
      fromCache: cached,
      adapter: await onnxAdapterInfo(),
      perf,
    },
  })
}

function describePrior(kinds: ('image' | 'audio')[] | undefined): string {
  if (!kinds?.length) return ''
  return kinds.map((k) => `[The user attached ${k === 'image' ? 'an image' : 'an audio clip'} here.]\n`).join('')
}

async function generate(requestId: string, messages: WireMessage[], settings: GenerationSettings) {
  if (!model || !processor?.tokenizer) throw new Error('Model is not loaded')
  const tokenizer = processor.tokenizer

  // The Gemma 4 processor only handles media for a single prompt, and audio is limited to one clip,
  // so only the latest turn carries real image/audio; earlier attachments are described in text.
  stage = 'build messages'
  let image: RawImage | null = null
  let audio: Float32Array | null = null
  const chat: Message[] = await Promise.all(
    messages.map(async (msg): Promise<Message> => {
      const text = describePrior(msg.priorAttachments) + msg.text
      // The Gemma 4 template trims the system turn's content directly, so it must be a plain string.
      // Text-only turns use a string too; only turns with media need content parts.
      if (!msg.image && !msg.audio) return { role: msg.role, content: text }

      const content: Exclude<Message['content'], string> = []
      if (msg.image) {
        image = await RawImage.read(msg.image)
        content.push({ type: 'image' })
      }
      if (msg.audio) {
        audio = msg.audio
        content.push({ type: 'audio' })
      }
      content.push({ type: 'text', text })
      return { role: msg.role, content }
    }),
  )

  stage = 'apply chat template'
  // Extra keys are forwarded to the Jinja chat template, but aren't part of the published options type.
  const templateOptions = { add_generation_prompt: true, enable_thinking: settings.enableThinking }
  const prompt = processor.apply_chat_template(chat, templateOptions as Parameters<Processor['apply_chat_template']>[1])
  console.debug('[llm worker] rendered prompt:\n' + prompt)

  stage = 'process inputs (tokenize / image / audio)'
  const start = performance.now()
  const inputs = await processor(prompt, image, audio, { add_special_tokens: false })
  const promptTokens: number = inputs.input_ids.dims.at(-1)

  let firstTokenAt: number | null = null
  let sessionTimeAtFirstToken: Record<string, number> = {}
  let outputTokens = 0
  const streamer = new ChannelStreamer(tokenizer, (text, thought) => post({ type: 'token', requestId, text, thought }), {
    skip_prompt: true,
    skip_special_tokens: true,
    token_callback_function: () => {
      if (firstTokenAt === null) {
        firstTokenAt = performance.now()
        sessionTimeAtFirstToken = { ...sessionTime }
      }
      outputTokens++
    },
  })

  stage = 'generate'
  releasePendingGpuTensors()
  stopping.reset()
  const sample = settings.temperature > 0
  await model.generate({
    ...inputs,
    max_new_tokens: settings.maxNewTokens,
    do_sample: sample,
    ...(sample ? { temperature: settings.temperature, top_k: settings.topK } : {}),
    streamer,
    stopping_criteria: stopping,
  })

  const end = performance.now()
  const first = firstTokenAt ?? end
  const decodeSec = (end - first) / 1000

  // Each token after the first costs exactly one forward pass, so per-step = time since first token / (tokens - 1).
  let profile: DecodeProfile | undefined
  const steps = outputTokens - 1
  if (steps > 0) {
    const msPerStep = (end - first) / steps
    const sessionMsPerStep: Record<string, number> = {}
    for (const [name, total] of Object.entries(sessionTime)) {
      const delta = total - (sessionTimeAtFirstToken[name] ?? 0)
      if (delta > 0) sessionMsPerStep[name] = delta / steps
    }
    const inSessions = Object.values(sessionMsPerStep).reduce((a, b) => a + b, 0)
    profile = { steps, msPerStep, sessionMsPerStep, otherMsPerStep: Math.max(0, msPerStep - inSessions) }
  }

  const metrics = {
    ttftMs: first - start,
    totalMs: end - start,
    promptTokens,
    outputTokens,
    tokensPerSec: outputTokens > 1 && decodeSec > 0 ? (outputTokens - 1) / decodeSec : 0,
    interrupted: stopping.interrupted,
    profile,
  }
  console.debug('[llm worker] generation metrics', metrics)
  post({ type: 'generate-done', requestId, metrics })
}

function reportError(scope: 'inspect' | 'load' | 'generate', err: unknown, requestId?: string) {
  console.error(`[llm worker] ${scope} failed during "${stage}":`, err)
  post({ type: 'error', scope, requestId, stage, message: errorMessage(err), stack: errorStack(err) })
}

self.addEventListener('message', async (event: MessageEvent<WorkerRequest>) => {
  const req = event.data
  switch (req.type) {
    case 'inspect':
      try {
        stage = 'inspect'
        await inspect(req.modelId)
      } catch (err) {
        reportError('inspect', err)
      }
      break
    case 'load':
      try {
        console.info('[llm worker] loading', req.modelId, 'with session config', sessionConfig(req.perf))
        await load(req.modelId, req.perf)
      } catch (err) {
        reportError('load', err)
        await unload().catch(() => {})
      }
      break
    case 'unload':
      await unload()
      post({ type: 'unloaded' })
      break
    case 'generate':
      try {
        await generate(req.requestId, req.messages, req.settings)
      } catch (err) {
        reportError('generate', err, req.requestId)
      }
      break
    case 'interrupt':
      stopping.interrupt()
      break
  }
})
