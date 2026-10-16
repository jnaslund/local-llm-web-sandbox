/// <reference lib="webworker" />
// LiteRT-LM Web runtime. Speaks the same protocol as llm.worker.ts (Transformers.js); llm.ts runs one or the other.
import {
  Engine,
  LiteRtLm,
  SamplerType,
  getGlobalLiteRtLm,
  loadLiteRtLm,
  type Conversation,
  type Message,
  type SamplerParameters,
} from '@litert-lm/core'
import { findModel } from '../config/models'
import { inspectLiteRtModel, openLiteRtModel } from '../lib/litertCache'
import type { AdapterInfo, GenerationSettings, WireMessage, WorkerRequest, WorkerResponse } from './protocol'

declare const self: DedicatedWorkerGlobalScope

/** jsDelivr copy of the package's ~109 MB of wasm variants (the loader picks one by feature detection). */
const WASM_PATH = LiteRtLm.DEFAULT_WASM_PATH
/** Context length (prompt + output). */
const MAX_NUM_TOKENS = 8192

let engine: Engine | null = null
let conversation: Conversation | null = null
let interrupted = false
let wasmReady: Promise<LiteRtLm> | null = null

function post(msg: WorkerResponse) {
  self.postMessage(msg)
}

/** Which step of the current request is running, so errors can say where they happened. */
let stage = ''

function urlFor(modelId: string): string {
  const url = findModel(modelId)?.litert.url
  if (!url) throw new Error(`Unknown model: ${modelId}`)
  return url
}

/**
 * LiteRT-LM's loader runs its Emscripten glue with importScripts() inside workers, which module workers don't
 * support, and the glue would look for its .wasm next to this worker's URL. Provide a synchronous importScripts with
 * the same semantics (sync XHR + global-scope eval) and point locateFile at the wasm directory.
 */
function loadWasm(): Promise<LiteRtLm> {
  wasmReady ??= (() => {
    const scope = self as unknown as { importScripts: (...urls: string[]) => void; Module?: object }
    scope.importScripts = (...urls) => {
      for (const url of urls) {
        const xhr = new XMLHttpRequest()
        xhr.open('GET', url, false)
        xhr.send()
        if (xhr.status !== 200) throw new Error(`Failed to load ${url}: HTTP ${xhr.status}`)
        // Indirect eval runs in global scope, so the script's `var ModuleFactory` becomes a global like importScripts.
        // oxlint-disable-next-line no-eval
        ;(0, eval)(xhr.responseText)
      }
    }
    scope.Module = { locateFile: (path: string) => `${WASM_PATH}/${path}` }
    return loadLiteRtLm(WASM_PATH).catch((err) => {
      wasmReady = null
      throw err
    })
  })()
  return wasmReady
}

function adapterInfo(): AdapterInfo | undefined {
  try {
    const info = getGlobalLiteRtLm().liteRtLmWasm.preinitializedWebGPUDevice?.adapterInfo
    if (!info) return undefined
    return { vendor: info.vendor || undefined, architecture: info.architecture || undefined, description: info.description || undefined }
  } catch {
    return undefined
  }
}

async function inspect(modelId: string) {
  const { cached, totalBytes } = await inspectLiteRtModel(urlFor(modelId))
  post({ type: 'inspect-result', runtime: 'litert', modelId, cached, totalBytes })
}

async function unload() {
  conversation?.cancel()
  await dropLive()
  await engine?.delete()
  engine = null
}

async function load(modelId: string) {
  const url = urlFor(modelId)
  stage = 'unload previous model'
  await unload()

  const start = performance.now()
  let lastPost = 0
  const onProgress = (loaded: number, total: number) => {
    const now = performance.now()
    if (now - lastPost < 100 && loaded < total) return
    lastPost = now
    post({ type: 'load-progress', phase: 'downloading', loaded, total, files: { [url.split('/').pop()!]: { loaded, total } } })
  }

  stage = 'load wasm + open model'
  const [, source] = await Promise.all([loadWasm(), openLiteRtModel(url, onProgress)])
  // For a Blob (cache hit or fresh cache entry) the download is done; a raw stream (uncached) downloads during init.
  const downloadEnd = performance.now()
  if (source.warning) console.warn('[litert worker]', source.warning)

  stage = 'create engine'
  post({ type: 'load-progress', phase: 'initializing', loaded: 1, total: 1, files: {} })
  engine = await Engine.create({
    model: source.model,
    mainExecutorSettings: { maxNumTokens: MAX_NUM_TOKENS },
    benchmarkEnabled: true,
  })

  const end = performance.now()
  post({
    type: 'load-done',
    metrics: {
      runtime: 'litert',
      modelId,
      dtype: 'litertlm-web',
      downloadMs: downloadEnd - start,
      initMs: end - downloadEnd,
      totalMs: end - start,
      fromCache: source.fromCache,
      adapter: adapterInfo(),
      warning: source.warning,
    },
  })
}

function describePrior(kinds: ('image' | 'audio')[] | undefined): string {
  if (!kinds?.length) return ''
  return kinds.map((k) => `[The user attached ${k === 'image' ? 'an image' : 'an audio clip'} here.]\n`).join('')
}

/**
 * Text only: LiteRT-LM 0.18's LLM Engine has no vision/audio executor settings, so a conversation with
 * `visionModalityEnabled`/`audioModalityEnabled` fails with "Vision options should not be null" (verified
 * 2026-10-16; upstream main is the same). The UI disables attachments for this runtime; media that still arrives
 * (e.g. regenerating a turn sent with Transformers.js) is described in text like earlier turns' attachments.
 * If it lands: parts go to the C++ side as `{type: 'image'|'audio', blob: base64}`, not the JS types' `data: Blob`
 * (messages are JSON-serialized as-is, so a Blob would arrive as `{}`).
 */
function toMessage(msg: WireMessage): Message {
  const media: ('image' | 'audio')[] = [...(msg.image ? ['image' as const] : []), ...(msg.audio ? ['audio' as const] : [])]
  return { role: msg.role, content: describePrior([...(msg.priorAttachments ?? []), ...media]) + msg.text }
}

function sampler(settings: GenerationSettings): SamplerParameters {
  return settings.temperature > 0
    ? { type: SamplerType.TOP_K, k: settings.topK, temperature: settings.temperature }
    : { type: SamplerType.GREEDY }
}

/** Streamed chunks carry answer text in `content` and reasoning in `channels.thought`. */
function chunkParts(chunk: Message): { thought: string; text: string } {
  const thought = typeof chunk.channels?.thought === 'string' ? chunk.channels.thought : ''
  const c = chunk.content
  const text = typeof c === 'string' ? c : (c ?? []).map((p) => (p.type === 'text' ? p.text : '')).join('')
  return { thought, text }
}

/**
 * Stateful mode: at most one conversation lives on between requests (its KV cache stays on the GPU). It is reused
 * only when the request continues it exactly: same id (chat session), same settings, and a history equal to what it
 * has seen. Anything else (regenerate, edited/deleted turns, system prompt or settings change, another chat, a
 * stateless request) drops it and rebuilds from the full history. Only one conversation exists at a time.
 */
let live: { id: string; settingsKey: string; history: string[]; conv: Conversation } | null = null

const messageKey = (m: Message) => JSON.stringify([m.role, m.content])

async function dropLive() {
  const l = live
  live = null
  await l?.conv.delete().catch(() => {})
}

function continuesLive(id: string | undefined, settingsKey: string, preface: Message[]): boolean {
  if (!id || !live || live.id !== id || live.settingsKey !== settingsKey) return false
  const keys = preface.map(messageKey)
  return keys.length === live.history.length && keys.every((k, i) => k === live!.history[i])
}

async function generate(requestId: string, messages: WireMessage[], settings: GenerationSettings, conversationId?: string) {
  if (!engine) throw new Error('Model is not loaded')
  const last = messages.at(-1)
  if (last?.role !== 'user') throw new Error('The last message must be a user turn')
  interrupted = false

  stage = 'build messages'
  const preface = messages.slice(0, -1).map(toMessage)
  const message = toMessage(last)
  const settingsKey = JSON.stringify(settings)

  const start = performance.now()
  const reused = continuesLive(conversationId, settingsKey, preface)
  let conv: Conversation
  if (reused) {
    conv = live!.conv
    live = null // re-registered below once this turn completes cleanly
  } else {
    // Stateless (or a rebuild): a fresh conversation prefilled from the full history, like the Transformers.js path.
    stage = 'create conversation'
    await dropLive()
    conv = await engine.createConversation({
      preface: { messages: preface, extra_context: { enable_thinking: settings.enableThinking } },
      sessionConfig: {
        samplerParams: sampler(settings),
        maxOutputTokens: settings.maxNewTokens,
      },
    })
  }
  conversation = conv

  let firstTokenAt: number | null = null
  let chunks = 0
  let output = ''
  let keep = false
  let info: Awaited<ReturnType<Conversation['getBenchmarkInfo']>> | undefined
  try {
    stage = 'generate'
    const reader = conv.sendMessageStreaming(message).getReader()
    let thinking = false
    try {
      while (!interrupted) {
        const { done, value } = await reader.read()
        if (done) break
        const { thought, text } = chunkParts(value)
        if (!thought && !text) continue
        firstTokenAt ??= performance.now()
        chunks++
        // Reasoning is streamed inline (as with Transformers.js), separated from the answer by a rule.
        if (thought) thinking = true
        const sep = thinking && !thought && text ? '\n\n---\n\n' : ''
        if (sep) thinking = false
        output += sep + thought + text
        post({ type: 'token', requestId, text: sep + thought + text })
      }
    } catch (err) {
      if (!interrupted) throw err
    } finally {
      if (interrupted) await reader.cancel().catch(() => {})
      reader.releaseLock()
    }
    const end = performance.now()

    stage = 'read benchmark info'
    info = await conv.getBenchmarkInfo().catch(() => undefined)

    const first = firstTokenAt ?? end
    // Chunks aren't guaranteed to be one token each, so the token count comes from LiteRT-LM.
    const outputTokens = info?.lastDecodeTokenCount || chunks
    const decodeSec = (end - first) / 1000
    const metrics = {
      ttftMs: first - start,
      totalMs: end - start,
      // With a reused conversation this is only the new turn's tokens: that's what was prefilled.
      promptTokens: info?.lastPrefillTokenCount ?? 0,
      outputTokens,
      tokensPerSec: outputTokens > 1 && decodeSec > 0 ? (outputTokens - 1) / decodeSec : 0,
      interrupted,
      reusedContext: conversationId ? reused : undefined,
      runtimeInfo: info && {
        prefillTokens: info.lastPrefillTokenCount,
        prefillTokensPerSec: info.lastPrefillTokensPerSecond,
        decodeTokens: info.lastDecodeTokenCount,
        decodeTokensPerSec: info.lastDecodeTokensPerSecond,
        ttftMs: info.timeToFirstTokenInSecond * 1000,
      },
    }
    console.debug('[litert worker] generation metrics', metrics, { chunks })

    // An interrupted turn may leave a partial reply in the conversation; don't try to continue from it.
    // The UI sends back exactly the streamed text as the assistant turn, so that's what the next request must match.
    if (conversationId && !interrupted && output) {
      live = {
        id: conversationId,
        settingsKey,
        history: [...preface, message, { role: 'assistant', content: output }].map(messageKey),
        conv,
      }
      keep = true
    }
    post({ type: 'generate-done', requestId, metrics })
  } finally {
    conversation = null
    if (!keep) await conv.delete().catch(() => {})
  }
}

function reportError(scope: 'inspect' | 'load' | 'generate', err: unknown, requestId?: string) {
  console.error(`[litert worker] ${scope} failed during "${stage}":`, err)
  const message = err instanceof Error ? err.message : String(err)
  post({ type: 'error', scope, requestId, stage, message, stack: err instanceof Error ? err.stack : undefined })
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
        console.info('[litert worker] loading', req.modelId)
        await load(req.modelId)
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
        await generate(req.requestId, req.messages, req.settings, req.conversationId)
      } catch (err) {
        reportError('generate', err, req.requestId)
      }
      break
    case 'interrupt':
      interrupted = true
      conversation?.cancel()
      break
  }
})
