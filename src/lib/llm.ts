import { findModel, type Runtime } from '../config/models'
import { useSession } from '../state/session'
import { useSettings } from '../state/settings'
import type {
  GenerationMetrics,
  GenerationSettings,
  PerfOptions,
  WireMessage,
  WorkerRequest,
  WorkerResponse,
} from '../worker/protocol'
import { inspectLiteRtModel } from './litertCache'

type PendingGeneration = {
  onToken: (text: string, thought: boolean) => void
  resolve: (metrics: GenerationMetrics) => void
  reject: (err: Error) => void
}

const pending = new Map<string, PendingGeneration>()

// Each runtime has its own worker (and bundle chunk). Only one exists at a time, so switching runtime frees the
// other runtime's GPU memory. The `new Worker(new URL(…))` calls must stay literal for Vite to bundle them.
function spawnWorker(runtime: Runtime): Worker {
  return runtime === 'litert'
    ? new Worker(new URL('../worker/litert.worker.ts', import.meta.url), { type: 'module' })
    : new Worker(new URL('../worker/llm.worker.ts', import.meta.url), { type: 'module' })
}

function createWorker(runtime: Runtime): Worker {
  const w = spawnWorker(runtime)
  w.addEventListener('message', onMessage)
  w.addEventListener('error', onError)
  return w
}

let workerRuntime: Runtime = useSettings.getState().runtime
let worker = createWorker(workerRuntime)
/** Runtime + perf options the current worker's model/WebGPU context was created with (null until the first load). */
let workerKey: string | null = null

// A fresh worker has no model loaded, so make sure the UI doesn't claim otherwise. This matters after a Vite hot
// update re-runs this module: the stores survive, but the old worker and its pending requests are gone.
useSession.getState().set({
  status: 'idle',
  loadedModelId: null,
  loadMetrics: null,
  progress: null,
  generating: false,
  benchmarking: false,
})

function send(req: WorkerRequest, transfer: Transferable[] = []) {
  worker.postMessage(req, transfer)
}

function rejectAll(message: string) {
  for (const p of pending.values()) p.reject(new Error(message))
  pending.clear()
}

function onMessage(event: MessageEvent<WorkerResponse>) {
  const msg = event.data
  const session = useSession.getState()
  switch (msg.type) {
    case 'inspect-result':
      setInspection(msg.runtime, msg.modelId, msg.cached, msg.totalBytes)
      break
    case 'load-progress':
      session.set({ progress: { phase: msg.phase, loaded: msg.loaded, total: msg.total, files: msg.files } })
      break
    case 'load-done':
      session.set({ status: 'ready', loadedModelId: msg.metrics.modelId, loadMetrics: msg.metrics, progress: null })
      inspectModel(workerRuntime, msg.metrics.modelId)
      break
    case 'unloaded':
      session.set({ status: 'idle', loadedModelId: null })
      break
    case 'token':
      pending.get(msg.requestId)?.onToken(msg.text, !!msg.thought)
      break
    case 'generate-done':
      pending.get(msg.requestId)?.resolve(msg.metrics)
      pending.delete(msg.requestId)
      break
    case 'error': {
      const detail = `${msg.message} (during: ${msg.stage})`
      console.error(`[llm] ${msg.scope} failed during "${msg.stage}": ${msg.message}`, msg.stack ? `\n${msg.stack}` : '')
      if (msg.scope === 'load') {
        session.set({ status: 'error', loadError: detail, loadedModelId: null, progress: null })
      } else if (msg.scope === 'generate' && msg.requestId) {
        pending.get(msg.requestId)?.reject(new Error(detail))
        pending.delete(msg.requestId)
      }
      break
    }
  }
}

function onError(event: ErrorEvent) {
  const message = event.message || 'The inference worker crashed.'
  useSession.getState().set({ status: 'error', loadError: message, loadedModelId: null, generating: false })
  rejectAll(message)
}

function setInspection(runtime: Runtime, modelId: string, cached: boolean, totalBytes: number | null) {
  const session = useSession.getState()
  session.set({ inspections: { ...session.inspections, [inspectionKey(runtime, modelId)]: { cached, totalBytes } } })
}

export function inspectionKey(runtime: Runtime, modelId: string): string {
  return `${runtime}:${modelId}`
}

/**
 * Cached state + size of a model for a runtime. LiteRT-LM files are checked on this thread (Cache Storage + HEAD);
 * Transformers.js needs its library, so it asks the Transformers.js worker, or a short-lived one if the active worker
 * is LiteRT-LM's.
 */
export function inspectModel(runtime: Runtime, modelId: string) {
  if (runtime === 'litert') {
    const url = findModel(modelId)?.litert.url
    if (url) void inspectLiteRtModel(url).then((r) => setInspection(runtime, modelId, r.cached, r.totalBytes))
  } else if (workerRuntime === runtime) {
    send({ type: 'inspect', modelId })
  } else {
    const w = spawnWorker(runtime)
    const done = () => w.terminate()
    w.addEventListener('message', (event: MessageEvent<WorkerResponse>) => {
      const msg = event.data
      if (msg.type === 'inspect-result') setInspection(msg.runtime, msg.modelId, msg.cached, msg.totalBytes)
      if (msg.type === 'inspect-result' || msg.type === 'error') done()
    })
    w.addEventListener('error', done)
    w.postMessage({ type: 'inspect', modelId } satisfies WorkerRequest)
  }
}

/**
 * Release the old worker's model (ORT sessions / LiteRT engine destroy their GPU buffers) before terminating it.
 * terminate() alone leaves freeing the VRAM to the browser, so it can overlap the next model's load and push a 10 GB
 * GPU into paging (seen as Transformers.js E4B at 2–17 tok/s right after LiteRT E4B).
 */
function retireWorker(w: Worker): Promise<void> {
  w.removeEventListener('message', onMessage)
  w.removeEventListener('error', onError)
  return new Promise((resolve) => {
    const done = () => {
      clearTimeout(timer)
      w.terminate()
      resolve()
    }
    const timer = setTimeout(done, 5000)
    w.addEventListener('message', (e: MessageEvent<WorkerResponse>) => e.data.type === 'unloaded' && done())
    w.postMessage({ type: 'unload' } satisfies WorkerRequest)
  })
}

let loadSeq = 0

export async function loadModel(runtime: Runtime, modelId: string, perf: PerfOptions) {
  const seq = ++loadSeq
  useSession.getState().set({ status: 'loading', loadError: null, progress: null, loadMetrics: null, loadedModelId: null })
  // A different runtime needs its own worker; WebGPU EP options are baked into ORT's device context on first use.
  // Either change needs a fresh worker.
  const key = JSON.stringify(runtime === 'transformers' ? { runtime, perf } : { runtime })
  if (runtime !== workerRuntime || (workerKey !== null && workerKey !== key)) {
    rejectAll('The inference worker was restarted.')
    const old = worker
    workerRuntime = runtime
    worker = createWorker(runtime)
    // The new worker exists right away (inspections can use it), but loads only after the old one has let go.
    await retireWorker(old)
    if (seq !== loadSeq) return
  }
  workerKey = key
  send({ type: 'load', modelId, perf })
}

export function unloadModel() {
  send({ type: 'unload' })
}

export function interrupt() {
  send({ type: 'interrupt' })
}

export function generate(
  messages: WireMessage[],
  settings: GenerationSettings,
  onToken: (text: string, thought: boolean) => void = () => {},
  /** LiteRT-LM: keep and reuse the conversation (KV cache) under this id. Ignored by Transformers.js. */
  conversationId?: string,
): Promise<GenerationMetrics> {
  const requestId = crypto.randomUUID()
  console.debug(
    '[llm] generate',
    { settings, messages: messages.map((m) => ({ role: m.role, text: m.text, image: !!m.image, audio: m.audio?.length, prior: m.priorAttachments })) },
  )
  return new Promise((resolve, reject) => {
    pending.set(requestId, { onToken, resolve, reject })
    // Audio buffers are copied (not transferred) so the chat history keeps its copy for regenerate.
    send({ type: 'generate', requestId, messages, settings, conversationId })
  })
}

if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    worker.terminate()
    rejectAll('The inference worker was reloaded after a code change. Load the model again.')
  })
}
