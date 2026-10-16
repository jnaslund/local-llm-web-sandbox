// Typed message protocol between the UI thread and the inference workers. Both runtimes' workers
// (llm.worker.ts = Transformers.js, litert.worker.ts = LiteRT-LM) speak it.

import type { Runtime } from '../config/models'

export type GenerationSettings = {
  temperature: number
  /** Sample from the K most likely tokens. (Transformers.js 4.3 ignores top_p, so it isn't offered.) */
  topK: number
  maxNewTokens: number
  enableThinking: boolean
}

export type WireMessage = {
  role: 'system' | 'user' | 'assistant'
  text: string
  image?: Blob
  /** 16 kHz mono PCM. */
  audio?: Float32Array
  /** Attachments from earlier turns, described as text (only the latest turn's media is sent). */
  priorAttachments?: ('image' | 'audio')[]
}

export type GenerationMetrics = {
  /** Time from request to first generated token (includes prompt processing / prefill). */
  ttftMs: number
  totalMs: number
  promptTokens: number
  outputTokens: number
  /** Decode speed, excluding the first token. */
  tokensPerSec: number
  interrupted: boolean
  profile?: DecodeProfile
  /**
   * LiteRT-LM stateful mode: true when the request continued the live conversation (KV cache kept, only the new
   * message prefilled), false when it rebuilt one from the full history. Undefined for stateless requests.
   */
  reusedContext?: boolean
  /** LiteRT-LM's own numbers for this request (`Conversation.getBenchmarkInfo()`). */
  runtimeInfo?: LiteRtBenchmarkInfo
}

export type LiteRtBenchmarkInfo = {
  prefillTokens: number
  prefillTokensPerSec: number
  decodeTokens: number
  decodeTokensPerSec: number
  ttftMs: number
}

/** Where the time goes per generated token, measured over the decode steps after the first token. */
export type DecodeProfile = {
  steps: number
  msPerStep: number
  /** Average wall time per step inside each ONNX session's run() (GPU work + dispatch + output readback). */
  sessionMsPerStep: Record<string, number>
  /** Everything outside the sessions: logits processing, sampling, streaming, JS overhead. */
  otherMsPerStep: number
}

/**
 * ONNX Runtime WebGPU options to A/B test. They configure the WebGPU context, which is created once per worker,
 * so changing them restarts the worker.
 */
export type PerfOptions = {
  /** Register int64 kernels on WebGPU so int64 ops (token ids, masks, positions) don't fall back to the CPU. */
  int64OnGpu: boolean
  /** Skip ORT's own WebGPU validation (`validationMode: 'disabled'`, equivalent to 'wgpuOnly' in browsers). */
  disableValidation: boolean
  /**
   * Keep embed_tokens outputs on the GPU during decoding so they feed the decoder directly, instead of a
   * GPU→CPU readback and re-upload every token.
   */
  embedOutputsOnGpu: boolean
}

// Measured on an RTX 3080 (see AGENTS.md): embedOutputsOnGpu ≈ +20% decode speed; int64OnGpu neutral-to-slightly
// positive; disableValidation within noise, so it stays off.
export const DEFAULT_PERF: PerfOptions = { int64OnGpu: true, disableValidation: false, embedOutputsOnGpu: true }

export type AdapterInfo = { vendor?: string; architecture?: string; description?: string }

export type LoadMetrics = {
  /** Missing on metrics from before the runtime switch existed (= Transformers.js). */
  runtime?: Runtime
  modelId: string
  dtype: string
  downloadMs: number
  initMs: number
  totalMs: number
  fromCache: boolean
  /** The GPU adapter ONNX Runtime actually created its device on (may differ from the UI's own probe). */
  adapter?: AdapterInfo
  /** Transformers.js only. */
  perf?: PerfOptions
  /** Non-fatal problem during load, e.g. the model could not be cached. */
  warning?: string
}

export type FileProgress = { loaded: number; total: number }

export type WorkerRequest =
  | { type: 'inspect'; modelId: string }
  | { type: 'load'; modelId: string; perf: PerfOptions }
  | { type: 'unload' }
  | {
      type: 'generate'
      requestId: string
      messages: WireMessage[]
      settings: GenerationSettings
      /** LiteRT-LM only: keep the conversation (KV cache) under this id and reuse it while the history matches. */
      conversationId?: string
    }
  | { type: 'interrupt' }

export type WorkerResponse =
  | { type: 'inspect-result'; runtime: Runtime; modelId: string; cached: boolean; totalBytes: number | null }
  | {
      type: 'load-progress'
      phase: 'downloading' | 'initializing'
      loaded: number
      total: number
      files: Record<string, FileProgress>
    }
  | { type: 'load-done'; metrics: LoadMetrics }
  | { type: 'unloaded' }
  /** `thought`: the text is the model's reasoning (thinking mode), not part of the answer. */
  | { type: 'token'; requestId: string; text: string; thought?: boolean }
  | { type: 'generate-done'; requestId: string; metrics: GenerationMetrics }
  | {
      type: 'error'
      requestId?: string
      scope: 'inspect' | 'load' | 'generate'
      /** The step that was running when the error happened, e.g. "apply chat template". */
      stage: string
      message: string
      stack?: string
    }
