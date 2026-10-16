export type Runtime = 'transformers' | 'litert'

export const RUNTIMES: { id: Runtime; label: string; short: string; description: string }[] = [
  {
    id: 'transformers',
    label: 'Transformers.js',
    short: 'TJS',
    description: 'Transformers.js 4.3 on ONNX Runtime WebGPU (ONNX weights from the Hugging Face Hub).',
  },
  {
    id: 'litert',
    label: 'LiteRT-LM',
    short: 'LiteRT',
    description: "Google's LiteRT-LM Web (early preview), Gemma-specific WebGPU runtime with GPU sampling.",
  },
]

export const DEFAULT_RUNTIME: Runtime = 'transformers'

export type ModelOption = {
  id: string
  label: string
  description: string
  transformers: { repo: string; dtype: string }
  litert: { url: string }
}

export const DTYPE = 'q4f16'

const litertUrl = (size: string) =>
  `https://huggingface.co/litert-community/gemma-4-${size}-it-litert-lm/resolve/main/gemma-4-${size}-it-web.litertlm`

export const MODELS: ModelOption[] = [
  {
    id: 'gemma-4-e2b',
    label: 'Gemma 4 E2B',
    description: '~2.3B effective params. Fastest, smallest download.',
    transformers: { repo: 'onnx-community/gemma-4-E2B-it-ONNX', dtype: DTYPE },
    litert: { url: litertUrl('E2B') },
  },
  {
    id: 'gemma-4-e4b',
    label: 'Gemma 4 E4B',
    description: '~4B effective params. Better quality, ~2x download and memory.',
    transformers: { repo: 'onnx-community/gemma-4-E4B-it-ONNX', dtype: DTYPE },
    litert: { url: litertUrl('E4B') },
  },
]

export const DEFAULT_MODEL_ID = MODELS[0].id

/** Looks up a model by id; also accepts the HF repo ids that older settings and chats stored. */
export function findModel(id: string): ModelOption | undefined {
  return MODELS.find((m) => m.id === id || m.transformers.repo === id)
}

export function modelLabel(id: string): string {
  return findModel(id)?.label ?? id
}

export function runtimeLabel(runtime: Runtime | undefined, short = false): string {
  const r = RUNTIMES.find((r) => r.id === (runtime ?? 'transformers'))
  return (short ? r?.short : r?.label) ?? String(runtime)
}

/** Weight format shown in the UI and reports. */
export function quantLabel(runtime: Runtime | undefined): string {
  return runtime === 'litert' ? 'litertlm-web' : DTYPE
}
