import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { DEFAULT_MODEL_ID, DEFAULT_RUNTIME, findModel, type Runtime } from '../config/models'
import { DEFAULT_PERF, type GenerationSettings, type PerfOptions } from '../worker/protocol'

export type Settings = GenerationSettings & {
  runtime: Runtime
  modelId: string
  systemPrompt: string
  perf: PerfOptions
  /** LiteRT-LM: keep each chat's conversation (KV cache) between turns instead of re-prefilling the history. */
  litertStateful: boolean
}

export const DEFAULT_SETTINGS: Settings = {
  runtime: DEFAULT_RUNTIME,
  modelId: DEFAULT_MODEL_ID,
  systemPrompt: 'You are a helpful assistant. Answer concisely.',
  temperature: 0.7,
  topK: 64,
  maxNewTokens: 1024,
  enableThinking: false,
  perf: DEFAULT_PERF,
  litertStateful: true,
}

type SettingsStore = Settings & {
  update: (patch: Partial<Settings>) => void
  reset: () => void
}

export const useSettings = create<SettingsStore>()(
  persist(
    (set) => ({
      ...DEFAULT_SETTINGS,
      update: (patch) => set(patch),
      reset: () => set(DEFAULT_SETTINGS),
    }),
    {
      name: 'llm-proto.settings',
      version: 3,
      // v1 had a top-p setting, which Transformers.js ignores; replaced by top-k.
      // v2 stored the HF repo as modelId; v3 has runtime-neutral model ids plus `runtime`.
      migrate: (persisted) => {
        const { topP: _topP, ...rest } = persisted as Settings & { topP?: number }
        const modelId = findModel(rest.modelId ?? '')?.id ?? DEFAULT_MODEL_ID
        return { ...DEFAULT_SETTINGS, ...rest, modelId }
      },
      // Deep-merge `perf` so experiments added later get their defaults instead of `undefined`.
      merge: (persisted, current) => {
        const p = (persisted ?? {}) as Partial<Settings>
        return { ...current, ...p, perf: { ...DEFAULT_PERF, ...p.perf } }
      },
    },
  ),
)

export function generationSettings(s: Settings): GenerationSettings {
  return { temperature: s.temperature, topK: s.topK, maxNewTokens: s.maxNewTokens, enableThinking: s.enableThinking }
}
