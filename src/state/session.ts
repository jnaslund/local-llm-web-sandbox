import { create } from 'zustand'
import type { Runtime } from '../config/models'
import type { GpuInfo } from '../lib/webgpu'
import type { FileProgress, GenerationMetrics, LoadMetrics, PerfOptions } from '../worker/protocol'

export type ModelStatus = 'idle' | 'loading' | 'ready' | 'error'

export type LoadProgress = {
  phase: 'downloading' | 'initializing'
  loaded: number
  total: number
  files: Record<string, FileProgress>
}

export type ModelInspection = { cached: boolean; totalBytes: number | null }

export type BenchmarkResult = {
  name: string
  metrics: GenerationMetrics
}

export type BenchmarkRun = {
  runtime: Runtime
  modelId: string
  perf?: PerfOptions
  startedAt: string
  results: BenchmarkResult[]
}

type SessionStore = {
  gpu: GpuInfo | null
  status: ModelStatus
  loadedModelId: string | null
  loadError: string | null
  progress: LoadProgress | null
  loadMetrics: LoadMetrics | null
  /** Keyed by `inspectionKey(runtime, modelId)`. */
  inspections: Record<string, ModelInspection>
  generating: boolean
  benchmarking: boolean
  benchmarks: BenchmarkRun[]
  set: (patch: Partial<SessionStore>) => void
}

export const useSession = create<SessionStore>()((set) => ({
  gpu: null,
  status: 'idle',
  loadedModelId: null,
  loadError: null,
  progress: null,
  loadMetrics: null,
  inspections: {},
  generating: false,
  benchmarking: false,
  benchmarks: [],
  set: (patch) => set(patch),
}))
