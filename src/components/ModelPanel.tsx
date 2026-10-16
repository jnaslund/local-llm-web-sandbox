import { useEffect } from 'react'
import { MODELS, RUNTIMES, modelLabel, quantLabel, runtimeLabel, type Runtime } from '../config/models'
import { DEFAULT_PERF } from '../worker/protocol'
import { formatBytes, formatMs } from '../lib/format'
import { clearLiteRtCache } from '../lib/litertCache'
import { inspectionKey, inspectModel, loadModel } from '../lib/llm'
import { useSession } from '../state/session'
import { useSettings } from '../state/settings'

export function ModelPanel() {
  const runtime = useSettings((s) => s.runtime)
  const modelId = useSettings((s) => s.modelId)
  const perf = useSettings((s) => s.perf)
  const update = useSettings((s) => s.update)
  const { status, loadedModelId, loadError, progress, loadMetrics, inspections, gpu, generating } = useSession()

  useEffect(() => {
    MODELS.forEach((m) => inspectModel(runtime, m.id))
  }, [runtime])

  const loading = status === 'loading'
  const loadedRuntime = loadMetrics?.runtime ?? 'transformers'
  const sameModel = loadedModelId === modelId && loadedRuntime === runtime
  const perfChanged =
    runtime === 'transformers' && !!loadMetrics && JSON.stringify(loadMetrics.perf ?? DEFAULT_PERF) !== JSON.stringify(perf)
  const isLoaded = status === 'ready' && sameModel && !perfChanged
  const inspection = inspections[inspectionKey(runtime, modelId)]
  const anyLiteRtCached = MODELS.some((m) => inspections[inspectionKey('litert', m.id)]?.cached)
  const pct = progress && progress.total > 0 ? (progress.loaded / progress.total) * 100 : 0

  return (
    <section className="panel space-y-3">
      <div>
        <label className="label" htmlFor="runtime">
          Runtime
        </label>
        <select
          id="runtime"
          className="input"
          value={runtime}
          disabled={loading || generating}
          onChange={(e) => update({ runtime: e.target.value as Runtime })}
        >
          {RUNTIMES.map((r) => (
            <option key={r.id} value={r.id}>
              {r.label}
            </option>
          ))}
        </select>
        <p className="mt-1 text-xs text-zinc-500">{RUNTIMES.find((r) => r.id === runtime)?.description}</p>
      </div>

      <div>
        <label className="label" htmlFor="model">
          Model
        </label>
        <select
          id="model"
          className="input"
          value={modelId}
          disabled={loading || generating}
          onChange={(e) => update({ modelId: e.target.value })}
        >
          {MODELS.map((m) => {
            const info = inspections[inspectionKey(runtime, m.id)]
            const meta = info ? ` · ${formatBytes(info.totalBytes)}${info.cached ? ' · cached' : ''}` : ''
            return (
              <option key={m.id} value={m.id}>
                {m.label}
                {meta}
              </option>
            )
          })}
        </select>
        <p className="mt-1 text-xs text-zinc-500">
          {MODELS.find((m) => m.id === modelId)?.description} Weights: {quantLabel(runtime)}.
        </p>
      </div>

      {gpu?.supported && !gpu.shaderF16 && (
        <p className="rounded-lg bg-amber-50 p-2 text-xs text-amber-800 dark:bg-amber-950 dark:text-amber-300">
          Your GPU adapter does not report <code>shader-f16</code>. The {quantLabel(runtime)} model may fail to load or run slowly.
        </p>
      )}

      {!isLoaded && (
        <button className="btn-primary w-full" disabled={loading || generating} onClick={() => void loadModel(runtime, modelId, perf)}>
          {loading
            ? 'Loading…'
            : perfChanged && sameModel
              ? 'Reload with new performance settings'
              : loadedModelId
              ? `Switch to ${modelLabel(modelId)} · ${runtimeLabel(runtime, true)}`
              : inspection?.cached
                ? 'Load model (cached)'
                : `Download & load (${formatBytes(inspection?.totalBytes)})`}
        </button>
      )}

      {!isLoaded && generating && !loading && (
        <p className="text-xs text-zinc-500">Waiting for the current response to finish before reloading.</p>
      )}

      {loading && (
        <div className="space-y-1">
          <div className="h-2 overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
            <div
              className={`h-full bg-indigo-500 transition-all ${progress?.phase === 'initializing' ? 'animate-pulse' : ''}`}
              style={{ width: `${progress?.phase === 'initializing' ? 100 : pct}%` }}
            />
          </div>
          <p className="text-xs text-zinc-500">
            {progress?.phase === 'initializing'
              ? 'Initializing on GPU (compiling shaders)…'
              : progress
                ? `Downloading ${formatBytes(progress.loaded)} / ${formatBytes(progress.total)} (${pct.toFixed(0)}%)`
                : 'Preparing…'}
          </p>
        </div>
      )}

      {status === 'error' && loadError && (
        <div className="space-y-2 rounded-lg bg-red-50 p-2 text-xs text-red-800 dark:bg-red-950 dark:text-red-300">
          <p className="font-medium">Failed to load the model</p>
          <p className="font-mono break-words">{loadError}</p>
          <p>If this is an out-of-memory error, try the smaller E2B model or close other GPU-heavy tabs.</p>
        </div>
      )}

      {status === 'ready' && loadMetrics && (
        <div className="flex items-center gap-2 text-xs text-zinc-600 dark:text-zinc-400">
          <span className="inline-block size-2 rounded-full bg-emerald-500" />
          {modelLabel(loadMetrics.modelId)} · {runtimeLabel(loadMetrics.runtime, true)} ready in{' '}
          {formatMs(loadMetrics.totalMs)}
          {loadMetrics.fromCache && ' (from cache)'}
        </div>
      )}

      {status === 'ready' && loadMetrics?.warning && (
        <p className="rounded-lg bg-amber-50 p-2 text-xs text-amber-800 dark:bg-amber-950 dark:text-amber-300">
          {loadMetrics.warning}
        </p>
      )}

      {runtime === 'litert' && anyLiteRtCached && !loading && !(status === 'ready' && loadedRuntime === 'litert') && (
        <button
          className="text-xs text-zinc-500 underline hover:text-zinc-900 dark:hover:text-white"
          onClick={async () => {
            if (!confirm('Delete all cached LiteRT-LM model files from this browser?')) return
            await clearLiteRtCache()
            MODELS.forEach((m) => inspectModel('litert', m.id))
          }}
        >
          Delete cached LiteRT-LM models
        </button>
      )}
    </section>
  )
}
