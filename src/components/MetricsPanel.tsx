import { useState } from 'react'
import { modelLabel, quantLabel, runtimeLabel } from '../config/models'
import { runBenchmark } from '../lib/benchmark'
import { formatMs, formatProfile, formatRuntimeInfo, formatTps } from '../lib/format'
import { copyReport, downloadReport, sessionAverages } from '../lib/report'
import { gpuLabel } from '../lib/webgpu'
import { useActiveMessages } from '../state/chat'
import { useSession } from '../state/session'

export function MetricsPanel() {
  const { gpu, loadMetrics, loadedModelId, status, generating, benchmarking, benchmarks } = useSession()
  useActiveMessages() // re-render when chat metrics change
  const avg = sessionAverages()
  const [includeTranscript, setIncludeTranscript] = useState(false)
  const [copied, setCopied] = useState(false)

  return (
    <div className="space-y-5 text-sm">
      <Section title="Device">
        <Row k="GPU" v={gpuLabel(gpu)} />
        {gpu?.description && <Row k="Adapter" v={gpu.description} />}
        <Row k="shader-f16" v={gpu?.shaderF16 ? 'yes' : 'no'} />
        <Row k="Max buffer" v={gpu?.maxBufferSizeMB ? `${gpu.maxBufferSizeMB} MB` : '–'} />
        <Row k="CPU threads" v={String(navigator.hardwareConcurrency ?? '–')} />
      </Section>

      <Section title="Model load">
        {loadMetrics ? (
          <>
            <Row k="Runtime" v={runtimeLabel(loadMetrics.runtime)} />
            <Row k="Model" v={`${modelLabel(loadMetrics.modelId)} (${loadMetrics.dtype ?? quantLabel(loadMetrics.runtime)})`} />
            <Row k="Download" v={formatMs(loadMetrics.downloadMs)} />
            <Row k="Init / compile" v={formatMs(loadMetrics.initMs)} />
            <Row k="Total" v={formatMs(loadMetrics.totalMs)} />
            <Row k="From cache" v={loadMetrics.fromCache ? 'yes' : 'no'} />
            {loadMetrics.runtime !== 'litert' && (
            <Row
              k="Experiments"
              v={
                [
                  loadMetrics.perf?.int64OnGpu && 'int64 on GPU',
                  loadMetrics.perf?.disableValidation && 'no validation',
                  loadMetrics.perf?.embedOutputsOnGpu && 'embeddings on GPU',
                ]
                  .filter(Boolean)
                  .join(', ') || 'none'
              }
            />
            )}
            <Row
              k="Inference GPU"
              v={
                loadMetrics.adapter
                  ? [loadMetrics.adapter.vendor, loadMetrics.adapter.architecture, loadMetrics.adapter.description]
                      .filter(Boolean)
                      .join(' · ') || 'unknown'
                  : 'unknown'
              }
            />
          </>
        ) : (
          <p className="text-xs text-zinc-500">No model loaded yet.</p>
        )}
      </Section>

      <Section title="This chat">
        {avg ? (
          <>
            <Row k="Responses" v={String(avg.responses)} />
            <Row k="Avg speed" v={formatTps(avg.avgTokensPerSec)} />
            <Row k="Avg TTFT" v={formatMs(avg.avgTtftMs)} />
          </>
        ) : (
          <p className="text-xs text-zinc-500">No completed responses yet.</p>
        )}
      </Section>

      <Section title="Benchmark">
        <button
          className="btn-primary w-full"
          disabled={status !== 'ready' || generating}
          onClick={() => void runBenchmark()}
        >
          {benchmarking ? 'Running benchmark…' : 'Run benchmark'}
        </button>
        {status !== 'ready' && <p className="text-xs text-zinc-500">Load a model first (sidebar, top).</p>}
        {status === 'ready' && generating && !benchmarking && (
          <p className="text-xs text-zinc-500">Waiting for the current response to finish.</p>
        )}
        {benchmarks.length > 0 && (
          <table className="mt-2 w-full text-xs tabular-nums">
            <thead className="text-left text-zinc-500">
              <tr>
                <th className="font-medium">Run</th>
                <th className="font-medium">TTFT</th>
                <th className="font-medium">Speed</th>
              </tr>
            </thead>
            <tbody>
              {benchmarks.flatMap((run, i) =>
                run.results.flatMap((r) => [
                  <tr key={`${i}-${r.name}`} className="border-t border-zinc-200 dark:border-zinc-800">
                    <td className="py-1">
                      #{i + 1} {r.name}
                      <span className="block text-[10px] text-zinc-500">
                        {modelLabel(run.modelId)} · {runtimeLabel(run.runtime, true)}
                        {run.perf?.int64OnGpu && ' · int64'}
                        {run.perf?.disableValidation && ' · no-val'}
                        {run.perf?.embedOutputsOnGpu && ' · emb-gpu'}
                      </span>
                    </td>
                    <td>{formatMs(r.metrics.ttftMs)}</td>
                    <td>{formatTps(r.metrics.tokensPerSec)}</td>
                  </tr>,
                  (r.metrics.profile || r.metrics.runtimeInfo) && (
                    <tr key={`${i}-${r.name}-profile`}>
                      <td colSpan={3} className="pb-1 text-[10px] text-zinc-500">
                        {r.metrics.profile ? formatProfile(r.metrics.profile) : formatRuntimeInfo(r.metrics.runtimeInfo)}
                      </td>
                    </tr>
                  ),
                ]),
              )}
            </tbody>
          </table>
        )}
      </Section>

      <Section title="Export report">
        <label className="flex items-center gap-2 text-xs">
          <input
            type="checkbox"
            className="accent-indigo-600"
            checked={includeTranscript}
            onChange={(e) => setIncludeTranscript(e.target.checked)}
          />
          Include chat transcript
        </label>
        <div className="flex gap-2">
          <button className="btn-secondary flex-1" onClick={() => downloadReport(includeTranscript)}>
            Download JSON
          </button>
          <button
            className="btn-secondary flex-1"
            onClick={async () => {
              await copyReport(includeTranscript)
              setCopied(true)
              setTimeout(() => setCopied(false), 1500)
            }}
          >
            {copied ? 'Copied!' : 'Copy'}
          </button>
        </div>
        {!loadedModelId && <p className="text-xs text-zinc-500">Tip: load a model and run the benchmark first.</p>}
      </Section>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-1.5">
      <h3 className="label">{title}</h3>
      {children}
    </section>
  )
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-3 text-xs">
      <span className="text-zinc-500">{k}</span>
      <span className="truncate text-right tabular-nums" title={v}>
        {v}
      </span>
    </div>
  )
}
