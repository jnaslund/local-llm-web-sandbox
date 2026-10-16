import { getActiveSession } from '../state/chat'
import { useSession } from '../state/session'
import { useSettings } from '../state/settings'

export function sessionAverages() {
  const metrics = getActiveSession().messages.flatMap((m) => (m.metrics && !m.metrics.interrupted ? [m.metrics] : []))
  if (!metrics.length) return null
  const avg = (f: (m: (typeof metrics)[number]) => number) => metrics.reduce((s, m) => s + f(m), 0) / metrics.length
  return {
    responses: metrics.length,
    avgTokensPerSec: avg((m) => m.tokensPerSec),
    avgTtftMs: avg((m) => m.ttftMs),
  }
}

export function buildReport(includeTranscript: boolean) {
  const session = useSession.getState()
  const { update: _u, reset: _r, ...settings } = useSettings.getState()
  const nav = navigator as Navigator & { deviceMemory?: number }
  return {
    generatedAt: new Date().toISOString(),
    app: 'local-llm-web-prototype',
    device: {
      userAgent: navigator.userAgent,
      platform: navigator.platform,
      hardwareConcurrency: navigator.hardwareConcurrency,
      deviceMemoryGB: nav.deviceMemory ?? null,
      screen: `${screen.width}x${screen.height}@${devicePixelRatio}`,
      gpu: session.gpu,
    },
    runtime: session.loadMetrics?.runtime ?? null,
    model: session.loadedModelId,
    chat: (({ title, createdAt, modelId }) => ({ title, createdAt: new Date(createdAt).toISOString(), modelId }))(
      getActiveSession(),
    ),
    load: session.loadMetrics,
    settings,
    sessionAverages: sessionAverages(),
    benchmarks: session.benchmarks,
    transcript: includeTranscript
      ? getActiveSession().messages.map((m) => ({
          role: m.role,
          text: m.text,
          thinking: m.thinking,
          attachments: m.attachments?.map((a) => a.kind),
          metrics: m.metrics,
          error: m.error,
        }))
      : undefined,
  }
}

export function downloadReport(includeTranscript: boolean) {
  const report = buildReport(includeTranscript)
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `llm-report-${new Date().toISOString().replace(/[:.]/g, '-')}.json`
  a.click()
  URL.revokeObjectURL(url)
}

export async function copyReport(includeTranscript: boolean) {
  await navigator.clipboard.writeText(JSON.stringify(buildReport(includeTranscript), null, 2))
}
