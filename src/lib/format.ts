import type { DecodeProfile, LiteRtBenchmarkInfo } from '../worker/protocol'

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null || !Number.isFinite(bytes)) return '?'
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(0)} KB`
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(0)} MB`
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`
}

export function formatMs(ms: number | null | undefined): string {
  if (ms == null || !Number.isFinite(ms)) return '–'
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`
}

const SESSION_LABELS: Record<string, string> = {
  decoder_model_merged: 'decoder',
  embed_tokens: 'embed',
  vision_encoder: 'vision',
  audio_encoder: 'audio',
}

/** e.g. "45.2 ms/tok · decoder 38.1 · embed 3.0 · other 4.1" */
export function formatProfile(p: DecodeProfile | undefined): string {
  if (!p) return '–'
  const parts = Object.entries(p.sessionMsPerStep)
    .sort((a, b) => b[1] - a[1])
    .map(([name, ms]) => `${SESSION_LABELS[name] ?? name} ${ms.toFixed(1)}`)
  return [`${p.msPerStep.toFixed(1)} ms/tok`, ...parts, `other ${p.otherMsPerStep.toFixed(1)}`].join(' · ')
}

export function formatTps(tps: number | null | undefined): string {
  if (tps == null || !Number.isFinite(tps)) return '–'
  return `${tps.toFixed(1)} tok/s`
}

/** e.g. "LiteRT: prefill 28 tok @ 450 tok/s · decode 255 tok @ 60.1 tok/s · TTFT 80 ms" */
export function formatRuntimeInfo(i: LiteRtBenchmarkInfo | undefined): string {
  if (!i) return '–'
  return [
    `LiteRT: prefill ${i.prefillTokens} tok @ ${i.prefillTokensPerSec.toFixed(0)} tok/s`,
    `decode ${i.decodeTokens} tok @ ${formatTps(i.decodeTokensPerSec)}`,
    `TTFT ${formatMs(i.ttftMs)}`,
  ].join(' · ')
}
