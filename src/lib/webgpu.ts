export type GpuInfo = {
  supported: boolean
  reason?: string
  vendor?: string
  architecture?: string
  description?: string
  shaderF16: boolean
  maxBufferSizeMB?: number
}

export async function detectWebGPU(): Promise<GpuInfo> {
  if (!('gpu' in navigator) || !navigator.gpu) {
    return { supported: false, shaderF16: false, reason: 'This browser does not expose WebGPU (navigator.gpu is missing).' }
  }
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
    if (!adapter) {
      return { supported: false, shaderF16: false, reason: 'No suitable GPU adapter was found. WebGPU may be disabled or blocklisted for this GPU.' }
    }
    const info = adapter.info
    return {
      supported: true,
      vendor: info?.vendor || undefined,
      architecture: info?.architecture || undefined,
      description: info?.description || undefined,
      shaderF16: adapter.features.has('shader-f16'),
      maxBufferSizeMB: Math.round(adapter.limits.maxBufferSize / (1024 * 1024)),
    }
  } catch (err) {
    return { supported: false, shaderF16: false, reason: err instanceof Error ? err.message : String(err) }
  }
}

export function gpuLabel(info: GpuInfo | null): string {
  if (!info?.supported) return 'No WebGPU'
  return [info.vendor, info.architecture].filter(Boolean).join(' · ') || info.description || 'Unknown GPU'
}
