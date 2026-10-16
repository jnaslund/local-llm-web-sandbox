// Cache Storage for LiteRT-LM model files. LiteRT-LM itself does a plain fetch() on every load, so we download the
// file once into Cache Storage and hand the engine a Blob read back from it (disk-backed, not held in JS memory).
// Works on the main thread and in workers; it must not import @litert-lm/core.

const CACHE_NAME = 'litert-models-v1'

async function openCache(): Promise<Cache | null> {
  try {
    return 'caches' in globalThis ? await caches.open(CACHE_NAME) : null
  } catch {
    return null
  }
}

export async function inspectLiteRtModel(url: string): Promise<{ cached: boolean; totalBytes: number | null }> {
  const hit = await (await openCache())?.match(url)
  if (hit) return { cached: true, totalBytes: Number(hit.headers.get('content-length')) || null }
  try {
    const res = await fetch(url, { method: 'HEAD' })
    return { cached: false, totalBytes: (res.ok && Number(res.headers.get('content-length'))) || null }
  } catch {
    return { cached: false, totalBytes: null }
  }
}

export async function clearLiteRtCache(): Promise<void> {
  if ('caches' in globalThis) await caches.delete(CACHE_NAME)
}

export type ModelSource = {
  model: Blob | ReadableStream<Uint8Array>
  fromCache: boolean
  /** Set when the file could not be cached (it is then streamed straight from the network). */
  warning?: string
}

async function fetchWithProgress(url: string, onProgress: (loaded: number, total: number) => void) {
  const res = await fetch(url)
  if (!res.ok || !res.body) throw new Error(`Download failed: HTTP ${res.status} for ${url}`)
  const total = Number(res.headers.get('content-length')) || 0
  let loaded = 0
  const counter = new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      loaded += chunk.byteLength
      onProgress(loaded, total)
      controller.enqueue(chunk)
    },
  })
  return { body: res.body.pipeThrough(counter), headers: res.headers, total }
}

function errorText(err: unknown): string {
  return err instanceof Error ? `${err.name}: ${err.message}` : String(err)
}

/** The model file from Cache Storage, downloading and caching it first if needed. */
export async function openLiteRtModel(
  url: string,
  onProgress: (loaded: number, total: number) => void,
): Promise<ModelSource> {
  const cache = await openCache()
  const hit = await cache?.match(url)
  if (hit) return { model: await hit.blob(), fromCache: true }

  const download = await fetchWithProgress(url, onProgress)
  if (!cache) {
    return { model: download.body, fromCache: false, warning: 'Cache Storage is unavailable; the model is not cached.' }
  }

  // Skip caching up front when it clearly won't fit, rather than downloading twice.
  const estimate = await navigator.storage?.estimate?.().catch(() => undefined)
  const free = estimate?.quota != null ? estimate.quota - (estimate.usage ?? 0) : Infinity
  if (download.total > free) {
    const warning = `Not enough storage quota to cache the model (${(free / 1024 ** 3).toFixed(1)} GB free); it is not cached.`
    return { model: download.body, fromCache: false, warning }
  }

  try {
    await cache.put(url, new Response(download.body, { headers: download.headers }))
    const stored = await cache.match(url)
    if (!stored) throw new Error('The cached model could not be read back.')
    return { model: await stored.blob(), fromCache: false }
  } catch (err) {
    // Typically QuotaExceededError partway through. Download again and stream it into the engine uncached.
    await cache.delete(url).catch(() => {})
    const retry = await fetchWithProgress(url, onProgress)
    return { model: retry.body, fromCache: false, warning: `Could not cache the model (${errorText(err)}); it is not cached.` }
  }
}
