import type { GpuInfo } from '../lib/webgpu'

export function UnsupportedScreen({ gpu }: { gpu: GpuInfo }) {
  return (
    <div className="flex h-full items-center justify-center p-4">
      <div className="panel max-w-lg space-y-3">
        <h1 className="text-lg font-semibold">WebGPU is not available</h1>
        <p className="text-sm text-zinc-600 dark:text-zinc-400">
          This prototype runs the language model entirely in your browser using WebGPU, which this browser or device
          does not currently support.
        </p>
        {gpu.reason && (
          <p className="rounded-lg bg-zinc-100 p-2 font-mono text-xs dark:bg-zinc-800">{gpu.reason}</p>
        )}
        <ul className="list-disc space-y-1 pl-5 text-sm">
          <li>Use a recent Chrome or Edge (113+) on desktop, or Safari 26+.</li>
          <li>Make sure hardware acceleration is enabled in the browser settings.</li>
          <li>The page must be served over HTTPS (or localhost).</li>
        </ul>
      </div>
    </div>
  )
}
