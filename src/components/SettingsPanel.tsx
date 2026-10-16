import { useSettings } from '../state/settings'

export function SettingsPanel() {
  const s = useSettings()

  return (
    <div className="space-y-4">
      <div>
        <label className="label" htmlFor="system-prompt">
          System prompt
        </label>
        <textarea
          id="system-prompt"
          className="input min-h-24"
          value={s.systemPrompt}
          onChange={(e) => s.update({ systemPrompt: e.target.value })}
        />
      </div>

      <Slider
        label="Temperature"
        hint="0 = greedy (deterministic)"
        min={0}
        max={2}
        step={0.05}
        value={s.temperature}
        onChange={(temperature) => s.update({ temperature })}
      />
      <Slider
        label="Top-k"
        hint="Sample from the k most likely tokens (Gemma default: 64)"
        min={1}
        max={256}
        step={1}
        value={s.topK}
        onChange={(topK) => s.update({ topK })}
      />
      <Slider
        label="Max new tokens"
        min={64}
        max={4096}
        step={64}
        value={s.maxNewTokens}
        onChange={(maxNewTokens) => s.update({ maxNewTokens })}
      />

      <label className="flex items-center justify-between gap-2 text-sm">
        <span>
          Thinking mode
          <span className="block text-xs text-zinc-500">Lets the model reason before answering (slower).</span>
        </span>
        <input
          type="checkbox"
          className="size-4 accent-indigo-600"
          checked={s.enableThinking}
          onChange={(e) => s.update({ enableThinking: e.target.checked })}
        />
      </label>

      <fieldset
        className="space-y-3 rounded-lg border border-zinc-200 p-3 disabled:opacity-50 dark:border-zinc-800"
        disabled={s.runtime !== 'transformers'}
      >
        <legend className="label px-1">Performance experiments (Transformers.js)</legend>
        <Toggle
          label="int64 ops on GPU"
          hint="ep.webgpuexecutionprovider.enableInt64 — keeps token-id/mask/position ops off the CPU."
          checked={s.perf.int64OnGpu}
          onChange={(int64OnGpu) => s.update({ perf: { ...s.perf, int64OnGpu } })}
        />
        <Toggle
          label="Disable ORT WebGPU validation"
          hint="validationMode: disabled — less CPU work per GPU dispatch."
          checked={s.perf.disableValidation}
          onChange={(disableValidation) => s.update({ perf: { ...s.perf, disableValidation } })}
        />
        <Toggle
          label="Keep embeddings on GPU"
          hint="preferredOutputLocation gpu-buffer for embed_tokens — skips a GPU→CPU readback per token."
          checked={s.perf.embedOutputsOnGpu}
          onChange={(embedOutputsOnGpu) => s.update({ perf: { ...s.perf, embedOutputsOnGpu } })}
        />
        <p className="text-xs text-zinc-500">
          {s.runtime === 'transformers'
            ? 'Applied when the model is (re)loaded; the inference worker restarts.'
            : 'Not used by LiteRT-LM.'}
        </p>
      </fieldset>

      <fieldset
        className="space-y-3 rounded-lg border border-zinc-200 p-3 disabled:opacity-50 dark:border-zinc-800"
        disabled={s.runtime !== 'litert'}
      >
        <legend className="label px-1">LiteRT-LM</legend>
        <Toggle
          label="Keep conversation state"
          hint="Reuse the chat's KV cache between turns, so only the new message is prefilled. Rebuilt after regenerate, a settings change or switching chats."
          checked={s.litertStateful}
          onChange={(litertStateful) => s.update({ litertStateful })}
        />
      </fieldset>

      <button className="btn-secondary w-full" onClick={s.reset}>
        Reset to defaults
      </button>
    </div>
  )
}

function Toggle({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string
  hint: string
  checked: boolean
  onChange: (v: boolean) => void
}) {
  return (
    <label className="flex items-start justify-between gap-2 text-sm">
      <span>
        {label}
        <span className="block text-xs break-words text-zinc-500">{hint}</span>
      </span>
      <input
        type="checkbox"
        className="mt-1 size-4 shrink-0 accent-indigo-600"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
    </label>
  )
}

type SliderProps = {
  label: string
  hint?: string
  min: number
  max: number
  step: number
  value: number
  onChange: (v: number) => void
}

function Slider({ label, hint, min, max, step, value, onChange }: SliderProps) {
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="label">{label}</span>
        <span className="text-xs tabular-nums">{value}</span>
      </div>
      <input
        type="range"
        className="w-full accent-indigo-600"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      {hint && <p className="text-xs text-zinc-500">{hint}</p>}
    </div>
  )
}
