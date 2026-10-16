# Spec: LiteRT-LM runtime for side-by-side benchmarking

Status: **implemented** (phase 1 + phase 2 stateful conversations; multimodal not possible in 0.18, see AGENTS.md) · Written 2026-10-16 · Read [AGENTS.md](../../AGENTS.md) first (architecture, gotchas,
testing recipes, perf history).

## 1. Why

The Transformers.js path is tuned about as far as it can go. On an RTX 3080 (Chrome 155, Gemma 4 E2B q4f16) it decodes
at **~28 ms/token (35–36 tok/s)**, and ~22 ms of that is the ONNX Runtime WebGPU decoder, which no option moved
(details in AGENTS.md → Performance findings). The GPU's bandwidth floor is ~3 ms/token, so a Gemma-specific runtime
could plausibly be 2–4× faster.

**Goal:** add Google's **LiteRT-LM Web** (`@litert-lm/core`) as a second, switchable runtime. Run the existing
benchmark on both runtimes with the same model size and compare speed, load time, and output quality on the same
machine.

**Non-goals:** removing or refactoring the Transformers.js path beyond what a provider switch needs, production
polish, tool calling, conversation cloning.

## 2. Success criteria

1. Users can pick **Runtime: Transformers.js | LiteRT-LM** and model **E2B | E4B** and chat with either. Streaming,
   Stop, multi-turn, sessions, and settings all work.
2. **Run benchmark** works on LiteRT-LM and produces the same metrics shape (TTFT, tok/s, token counts). Runs and
   exports are tagged with the runtime.
3. The model downloads once and is cached; later loads don't re-download.
4. Switching runtimes frees the other runtime's GPU memory.
5. Findings (tok/s for both runtimes, E2B and ideally E4B) are recorded in AGENTS.md.

## 3. What we know about LiteRT-LM Web (researched 2026-10-16)

| Item | Value |
|---|---|
| Package | `@litert-lm/core` **0.18.0** (latest; nightlies at 0.19.0-nightly). One dep: `@litertjs/wasm-utils`. |
| Status | "Early preview". README says text-in/text-out on WebGPU, but 0.18.0 types include `ImageContentPart`/`AudioContentPart` and `SessionConfig.visionModalityEnabled`/`audioModalityEnabled` → **multimodal may work; verify** (§6.6). |
| Models | Only the web builds are supported: `gemma-4-E2B-it-web.litertlm` (**2.0 GB**) and `gemma-4-E4B-it-web.litertlm` (**3.0 GB**). |
| Model URLs | `https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-web.litertlm`, same pattern for `E4B`. Public, Apache-2.0, not gated. |
| Wasm | ~109 MB of variants (`litertlm_wasm[_compat][_asyncify]_internal.wasm`), loaded by default from `https://cdn.jsdelivr.net/npm/@litert-lm/core@0.18.0/wasm` (`LiteRtLm.DEFAULT_WASM_PATH`). It picks a variant via feature detection. `loadLiteRtLm(path)` overrides the path. |
| Worker | No `document`/`window` use found in `dist/*.js`; it uses `fetch` and WebGPU, both available in workers → run it in a worker. |

API surface (from `dist/*.d.ts`):

```ts
import { Engine, Backend, SamplerType, type EngineSettings, type ConversationConfig, type Message } from '@litert-lm/core'

const engine = await Engine.create({
  model: url | Blob | ReadableStream<Uint8Array>,
  backend?: Backend,                 // default Backend.GPU_ARTISAN (streams weights; other backends load fully into wasm memory)
  mainExecutorSettings?: { maxNumTokens?: number /* context */, samplerBackend?: Backend, backendConfig?, advancedSettings? },
  benchmarkEnabled?: boolean,
})
const conv = await engine.createConversation({
  preface?: { messages?: Message[]; tools?; extra_context?: Record<string, JsonValue> }, // extra_context.enable_thinking
  sessionConfig?: {
    samplerParams?: { type?: SamplerType /* TOP_K | TOP_P | GREEDY */, k?, p?, temperature?, seed? },
    maxOutputTokens?: number, visionModalityEnabled?: boolean, audioModalityEnabled?: boolean, samplerBackend?: Backend,
  },
  prefillPrefaceOnInit?: boolean,
})
conv.sendMessage(msg | msg[]): Promise<Message>
conv.sendMessageStreaming(msg | msg[]): ReadableStream<Message>   // chunk.content[i] = { type: 'text', text }
conv.cancel()                       // breaking out of `for await` also cancels
conv.getBenchmarkInfo(): Promise<{ lastPrefillTokensPerSecond, lastPrefillTokenCount, lastDecodeTokensPerSecond,
                                    lastDecodeTokenCount, timeToFirstTokenInSecond }>
conv.getTokenCount(), conv.getHistory(), conv.clone(), conv.delete()
await engine.delete()               // release GPU/wasm resources
// Message = { role: string; content?: string | ContentPart[]; channels?: Record<string,string>; ... }
```

Notable differences from Transformers.js:
- **Sampling can run on the GPU** (`samplerBackend`). This avoids the per-token logits readback, a known cost in our
  Transformers.js profile. Note which sampler backend is used in results.
- **No model caching.** `Engine.create` does a plain `fetch(url)` every time. Non-streaming backends also buffer the
  whole file into wasm memory (`loadModelToVfs`, TODO b/477709280). We must cache the file ourselves (§6.3).
- **Stateful conversations** (KV cache kept between turns). Our app is stateless per request; see §6.4.
- The top-p sampler **is** real here (`SamplerType.TOP_P`), unlike Transformers.js.

## 4. Baseline to beat (Transformers.js, RTX 3080, Chrome 155, E2B)

Defaults `int64OnGpu` + `embedOutputsOnGpu`, long benchmark runs: **27.6–28.4 ms/token, 35–36 tok/s**, TTFT ~80–90 ms
on a 28-token prompt, model load (cached) ~5 s. Edge is ~30% slower than Chrome on the decoder; compare in the same
browser.

## 5. Design

### 5.1 Runtime as a provider, one worker per runtime

- Add `runtime: 'transformers' | 'litert'` to `Settings` (persisted; bump version + migrate; deep-merge like `perf`).
- Keep `src/worker/llm.worker.ts` as the Transformers.js worker. Add **`src/worker/litert.worker.ts`**, which speaks
  the **same `protocol.ts` messages** (`inspect`, `load`, `unload`, `generate`, `interrupt` → `inspect-result`,
  `load-progress`, `load-done`, `token`, `generate-done`, `error`). Add runtime-neutral fields as needed rather than
  forking the protocol.
- `src/lib/llm.ts` owns one worker at a time and creates the right one for `settings.runtime`. Switching runtime
  terminates the current worker (frees VRAM), the same way the perf-options restart works today (`workerPerf` key →
  include runtime in that key). Keep the HMR reset + `import.meta.hot.dispose` behaviour.
- Import `@litert-lm/core` **only inside `litert.worker.ts`** so it lands in that worker's chunk and the
  Transformers.js path doesn't pay for it.

### 5.2 Models per runtime

Extend `src/config/models.ts` so each entry has a runtime-specific source:

```ts
{ id: 'gemma-4-e2b', label: 'Gemma 4 E2B',
  transformers: { repo: 'onnx-community/gemma-4-E2B-it-ONNX', dtype: 'q4f16' },
  litert: { url: '.../gemma-4-E2B-it-web.litertlm' } }
```

Existing persisted `modelId` values are HF repo ids; migrate them. Report/benchmark records should carry
`{ runtime, modelId }`.

### 5.3 UI

- ModelPanel: a runtime select above the model select. Show size/cached state per runtime/model. The load button
  logic is unchanged ("Switch…" when runtime or model differs from the loaded one).
- Hide or disable controls that don't apply to the active runtime. Performance experiments are Transformers.js-only.
  Image/audio buttons depend on §6.6.
- Metrics: show the runtime in Model load and in benchmark rows (e.g. `Gemma 4 E2B · LiteRT`). Show the LiteRT
  `getBenchmarkInfo()` numbers next to ours.

## 6. Implementation details

### 6.1 Load

1. `loadLiteRtLm(DEFAULT_WASM_PATH)` (or `getOrLoadGlobalLiteRtLm()`). Use the CDN default for the prototype. Note
   in the README that self-hosting the ~109 MB wasm is a later option.
2. Get the model as a Blob/stream from our cache (§6.3), reporting `load-progress` while downloading.
3. `Engine.create({ model, mainExecutorSettings: { maxNumTokens: 8192 }, benchmarkEnabled: true })` with the default
   backend (`GPU_ARTISAN`). Record `downloadMs` / `initMs` the same way as the Transformers.js worker.
4. `load-done` with `LoadMetrics` (+ `runtime`). Adapter info: read `navigator.gpu.requestAdapter()` info in the
   worker. Check whether LiteRT exposes its device; `setupDefaultWebGpuDevice()` exists.

### 6.2 Generate

Per request (stateless, see §6.4):

```ts
const conv = await engine.createConversation({
  preface: { messages: [system?, ...history.slice(0, -1)], extra_context: { enable_thinking } },
  sessionConfig: { samplerParams, maxOutputTokens: settings.maxNewTokens },
})
const stream = conv.sendMessageStreaming(lastUserMessage)
for await (const chunk of stream) { /* post token text; record first-chunk time */ if (interrupted) break }
const info = await conv.getBenchmarkInfo()
await conv.delete()
```

- Sampler mapping: `temperature === 0` → `{ type: GREEDY }`, else `{ type: TOP_K, k: topK, temperature }`.
  Optionally expose top-p for this runtime only.
- `interrupt` → `conv.cancel()` (or break the loop). Report `interrupted: true`.
- Metrics: our wall-clock TTFT/total like today. Take token counts and decode tok/s from `getBenchmarkInfo()`, since
  chunks are not guaranteed to be one token each. Set `promptTokens = lastPrefillTokenCount` and
  `outputTokens = lastDecodeTokenCount`. `profile` (per-session breakdown) doesn't apply; leave it undefined. Consider
  adding an optional `runtimeInfo` field to `GenerationMetrics` for the raw benchmark info.
- Check whether thinking output arrives in `chunk.channels` (e.g. a thought channel) rather than `content`. Stream only
  the answer text, or prefix thoughts visibly. Note what you find.

### 6.3 Model caching (required: 2–3 GB downloads)

- Use the **Cache Storage API** (`caches.open('litert-models-v1')`) keyed by model URL. On load: `match` → use
  `await response.blob()` (disk-backed) or `response.body`. Otherwise `fetch` with a progress-reporting `TransformStream`,
  `cache.put` a tee'd copy, and feed the other branch to `Engine.create`. Alternatively download → put → match → blob.
  Do whatever avoids holding 2–3 GB in JS memory twice.
- `inspect`: `cached = !!(await cache.match(url))`; size from a `HEAD` request's `Content-Length` (HF returns it on
  redirect targets) or the HF tree API.
- Handle quota errors (`QuotaExceededError`): surface them, and continue uncached if possible.
- Optional: "Clear cached models" button for both runtimes (Transformers.js has `ModelRegistry.clear_cache`).

### 6.4 Stateless vs. stateful conversations

- **Phase 1 (fair comparison): stateless.** Rebuild a conversation per request from the full history, exactly like the
  Transformers.js path re-processes history every turn. Benchmarks are single-turn anyway.
- **Phase 2 (optional): stateful.** Keep one `Conversation` per chat session id while the engine is loaded and only send
  the new user message. TTFT should drop sharply on long chats. Invalidate on regenerate, system prompt or settings
  change, or session switch. Report it as a separate result, since it's a runtime capability rather than raw speed.

### 6.5 System prompt and template

Pass the system prompt as a preface message `{ role: 'system', content: string }`. LiteRT applies its own prompt
template (`applyPromptTemplateInSession`). The Transformers.js gotcha (system content must be a string) is a Jinja
detail there; keep strings here too.

### 6.6 Multimodal (verify, don't assume)

Test: `sessionConfig: { visionModalityEnabled: true, audioModalityEnabled: true }` and a user message with
`content: [{ type: 'image', data: Blob }, { type: 'text', text }]` (see `SUPPORTED_IMAGE_MIME_TYPES` /
`SUPPORTED_AUDIO_MIME_TYPES`). Audio may want the original file Blob rather than our 16 kHz PCM. Our `Attachment`
keeps the original `blob`.
- If it works: enable attachments for LiteRT and record it in AGENTS.md.
- If it errors: disable the image/audio buttons when LiteRT is active, with a tooltip.

## 7. Tasks

1. Settings `runtime` + model config refactor (§5.2) + migration. Build passes with the Transformers.js path unchanged.
2. `litert.worker.ts`: load (with cache + progress), inspect, unload (`engine.delete()`), generate (stateless), interrupt.
3. `llm.ts`: runtime-aware worker creation/restart. Benchmark and report carry `runtime`.
4. UI: runtime select, runtime-specific controls, metrics display.
5. Multimodal verification (§6.6) and wiring or disabling.
6. User runs benchmarks on both runtimes (E2B, then E4B) in Chrome. Record a comparison table in AGENTS.md and
   update the README (runtime switch, known limitations).
7. (Optional) Phase 2 stateful conversations; self-hosted wasm.

## 8. Risks / open questions

- **Preview API churn:** pin `@litert-lm/core@0.18.0` exactly; nightlies may break.
- **Memory:** the E4B file is 3 GB. Non-streaming backends buffer the whole file in wasm memory (4 GB wasm32 limit).
  `GPU_ARTISAN` streams weights. If E4B fails, record it rather than fighting it.
- **COOP/COEP:** threaded wasm variants need `crossOriginIsolated`. The loader feature-detects and should fall back.
  If performance looks CPU-bound in the wasm glue, try adding COOP/COEP headers in `vite.config.ts` (`server.headers`).
  Check that cross-origin model/CDN fetches still work under COEP (HF and jsDelivr send CORP/CORS?).
- **Fairness:** sampler on GPU vs CPU, prompt templates, and the default `maxNumTokens` differ. Use greedy for the
  long-greedy benchmark on both and note the context length.
- **Agent environment:** the agent can't see GPU results (WSL, GPU on Windows). The user runs benchmarks and pastes the
  JSON export. Headless Windows Chrome via CDP works for UI checks (see AGENTS.md → Testing); avoid triggering 2–3 GB
  downloads in a throwaway profile.

## 9. References

- Docs: https://ai.google.dev/edge/litert-lm/js · npm: https://www.npmjs.com/package/@litert-lm/core ·
  GitHub: https://github.com/google-ai-edge/LiteRT-LM
- Models: https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm,
  https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm
- Related Transformers.js context: KV-cache pinning for Gemma 3n/4
  ([huggingface/transformers.js#1757](https://github.com/huggingface/transformers.js/pull/1757), in 4.3.1); long-context
  decode slowdown from per-step KV buffer churn
  ([huggingface/transformers.js#1741](https://github.com/huggingface/transformers.js/issues/1741), open).
