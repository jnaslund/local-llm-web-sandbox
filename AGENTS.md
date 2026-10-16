# AGENTS.md

Guidance for AI agents working on this repo. Read [README.md](README.md) for the file map, [docs/PRD.md](docs/PRD.md)
for scope, and `docs/specs/` for planned work. This file records what is **not** obvious from the code.

## What this is

Prototype: chat with Gemma 4 E2B/E4B running fully in the browser (WebGPU, no backend) on two switchable runtimes:
Transformers.js 4.3 (`@huggingface/transformers`) on ONNX Runtime Web 1.31-dev, and LiteRT-LM Web
(`@litert-lm/core`, pinned to 0.18.0 — early preview). Goal is to evaluate how well in-browser inference works —
metrics and the benchmark/report flow matter as much as the chat.

## Commands

- `npm run dev` — the user usually has this running on :5173. Don't start another server on 5173; use another port.
- `npm run build` — runs `tsc -b`; this is the type check. Run it plus `npm run lint` (oxlint) after every change.
- No test suite. Store logic can be tested in plain Node 24 (native TS stripping) — see "Testing" below.

## Architecture rules

- All inference lives in workers: `src/worker/llm.worker.ts` (Transformers.js) and `src/worker/litert.worker.ts`
  (LiteRT-LM). Both speak the typed messages in `src/worker/protocol.ts`. `src/lib/llm.ts` owns **one** worker at a
  time (for `settings.runtime`) and routes responses into zustand stores. Loading on another runtime or with other
  perf options terminates the worker (frees its GPU memory). `@litert-lm/core` is imported only in its worker so
  it stays in that chunk.
- Model ids are runtime-neutral (`gemma-4-e2b`); `src/config/models.ts` maps them to the HF repo / `.litertlm` URL.
  `findModel` also accepts the old repo ids that chats persisted before settings v3.
- Inspections (cached/size) are keyed `runtime:modelId`. LiteRT files are inspected on the main thread
  (`lib/litertCache.ts`); Transformers.js inspection uses its worker, or a throwaway one when LiteRT's is active.
- Stores: `settings` and `chat` (sessions) persist to localStorage; `session` (model status, metrics, benchmarks) is
  in-memory only. Chat persistence is debounced (500 ms) because streaming updates the store once per token.
- Replies stream into the session they started in (`runAssistantTurn(sessionId)`), even if the user switches chats.
- Attachments (image Blob, audio PCM) are in-memory only; persisted messages keep `{kind, name}`.

## Gotchas (learned the hard way)

- **Gemma 4 chat template**: the system turn's content is `| trim`-ed directly, so it **must be a plain string**.
  Send text-only turns as strings; use content parts (`{type:'image'|'audio'|'text'}`) only for turns with media.
  Content parts on the system message → `Unknown ArrayValue filter: trim`.
- **Media**: the Gemma 4 processor handles one audio clip per prompt, so only the latest user turn carries real media;
  earlier attachments are described in text (`priorAttachments`).
- **`enable_thinking`** is forwarded to the Jinja template at runtime but missing from the TS options type — hence the cast.
- **top_p is a no-op in Transformers.js 4.3** (`TopPLogitsWarper` has no `_call`; wiring is a TODO in
  `modeling_utils.js`). We expose **top-k** instead. Sampling = CPU `top_k` over the 262k vocab; measured cost ≈ 1 ms.
- **Vite HMR + worker**: re-executing `llm.ts` creates a new worker while store state survives. `llm.ts` therefore
  resets session state on module init and terminates the old worker in `import.meta.hot.dispose`. Keep that when
  editing. If the user reports stuck/disabled buttons after you edited files, a page reload (F5) clears it.
- **Persisted shape changes**: bump the zustand `version` and add a `migrate`, and read new fields defensively
  (`loadMetrics.perf?.…`) — old in-memory/persisted objects lack them. Panels are wrapped in `ErrorBoundary` so a
  bad read doesn't blank the app.
- **Tailwind v4**: custom classes used with `@apply` must be declared with `@utility` (see `src/index.css`), not
  `@layer components`.
- **LiteRT-LM in a module worker**: its loader calls `importScripts()` in workers (module workers throw) and the
  Emscripten glue looks for the `.wasm` next to the worker URL. `litert.worker.ts` installs a sync-XHR +
  indirect-eval `importScripts` and sets `self.Module.locateFile` to the jsDelivr wasm dir. Keep both.
- **LiteRT-LM caching**: `Engine.create` fetches the model on every load. `lib/litertCache.ts` downloads into Cache
  Storage (`litert-models-v1`) and passes the engine a disk-backed Blob read back from it. The web `.litertlm` files
  only contain GPU "artisan" sections, so only `Backend.GPU_ARTISAN` works (CPU/CPU_ARTISAN fail). That means no
  Node/CPU testing.
- **LiteRT-LM is text-only (0.18)**: `visionModalityEnabled`/`audioModalityEnabled` fail at createConversation with
  "Vision/Audio options should not be null". The LLM `Engine` exposes no vision/audio executor settings (only
  `EmbeddingEngine` has them); upstream main is the same as of 2026-10-16. Attachments are disabled in the UI for it.
  When it lands, media parts must be `{type, blob: base64}` (the C++ format); the JS types' `data: Blob` would be
  JSON-serialized to `{}`.
- **LiteRT-LM stateful conversations** (Settings → LiteRT-LM → Keep conversation state, on by default): chats pass
  their session id as `conversationId`, and the worker keeps **one** live `Conversation` (KV cache on the GPU). It
  continues it only if the id and the settings match and the request's history (minus the new user turn) is exactly
  what it has seen. That history includes the assistant answer as streamed (without reasoning), which is what the chat store sends back.
  Any mismatch drops it and rebuilds from the full history; that covers regenerate, deleted turns, system prompt and
  settings changes, and switching chats. Interrupted turns aren't kept. Settings are part of the key because
  `maxOutputTokens`/sampler are fixed per conversation. Stateless requests (benchmarks, setting off) also drop it, so
  only one conversation ever exists. Replies show "KV reused"/"KV rebuilt" (`reusedContext`), and with reuse
  `promptTokens` counts only the new turn.
- **Thinking output**: workers tag streamed text with `thought: true` for reasoning. The store keeps it in
  `ChatMessage.thinking`, apart from `text`, and only `text` goes back to the model (Gemma's template strips thinking
  from past turns anyway). On Transformers.js, Gemma 4 emits `<|channel>thought\n…<channel|>`. Both markers are special
  tokens that `TextStreamer` drops *without flushing its word buffer*, so `ChannelStreamer` in the worker flushes at
  each marker and strips the `thought\n` header.
- **LiteRT-LM streaming**: answer text is in `chunk.content`, reasoning in `chunk.channels.thought`. Chunks aren't guaranteed to be single tokens; token counts come from
  `getBenchmarkInfo()`. Its "GetProfileSummary not implemented for GpuArtisan" console warnings are harmless.
- Inspect Transformers.js internals in `node_modules/@huggingface/transformers/src` (readable JS) and
  `node_modules/onnxruntime-web/dist/ort.webgpu.js` (unminified) rather than guessing.

## Performance findings (RTX 3080, Windows, E2B q4f16)

Per-token profile (`DecodeProfile`, shown under each reply and in benchmarks; sessions are timed by wrapping
`session.run`):

| | Chrome 155 | Edge 154 |
|---|---|---|
| total | ~35 ms (29 tok/s) | ~45 ms (22 tok/s) |
| `decoder_model_merged` | ~22 ms | ~33 ms |
| `embed_tokens` | ~11 ms | ~10 ms |
| other (sampling, JS) | ~1–2 ms | ~1–2 ms |

- Bottleneck is ONNX Runtime WebGPU, not app code: decoder weights are ~1.5 GB → bandwidth floor ~3 ms/token.
  Custom WGSL engines reach 100+ tok/s; Transformers.js reports cluster at 20–30 tok/s.
- `embed_tokens` is two `GatherBlockQuantized` + int64 `Less/Equal/Where` ops, yet costs ~11 ms/token.
- The wasm (CPU) build has **no** `GatherBlockQuantized` kernel — running `embed_tokens` on `wasm` is not an option.
- Experiments (Settings → Performance experiments, `PerfOptions`). They are applied at session creation and the
  WebGPU device context is created once per worker, so changing them restarts the worker:
  - `int64OnGpu` → session config `ep.webgpuexecutionprovider.enableInt64 = '1'`
  - `disableValidation` → session config `ep.webgpuexecutionprovider.validationMode = 'disabled'`
  - `embedOutputsOnGpu` → `preferredOutputLocation` gpu-buffer for `inputs_embeds`/`per_layer_inputs`. Decode steps
    feed GPU tensors to the decoder and free them after its run; prefill downloads them to CPU (media merge reads
    them). See `settleEmbedOutputs` in the worker.
- Results, Chrome 155, long runs (ms/token · decoder · embed):
  - baseline: 38–40 · 26–28 · 11 (an earlier baseline run: 34–35 · 21–23 · 11 — run-to-run noise is ~±3 ms)
  - + int64: 34.5 · 22 · 11 — embed unchanged, so int64 CPU fallback was **not** the embed cost
  - + int64 + no validation: 33.9 · 21.8 · 10.8 — within noise
  - Conclusion: neither flag meaningfully helps.
  - + int64 + `embedOutputsOnGpu`: **27.6–28.4 · 21.5–22.4 · 4.7** (35–36 tok/s, vs 33.6 ms / 29.8 tok/s with int64
    alone in the same session). The per-token readback of embed outputs was the embed cost. ~+20%.
  - Remaining: decoder ~22 ms is ORT's floor (unchanged by every option; includes one logits readback per token
    for CPU sampling). Embed ~4.7 ms, app overhead ~1.3 ms. Bigger gains need a different runtime.
  - Image prompts verified working with `embedOutputsOnGpu` (31 tok/s). `int64OnGpu` + `embedOutputsOnGpu` are now
    the defaults (`DEFAULT_PERF`).
- The first few decode steps (`short-answer`, 7 steps) are noisy/slow (warm-up); compare the long runs.
- Upstream (checked 2026-10-16): KV-cache GPU pinning for Gemma 3n/4 was fixed in
  huggingface/transformers.js#1757 (included in 4.3.1). No issue found for the embed_tokens output readback that
  `embedOutputsOnGpu` works around — a candidate to report. huggingface/transformers.js#1741 (open) documents
  long-context decode slowdowns from per-step KV buffer churn; relevant if long chats get slow.
- Smaller lever left on the Transformers.js path: ORT's JSPI build.

### LiteRT-LM vs Transformers.js (RTX 3080, Chrome 155)

LiteRT-LM 0.18.0, `GPU_ARTISAN`, default sampler backend, context 8192; stateless (fresh conversation per request,
full history prefilled, same as Transformers.js). Transformers.js with defaults (`int64OnGpu` + `embedOutputsOnGpu`).
User's report, normal Chrome window, 2026-10-16 (tok/s = our wall-clock decode rate; LiteRT's own figure agrees
within 0.3 tok/s):

| | TJS E2B | LiteRT E2B | TJS E4B | LiteRT E4B |
|---|---|---|---|---|
| long-greedy decode | 36.1 tok/s (27.7 ms/tok) | **84.5 tok/s** (11.8 ms/tok) | 30.5 tok/s (32.8 ms/tok) | **60.5 tok/s** (16.5 ms/tok) |
| long-sampled (T=1, top-k 64) | 31.7 tok/s | 84.6 tok/s | 32.5 tok/s | 61.5 tok/s |
| TTFT, 28-token prompt | 82–90 ms | 63–64 ms | 112–115 ms | 140–150 ms |
| load | ~5 s (cached) | ~2.4–3 s (cached) | ~7 s (cached) | 47 s uncached (44 s download + 2.9 s init) |
| download size | 3.17 GB | 1.87 GB | 4.84 GB | 2.77 GB |

- LiteRT-LM decodes **2.3×** (E2B) / **2.0×** (E4B) faster than Transformers.js. LiteRT E4B beats TJS E2B by 1.7×.
- TJS barely slows down from E2B to E4B (decoder 22 → 26 ms, embed ~4.7 ms on both). That fits overhead-bound
  dispatch, not bandwidth. LiteRT scales more like weight size (11.8 → 16.5 ms).
- Sampled ≈ greedy on LiteRT (GPU sampling). On TJS the sampled-vs-greedy gap is within run-to-run noise
  (−4 ms/token on E2B, +2 on E4B).
- E4B (3 GB file) loads fine with `GPU_ARTISAN` streaming; no wasm memory problem.
- `short-answer` (8 tokens) is noisy on both runtimes and includes warm-up; compare the long runs. The first request
  after a load can take seconds (shader warm-up).
- **Free VRAM before switching.** Switching LiteRT E4B → TJS E4B while the old worker was only `terminate()`d gave
  2.5 → 7.4 → 17.4 tok/s with `embed_tokens` at 190 → 56 → 17 ms/token. That is VRAM overcommit on the 10 GB card
  (WDDM paging). `retireWorker` in `llm.ts` now unloads the model and waits for `unloaded` before terminating. The
  same switch then gives the normal numbers above.
- Agent's headless run (same machine) measured LiteRT E2B at ~74 tok/s; headless is ~12% slower, so don't mix the two.
- **Follow-up turn** (benchmark `followup-*`: a second turn after the 256-token long-greedy reply, 302-token
  history; user's report, normal Chrome, E2B, 2026-10-16):

  | | TJS stateless | LiteRT stateless | LiteRT stateful |
  |---|---|---|---|
  | TTFT | 199 ms | 125 ms | **43 ms** |
  | tokens prefilled | 302 | 302 (~2,960 tok/s) | 16 |
  | decode | 38 tok/s | 84 tok/s | 87 tok/s |

  Keeping the conversation cuts TTFT ~3× here (~5× vs TJS), and the gap grows with history length. Decode speed is
  unaffected. In the agent's headless chat runs, reused turns took 47–56 ms vs 114–330 ms for rebuilds of 120–350
  tokens. Outputs can differ slightly between the two modes (64 vs 57 tokens). A kept conversation holds the model's
  raw turn, while a rebuild re-templates the streamed text.
- The same report had TJS long-greedy at 31.3 tok/s and long-sampled at 35.1, vs 36.1/31.7 earlier: TJS run-to-run
  noise is ~±4 ms/token. LiteRT was steady at 84–88 tok/s across runs.
- Quality spot checks match (multi-turn recall, thinking, 17×23 = 391).

## Testing

- **Store logic**: write a script that shims `localStorage` with `Object.defineProperty(globalThis, 'localStorage', …)`
  (Node 24 has its own global that shadows plain assignment), sets `globalThis.window = globalThis`, then
  `await import('<abs path>/src/state/chat.ts')`. Wait >500 ms before reading storage (debounce).
- **Chat template**: render the model's `chat_template.jinja` (download from the HF repo) with
  `node_modules/@huggingface/jinja/dist/index.js` to validate message shapes without a GPU.
- **UI in a real browser (this machine is WSL2; the GPU is on Windows)**:
  - WebGPU isn't available in WSL. Launch Windows Chrome headless with its own profile:
    `"/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --remote-debugging-port=9333 --user-data-dir='C:\Temp\<dir>' --enable-unsafe-webgpu`.
  - The debug port is only reachable from Windows, so run puppeteer-core with Windows Node:
    copy the script and `node_modules` to `C:\Temp\…` and run `cmd.exe /c "cd /d C:\Temp\… && node script.mjs"`.
    The page at `http://localhost:5173` reaches the WSL dev server.
  - Vite adds `?t=` to module URLs; to touch app state, import the exact URL from
    `performance.getEntriesByType('resource')` (e.g. `/src/state/session.ts?t=…`), or you get a separate store instance.
  - Vite HMR messages don't reliably reach a headless page across the WSL↔Windows relay; re-import a module with a
    fresh `?t=` to simulate a hot update.
  - Never kill the user's own Chrome: stop only processes whose command line contains your `--user-data-dir`.
    Clean up `C:\Temp` afterwards. Loading a model in a fresh profile downloads ~3 GB — avoid unless needed.
- You cannot see GPU results yourself; the user runs benchmarks and pastes the JSON report (Metrics → Export).

## Conventions

- Match existing style: no semicolons, single quotes, 2-space indent, concise comments only where non-obvious.
- Keep the README's "Known limitations" and this file current when behavior or findings change.
