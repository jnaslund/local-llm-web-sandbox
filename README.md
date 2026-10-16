# Local LLM Web Chat Prototype

Chat with Gemma 4 (E2B / E4B) running **entirely in the browser** via WebGPU, on a switchable runtime:
[Transformers.js](https://github.com/huggingface/transformers.js) (ONNX Runtime Web) or Google's
[LiteRT-LM Web](https://ai.google.dev/edge/litert-lm/js) (`@litert-lm/core`, early preview). No backend — the model
weights are downloaded from the Hugging Face Hub once and cached by the browser.

**Live demo:** https://jnaslund.github.io/local-llm-web-sandbox/ (deployed from `main` by
[.github/workflows/pages.yml](.github/workflows/pages.yml)).

See [docs/PRD.md](docs/PRD.md) for requirements and scope.

## Getting started

```bash
npm install
npm run dev
```

Open the printed URL in Chrome/Edge 113+ (or Safari 26+), pick a runtime and a model, and click **Download & load**.
The first load downloads 2–5 GB; later loads come from the browser cache. Switching runtime unloads the other one.
Run **Metrics → Run benchmark** on each runtime to compare; runs and exported reports are tagged with the runtime.

```bash
npm run build     # static output in dist/, deployable to any static host (HTTPS required for WebGPU)
npm run lint
```

## Structure

```
src/
  config/models.ts         runtimes + model list (HF repo / .litertlm URL per runtime)
  worker/llm.worker.ts     Transformers.js worker: load, generate (streamed), interrupt, metrics
  worker/litert.worker.ts  LiteRT-LM worker, same protocol
  worker/protocol.ts       typed UI <-> worker messages
  lib/llm.ts               worker client (one worker per runtime); routes worker events into the stores
  lib/litertCache.ts       Cache Storage for LiteRT-LM model files
  lib/chatActions.ts       send / regenerate / stop; builds the prompt from chat history
  lib/webgpu.ts            WebGPU + shader-f16 detection, adapter info
  lib/audio.ts             decode to 16 kHz mono, microphone recording
  lib/benchmark.ts         fixed greedy benchmark prompts
  lib/report.ts            JSON test report (device, load, benchmarks, optional transcript)
  state/                   zustand stores: settings + chat sessions (persisted to localStorage), session (in-memory)
  components/              ModelPanel, ChatList, ChatView, MessageBubble, Composer, SettingsPanel, MetricsPanel
```

## Known limitations

- Only the latest user turn's image/audio is sent to the model; earlier attachments are described in text
  (the Gemma 4 processor handles a single audio clip per prompt).
- Transformers.js re-processes the full chat history on every turn (no KV-cache reuse), so long chats get slower.
  LiteRT-LM keeps the current chat's conversation between turns (Settings → LiteRT-LM). It rebuilds from the history
  after regenerate, a settings or system prompt change, a stopped reply, or switching chats.
- LiteRT-LM is text-only here: 0.18.0 can't enable its vision/audio encoders from JS, so attachments are disabled
  for it. Its ~109 MB of wasm loads from jsDelivr (self-hosting it is a later option). Cached LiteRT-LM model files
  can be deleted from the model panel.
- Performance experiments (Settings) apply to Transformers.js only.
- With thinking mode on, the model's reasoning is streamed inline with the answer.
- Chat sessions live in this browser's localStorage (sidebar → Chats). Attachments are not persisted;
  after a reload only the text history remains.
