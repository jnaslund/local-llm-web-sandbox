# PRD: Local LLM Web Chat Prototype

## 1. Purpose
Build a prototype web app that runs **Gemma 4 (E2B / E4B)** fully in the user's browser via WebGPU, with no backend, and lets a small group of testers chat with it. The goal is to find out **how well in-browser LLM inference works** across real devices: load time, speed, output quality, and compatibility.

## 2. Goals / Non-goals
**Goals**
- Ship it as a static site with 100% client-side inference and no server runtime.
- Support multi-turn chat with streamed tokens, plus image and audio input.
- Collect comparable performance data from testers on mixed hardware.

**Non-goals**
- Server-side inference (Ollama, llama.cpp), user accounts, cloud sync.
- Production polish, analytics backend, or SEO.
- Guaranteed support for browsers that don't have WebGPU (detect and explain instead).

## 3. Users
A small group of testers (colleagues) on mixed laptops and desktops running Chrome, Edge or Safari 26+. Mobile is not a target but shouldn't break.

## 4. Tech stack
| Concern | Choice |
|---|---|
| Build / UI | Vite + React + TypeScript + Tailwind CSS |
| Inference | `@huggingface/transformers` v4 (≥ 4.3 for the Gemma 4 WebGPU KV-cache fix and Safari 26 support) |
| Backend | WebGPU (`device: "webgpu"`), `dtype: "q4f16"` |
| Threading | All inference runs in a **dedicated Web Worker**. The UI talks to it through typed `postMessage` messages. |
| Models | `onnx-community/gemma-4-E2B-it-ONNX` (default), `onnx-community/gemma-4-E4B-it-ONNX` |
| Weights | Loaded directly from the Hugging Face Hub and cached by the browser (Transformers.js Cache API) |
| Persistence | `localStorage` for settings and chat history (no image/audio blobs) |
| Hosting | Any static host (GitHub Pages, Netlify, Vercel, or Cloudflare Pages). HTTPS is required for WebGPU. |

Load and generate with `AutoProcessor`, `Gemma4ForConditionalGeneration` and `TextStreamer` (`skip_prompt`, `skip_special_tokens`). Build prompts with `processor.apply_chat_template(messages, { add_generation_prompt: true, enable_thinking })`.

## 5. Functional requirements

### 5.1 Startup and compatibility
- FR1: On load, check for `navigator.gpu` and request an adapter. If either fails, show a clear "WebGPU not supported" screen with browser recommendations.
- FR2: Check whether the adapter has the `shader-f16` feature. If it's missing, warn that q4f16 may fail or run slowly.
- FR3: Show the GPU adapter info (vendor and architecture, when available).

### 5.2 Model loading
- FR4: A model dropdown with **E2B** (default) and **E4B**, each showing its approximate download size.
- FR5: An explicit "Load model" button, so the download (GBs) never starts automatically.
- FR6: A progress UI that shows per-file and total progress (bytes and %) from `progress_callback`, plus the status phases: downloading, compiling/initializing, ready.
- FR7: Indicate when the model is cached and the load will be fast. Let the user switch models, which unloads and disposes the current one.
- FR8: Record load time (download time and init time, measured separately).

### 5.3 Chat
- FR9: Multi-turn chat. The full history is sent through the chat template on every turn.
- FR10: Stream tokens into the assistant bubble as they arrive. Render Markdown, including code blocks.
- FR11: A Stop button that interrupts generation (Transformers.js `InterruptableStoppingCriteria`).
- FR12: Buttons for a new chat, clearing the chat, and regenerating the last answer.
- FR13: Disable input while the model is loading or generating.

### 5.4 Multimodal input
- FR14: Attach an image (file picker, paste, or drag-drop). Preview it in the message.
- FR15: Attach an audio file, or record from the microphone. Decode to 16 kHz mono with `AudioContext({ sampleRate: 16000 })`.
- FR16: Map attachments to `{ type: "image" }` / `{ type: "audio" }` content parts in the message.

### 5.5 Settings (persisted)
- FR17: System prompt, temperature, top-p, max new tokens, thinking toggle (`enable_thinking`). Setting temperature to 0 means greedy decoding (`do_sample: false`).
- FR18: Keep chat history in localStorage, text only. Attachments show as placeholders after a reload.

### 5.6 Metrics and evaluation
- FR19: Per-response metrics: time to first token, tokens/sec (decode), total tokens, and total time. Show them under each assistant message.
- FR20: A session metrics panel showing model, dtype, load times, GPU adapter, browser/UA, and average tokens/sec.
- FR21: **Run benchmark**: a fixed prompt set (e.g. one short and one long-generation prompt, greedy, fixed `max_new_tokens`). Results go to the metrics panel.
- FR22: **Export report**: download or copy a JSON file with device info, model, settings, load metrics, benchmark results, and optionally the chat transcript.

## 6. Non-functional requirements
- The UI stays responsive (60 fps scrolling) during load and generation, which is why inference runs in the worker.
- Errors are shown to the user: out of memory, lost GPU device, failed download, unsupported operation. Include a "Reload model" recovery button.
- No data leaves the browser except the model weight downloads from Hugging Face.

## 7. Architecture sketch
```
src/
  worker/llm.worker.ts     # load model, generate, interrupt, emit progress/tokens/metrics
  worker/protocol.ts       # typed messages: load | generate | interrupt | unload  ->  progress | token | done | error | metrics
  lib/webgpu.ts            # capability detection, adapter info
  lib/audio.ts             # decode/resample to 16 kHz, mic recording
  lib/benchmark.ts         # fixed prompts + runner
  lib/report.ts            # JSON export
  state/                   # chat + settings store (persisted)
  components/              # ModelLoader, Chat, MessageBubble, Composer, SettingsPanel, MetricsPanel
```

## 8. Success criteria
- E2B loads and chats in Chrome on at least 3 different tester machines. E4B is also tested on at least one machine.
- Testers can send back an exported report that includes benchmark results.
- Image and audio prompts produce sensible responses.

## 9. Risks / open questions
- **Download size and time** (roughly 1.5–3 GB for E2B, about 2x that for E4B). Mitigate with explicit opt-in, progress UI, and caching.
- **VRAM / memory limits** on integrated GPUs, especially with E4B. Catch the errors and suggest switching to E2B.
- **Browser differences.** Safari and Firefox WebGPU support varies. Detect it and let the benchmark show it.
- **Per-turn cost.** Sending the full history on every turn makes long chats slow. KV-cache reuse across turns is a possible follow-up.
- Possible follow-ups: try the `qat-mobile` ONNX variants, compare against LiteRT-LM Web, add an Ollama provider for a baseline.
