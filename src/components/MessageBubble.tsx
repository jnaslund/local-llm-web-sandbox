import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { formatMs, formatProfile, formatRuntimeInfo, formatTps } from '../lib/format'
import type { ChatMessage } from '../state/chat'

export function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user'
  const m = message.metrics

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[85%] space-y-2 ${isUser ? 'items-end' : 'items-start'} flex flex-col`}>
        {message.attachments?.map((a, i) =>
          a.kind === 'image' && a.previewUrl ? (
            <img key={i} src={a.previewUrl} alt={a.name} className="max-h-60 rounded-lg border border-zinc-200 dark:border-zinc-800" />
          ) : a.kind === 'audio' && a.previewUrl ? (
            <audio key={i} src={a.previewUrl} controls className="max-w-full" />
          ) : (
            <span key={i} className="rounded-md bg-zinc-200 px-2 py-1 text-xs dark:bg-zinc-800">
              {a.kind === 'image' ? '🖼️' : '🎤'} {a.name} (not available after reload)
            </span>
          ),
        )}

        {(message.text || message.pending || isUser) && (
          <div
            className={`rounded-2xl px-4 py-2.5 text-sm ${
              isUser
                ? 'bg-indigo-600 whitespace-pre-wrap text-white'
                : 'border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900'
            }`}
          >
            {isUser ? (
              message.text
            ) : (
              <div className="prose prose-sm max-w-none dark:prose-invert prose-pre:bg-zinc-900 prose-pre:text-zinc-100">
                <Markdown remarkPlugins={[remarkGfm]}>{message.text}</Markdown>
                {message.pending && <span className="inline-block h-4 w-1.5 animate-pulse bg-zinc-400 align-middle" />}
              </div>
            )}
          </div>
        )}

        {message.error && (
          <div className="rounded-lg bg-red-50 px-3 py-2 text-xs text-red-800 dark:bg-red-950 dark:text-red-300">
            Generation failed: {message.error}
          </div>
        )}

        {m && (
          <div className="flex flex-wrap gap-x-3 text-[11px] text-zinc-500 tabular-nums">
            <span title="Time to first token">TTFT {formatMs(m.ttftMs)}</span>
            <span>{formatTps(m.tokensPerSec)}</span>
            <span>
              {m.promptTokens} in / {m.outputTokens} out
            </span>
            <span>{formatMs(m.totalMs)} total</span>
            {m.interrupted && <span className="text-amber-600">stopped</span>}
            {m.reusedContext !== undefined && (
              <span title="LiteRT-LM stateful mode: whether the conversation's KV cache was reused (only the new message prefilled) or rebuilt from the full history">
                {m.reusedContext ? 'KV reused' : 'KV rebuilt'}
              </span>
            )}
            {m.profile && (
              <span className="basis-full" title="Average per generated token: time inside each ONNX session vs. everything else (sampling, JS)">
                {formatProfile(m.profile)}
              </span>
            )}
            {m.runtimeInfo && (
              <span className="basis-full" title="LiteRT-LM's own measurements (Conversation.getBenchmarkInfo)">
                {formatRuntimeInfo(m.runtimeInfo)}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
