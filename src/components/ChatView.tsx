import { useEffect, useRef } from 'react'
import { regenerateLast } from '../lib/chatActions'
import { useActiveMessages, useChat } from '../state/chat'
import { useSession } from '../state/session'
import { Composer } from './Composer'
import { MessageBubble } from './MessageBubble'

export function ChatView() {
  const messages = useActiveMessages()
  const activeId = useChat((s) => s.activeId)
  const ready = useSession((s) => s.status === 'ready')
  const generating = useSession((s) => s.generating)
  const scroller = useRef<HTMLDivElement>(null)

  // Jump to the latest turn when opening a different session.
  useEffect(() => {
    if (scroller.current) scroller.current.scrollTop = scroller.current.scrollHeight
  }, [activeId])

  useEffect(() => {
    const el = scroller.current
    if (!el) return
    // Only auto-scroll when the user is already near the bottom.
    if (el.scrollHeight - el.scrollTop - el.clientHeight < 200) el.scrollTop = el.scrollHeight
  }, [messages])

  const last = messages.at(-1)

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div ref={scroller} className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl space-y-4 p-4">
          {messages.length === 0 && (
            <div className="py-16 text-center text-sm text-zinc-500">
              {ready ? 'Say hello, or attach an image or audio clip.' : 'Load a model from the sidebar to start chatting.'}
            </div>
          )}
          {messages.map((m) => (
            <MessageBubble key={m.id} message={m} />
          ))}
          {last?.role === 'assistant' && !last.pending && ready && !generating && (
            <div className="flex justify-start">
              <button className="btn-secondary text-xs" onClick={() => void regenerateLast()}>
                ↻ Regenerate
              </button>
            </div>
          )}
        </div>
      </div>
      <Composer />
    </div>
  )
}
