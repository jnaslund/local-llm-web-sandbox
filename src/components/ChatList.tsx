import { useState } from 'react'
import { modelLabel } from '../config/models'
import { useChat, type ChatSession } from '../state/chat'
import { useSession } from '../state/session'

export function ChatList({ onSelect }: { onSelect?: () => void }) {
  const sessions = useChat((s) => s.sessions)
  const activeId = useChat((s) => s.activeId)
  const newSession = useChat((s) => s.newSession)

  return (
    <div className="space-y-2">
      <button
        className="btn-secondary w-full"
        onClick={() => {
          newSession()
          onSelect?.()
        }}
      >
        + New chat
      </button>
      <ul className="-mx-1 space-y-0.5">
        {sessions.map((s) => (
          <ChatListItem key={s.id} session={s} active={s.id === activeId} onSelect={onSelect} />
        ))}
      </ul>
      <p className="text-[11px] text-zinc-500">
        Chats are stored in this browser only. Images and audio aren't kept after a reload.
      </p>
    </div>
  )
}

function ChatListItem({ session, active, onSelect }: { session: ChatSession; active: boolean; onSelect?: () => void }) {
  const { select, remove, rename } = useChat.getState()
  const generating = useSession((s) => s.generating)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(session.title)
  const streaming = generating && session.messages.some((m) => m.pending)
  const turns = session.messages.filter((m) => m.role === 'user').length

  const commit = () => {
    rename(session.id, draft)
    setEditing(false)
  }

  return (
    <li
      className={`group flex items-center gap-1 rounded-lg px-2 py-1.5 ${
        active ? 'bg-indigo-50 dark:bg-indigo-950/60' : 'hover:bg-zinc-100 dark:hover:bg-zinc-800'
      }`}
    >
      {editing ? (
        <input
          autoFocus
          className="input py-1"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === 'Enter') commit()
            if (e.key === 'Escape') setEditing(false)
          }}
        />
      ) : (
        <button
          className="min-w-0 flex-1 text-left"
          onClick={() => {
            select(session.id)
            onSelect?.()
          }}
          onDoubleClick={() => {
            setDraft(session.title)
            setEditing(true)
          }}
          title="Double-click to rename"
        >
          <span className={`block truncate text-sm ${active ? 'font-medium' : ''}`}>
            {streaming && <span className="mr-1 inline-block size-1.5 animate-pulse rounded-full bg-indigo-500 align-middle" />}
            {session.title}
          </span>
          <span className="block truncate text-[11px] text-zinc-500">
            {formatWhen(session.updatedAt)} · {turns} {turns === 1 ? 'turn' : 'turns'}
            {session.modelId && ` · ${modelLabel(session.modelId)}`}
          </span>
        </button>
      )}
      {!editing && (
        <button
          className="shrink-0 rounded p-1 text-xs text-zinc-400 opacity-0 group-hover:opacity-100 hover:text-red-600 focus:opacity-100 disabled:hidden"
          disabled={streaming}
          aria-label={`Delete ${session.title}`}
          onClick={() => {
            if (session.messages.length === 0 || confirm(`Delete "${session.title}"?`)) remove(session.id)
          }}
        >
          🗑
        </button>
      )}
    </li>
  )
}

function formatWhen(ts: number): string {
  const d = new Date(ts)
  const now = new Date()
  if (d.toDateString() === now.toDateString()) return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  const days = Math.floor((now.getTime() - ts) / 86_400_000)
  if (days < 7) return d.toLocaleDateString([], { weekday: 'short' })
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' })
}
