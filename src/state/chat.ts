import { create } from 'zustand'
import { persist, type PersistStorage, type StorageValue } from 'zustand/middleware'
import type { GenerationMetrics } from '../worker/protocol'

export type Attachment = {
  kind: 'image' | 'audio'
  name: string
  /** In-memory only; lost on reload (only text history is persisted). */
  blob?: Blob
  previewUrl?: string
  pcm?: Float32Array
}

export type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  text: string
  /** Assistant reasoning (thinking mode), kept apart from the answer and not sent back to the model. */
  thinking?: string
  attachments?: Attachment[]
  metrics?: GenerationMetrics
  error?: string
  pending?: boolean
}

export type ChatSession = {
  id: string
  title: string
  createdAt: number
  updatedAt: number
  /** Model that produced the most recent answer in this session. */
  modelId: string | null
  messages: ChatMessage[]
}

const DEFAULT_TITLE = 'New chat'
const EMPTY: ChatMessage[] = []

type ChatStore = {
  /** Most recently updated first. */
  sessions: ChatSession[]
  activeId: string
  newSession: () => void
  select: (id: string) => void
  remove: (id: string) => void
  rename: (id: string, title: string) => void
  add: (sessionId: string, msg: ChatMessage) => void
  patch: (sessionId: string, id: string, patch: Partial<ChatMessage>) => void
  appendText: (sessionId: string, id: string, text: string, thought?: boolean) => void
  removeFrom: (sessionId: string, id: string) => void
  setModel: (sessionId: string, modelId: string) => void
}

export function newId(): string {
  return crypto.randomUUID()
}

function createSession(): ChatSession {
  const now = Date.now()
  return { id: newId(), title: DEFAULT_TITLE, createdAt: now, updatedAt: now, modelId: null, messages: [] }
}

function titleFrom(msg: ChatMessage): string {
  const text = msg.text.trim().replace(/\s+/g, ' ')
  if (text) return text.length > 48 ? `${text.slice(0, 47)}…` : text
  const kinds = msg.attachments?.map((a) => a.kind) ?? []
  return kinds.includes('image') ? 'Image prompt' : kinds.includes('audio') ? 'Audio prompt' : DEFAULT_TITLE
}

type PersistedChat = Pick<ChatStore, 'sessions' | 'activeId'>

/**
 * localStorage storage that serializes at most every 500 ms. Streaming updates the store once per token, and
 * stringifying every session that often is wasted main-thread work.
 */
function debouncedStorage<S>(delayMs = 500): PersistStorage<S> {
  const pending = new Map<string, StorageValue<S>>()
  let timer: ReturnType<typeof setTimeout> | undefined
  const flush = () => {
    clearTimeout(timer)
    timer = undefined
    for (const [name, value] of pending) {
      try {
        localStorage.setItem(name, JSON.stringify(value))
      } catch (err) {
        console.warn('[chat] failed to save chats', err)
      }
    }
    pending.clear()
  }
  if (typeof window !== 'undefined') window.addEventListener?.('pagehide', flush)
  return {
    getItem: (name) => {
      const raw = localStorage.getItem(name)
      return raw ? (JSON.parse(raw) as StorageValue<S>) : null
    },
    setItem: (name, value) => {
      pending.set(name, value)
      timer ??= setTimeout(flush, delayMs)
    },
    removeItem: (name) => {
      pending.delete(name)
      localStorage.removeItem(name)
    },
  }
}

const initial = createSession()

export const useChat = create<ChatStore>()(
  persist<ChatStore, [], [], PersistedChat>(
    (set) => {
      /** Apply a change to one session; `touch` bumps it to the top of the list. */
      const update = (sessionId: string, fn: (s: ChatSession) => ChatSession, touch = false) =>
        set((state) => {
          const target = state.sessions.find((s) => s.id === sessionId)
          if (!target) return state
          const updated = fn(target)
          if (!touch) return { sessions: state.sessions.map((s) => (s.id === sessionId ? updated : s)) }
          return {
            sessions: [{ ...updated, updatedAt: Date.now() }, ...state.sessions.filter((s) => s.id !== sessionId)],
          }
        })

      const mapMessages = (sessionId: string, fn: (m: ChatMessage[]) => ChatMessage[], touch = false) =>
        update(sessionId, (s) => ({ ...s, messages: fn(s.messages) }), touch)

      return {
        sessions: [initial],
        activeId: initial.id,

        newSession: () =>
          set((state) => {
            // Reuse the active session if it's still empty instead of piling up blank chats.
            const active = state.sessions.find((s) => s.id === state.activeId)
            if (active && active.messages.length === 0) return state
            const session = createSession()
            return { sessions: [session, ...state.sessions], activeId: session.id }
          }),

        select: (id) => set({ activeId: id }),

        remove: (id) =>
          set((state) => {
            const sessions = state.sessions.filter((s) => s.id !== id)
            if (sessions.length === 0) {
              const session = createSession()
              return { sessions: [session], activeId: session.id }
            }
            return { sessions, activeId: state.activeId === id ? sessions[0].id : state.activeId }
          }),

        rename: (id, title) => update(id, (s) => ({ ...s, title: title.trim() || s.title })),

        add: (sessionId, msg) =>
          update(
            sessionId,
            (s) => ({
              ...s,
              title: s.title === DEFAULT_TITLE && msg.role === 'user' ? titleFrom(msg) : s.title,
              messages: [...s.messages, msg],
            }),
            true,
          ),

        patch: (sessionId, id, patch) =>
          mapMessages(sessionId, (ms) => ms.map((m) => (m.id === id ? { ...m, ...patch } : m))),

        appendText: (sessionId, id, text, thought) =>
          mapMessages(sessionId, (ms) =>
            ms.map((m) =>
              m.id !== id ? m : thought ? { ...m, thinking: (m.thinking ?? '') + text } : { ...m, text: m.text + text },
            ),
          ),

        removeFrom: (sessionId, id) =>
          mapMessages(sessionId, (ms) => {
            const i = ms.findIndex((m) => m.id === id)
            return i === -1 ? ms : ms.slice(0, i)
          }),

        setModel: (sessionId, modelId) => update(sessionId, (s) => ({ ...s, modelId })),
      }
    },
    {
      name: 'llm-proto.chat',
      version: 2,
      storage: debouncedStorage<PersistedChat>(),
      partialize: (s): PersistedChat => ({
        activeId: s.activeId,
        sessions: s.sessions.map((session) => ({
          ...session,
          messages: session.messages
            .filter((m) => !m.pending)
            .map((m) => ({ ...m, attachments: m.attachments?.map(({ kind, name }) => ({ kind, name })) })),
        })),
      }),
      migrate: (persisted, version): PersistedChat => {
        // v1 stored a single chat as `{ messages }`; wrap it into the first session.
        if (version < 2) {
          const messages = ((persisted as { messages?: ChatMessage[] })?.messages ?? []).filter((m) => !m.pending)
          const session = createSession()
          const firstUser = messages.find((m) => m.role === 'user')
          if (firstUser) session.title = titleFrom(firstUser)
          session.messages = messages
          return { sessions: [session], activeId: session.id }
        }
        return persisted as PersistedChat
      },
    },
  ),
)

export function useActiveSession(): ChatSession | undefined {
  return useChat((s) => s.sessions.find((x) => x.id === s.activeId))
}

export function useActiveMessages(): ChatMessage[] {
  return useChat((s) => s.sessions.find((x) => x.id === s.activeId)?.messages ?? EMPTY)
}

export function getActiveSession(): ChatSession {
  const { sessions, activeId } = useChat.getState()
  return sessions.find((s) => s.id === activeId) ?? sessions[0]
}
