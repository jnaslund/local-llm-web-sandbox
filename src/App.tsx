import { useEffect, useState } from 'react'
import { ChatList } from './components/ChatList'
import { ChatView } from './components/ChatView'
import { ErrorBoundary } from './components/ErrorBoundary'
import { MetricsPanel } from './components/MetricsPanel'
import { ModelPanel } from './components/ModelPanel'
import { SettingsPanel } from './components/SettingsPanel'
import { UnsupportedScreen } from './components/UnsupportedScreen'
import { detectWebGPU, gpuLabel } from './lib/webgpu'
import { useActiveSession, useChat } from './state/chat'
import { useSession } from './state/session'

type Tab = 'chats' | 'settings' | 'metrics'

export default function App() {
  const gpu = useSession((s) => s.gpu)
  const newSession = useChat((s) => s.newSession)
  const activeTitle = useActiveSession()?.title
  const [tab, setTab] = useState<Tab>('chats')
  const [sidebarOpen, setSidebarOpen] = useState(false)

  useEffect(() => {
    detectWebGPU().then((info) => useSession.getState().set({ gpu: info }))
  }, [])

  if (!gpu) {
    return <div className="flex h-full items-center justify-center text-sm text-zinc-500">Checking WebGPU support…</div>
  }
  if (!gpu.supported) return <UnsupportedScreen gpu={gpu} />

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center gap-3 border-b border-zinc-200 bg-white px-4 py-2.5 dark:border-zinc-800 dark:bg-zinc-900">
        <button className="btn-secondary px-2 lg:hidden" onClick={() => setSidebarOpen((o) => !o)} aria-label="Toggle sidebar">
          ☰
        </button>
        <h1 className="font-semibold">Local LLM Chat</h1>
        <span className="hidden min-w-0 truncate text-sm text-zinc-500 md:inline">/ {activeTitle}</span>
        <span className="hidden shrink-0 rounded-full bg-zinc-100 px-2 py-0.5 text-xs text-zinc-600 sm:inline dark:bg-zinc-800 dark:text-zinc-400">
          {gpuLabel(gpu)}
        </span>
        <div className="flex-1" />
        <button className="btn-secondary shrink-0" onClick={newSession}>
          New chat
        </button>
      </header>

      <div className="flex min-h-0 flex-1">
        <aside
          className={`${sidebarOpen ? 'block' : 'hidden'} absolute inset-x-0 top-[53px] bottom-0 z-10 w-full space-y-3 overflow-y-auto bg-zinc-50 p-3 lg:static lg:block lg:w-80 lg:shrink-0 lg:border-r lg:border-zinc-200 dark:bg-zinc-950 lg:dark:border-zinc-800`}
        >
          <ErrorBoundary name="model">
            <ModelPanel />
          </ErrorBoundary>
          <section className="panel">
            <div className="mb-4 flex gap-1 rounded-lg bg-zinc-100 p-1 dark:bg-zinc-800">
              {(['chats', 'settings', 'metrics'] as const).map((t) => (
                <button
                  key={t}
                  className={`flex-1 rounded-md py-1 text-sm capitalize ${
                    tab === t ? 'bg-white shadow-sm dark:bg-zinc-900' : 'text-zinc-500'
                  }`}
                  onClick={() => setTab(t)}
                >
                  {t}
                </button>
              ))}
            </div>
            <ErrorBoundary key={tab} name={tab}>
              {tab === 'chats' && <ChatList onSelect={() => setSidebarOpen(false)} />}
              {tab === 'settings' && <SettingsPanel />}
              {tab === 'metrics' && <MetricsPanel />}
            </ErrorBoundary>
          </section>
        </aside>

        <main className="min-w-0 flex-1">
          <ErrorBoundary name="chat">
            <ChatView />
          </ErrorBoundary>
        </main>
      </div>
    </div>
  )
}
