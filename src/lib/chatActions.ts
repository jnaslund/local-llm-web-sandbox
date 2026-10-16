import { getActiveSession, newId, useChat, type Attachment, type ChatMessage } from '../state/chat'
import { useSession } from '../state/session'
import { generationSettings, useSettings } from '../state/settings'
import type { WireMessage } from '../worker/protocol'
import { generate, interrupt } from './llm'

/** Convert chat history to worker messages. Only the latest user turn carries real media. */
function toWire(history: ChatMessage[], systemPrompt: string): WireMessage[] {
  const lastUserIdx = history.findLastIndex((m) => m.role === 'user')
  const wire: WireMessage[] = []
  if (systemPrompt.trim()) wire.push({ role: 'system', text: systemPrompt.trim() })

  history.forEach((m, i) => {
    if (m.role === 'assistant' && (m.error || !m.text)) return
    const msg: WireMessage = { role: m.role, text: m.text }
    if (i === lastUserIdx) {
      const image = m.attachments?.find((a) => a.kind === 'image' && a.blob)
      const audio = m.attachments?.find((a) => a.kind === 'audio' && a.pcm)
      if (image?.blob) msg.image = image.blob
      if (audio?.pcm) msg.audio = audio.pcm
    } else if (m.attachments?.length) {
      msg.priorAttachments = m.attachments.map((a) => a.kind)
    }
    wire.push(msg)
  })
  return wire
}

/** Generate a reply into `sessionId`; it keeps streaming there even if the user switches sessions meanwhile. */
async function runAssistantTurn(sessionId: string) {
  const settings = useSettings.getState()
  const chat = useChat.getState()
  const history = chat.sessions.find((s) => s.id === sessionId)?.messages ?? []
  const assistantId = newId()
  chat.add(sessionId, { id: assistantId, role: 'assistant', text: '', pending: true })
  const loadedModelId = useSession.getState().loadedModelId
  if (loadedModelId) chat.setModel(sessionId, loadedModelId)
  useSession.getState().set({ generating: true })

  try {
    const metrics = await generate(
      toWire(history, settings.systemPrompt),
      generationSettings(settings),
      (text) => useChat.getState().appendText(sessionId, assistantId, text),
      settings.litertStateful ? sessionId : undefined,
    )
    useChat.getState().patch(sessionId, assistantId, { metrics, pending: false })
  } catch (err) {
    useChat
      .getState()
      .patch(sessionId, assistantId, { error: err instanceof Error ? err.message : String(err), pending: false })
  } finally {
    useSession.getState().set({ generating: false })
  }
}

export async function sendMessage(text: string, attachments: Attachment[]) {
  if (useSession.getState().generating) return
  const sessionId = getActiveSession().id
  useChat.getState().add(sessionId, { id: newId(), role: 'user', text, attachments })
  await runAssistantTurn(sessionId)
}

export async function regenerateLast() {
  const session = getActiveSession()
  const last = session.messages.at(-1)
  if (!last || last.role !== 'assistant' || useSession.getState().generating) return
  useChat.getState().removeFrom(session.id, last.id)
  await runAssistantTurn(session.id)
}

export function stopGeneration() {
  interrupt()
}
