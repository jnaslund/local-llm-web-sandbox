import { useEffect, useRef, useState, type ClipboardEvent, type DragEvent, type KeyboardEvent } from 'react'
import { decodeToMono16k, startRecording, type Recorder } from '../lib/audio'
import { sendMessage, stopGeneration } from '../lib/chatActions'
import type { Attachment } from '../state/chat'
import { useSession } from '../state/session'

export function Composer() {
  const [text, setText] = useState('')
  const [image, setImage] = useState<Attachment | null>(null)
  const [audio, setAudio] = useState<Attachment | null>(null)
  const [recorder, setRecorder] = useState<Recorder | null>(null)
  const [error, setError] = useState<string | null>(null)
  const imageInput = useRef<HTMLInputElement>(null)
  const audioInput = useRef<HTMLInputElement>(null)

  const ready = useSession((s) => s.status === 'ready')
  // LiteRT-LM 0.18 is text-only in practice (see litert.worker.ts).
  const textOnly = useSession((s) => s.loadMetrics?.runtime === 'litert')
  const mediaReady = ready && !textOnly
  const mediaHint = textOnly ? 'Attachments are not supported by LiteRT-LM yet' : null
  const generating = useSession((s) => s.generating)
  const benchmarking = useSession((s) => s.benchmarking)

  const recorderRef = useRef<Recorder | null>(null)
  // Release the microphone if the component unmounts mid-recording.
  useEffect(() => () => recorderRef.current?.cancel(), [])

  const attachImage = (file: File | Blob, name = 'image') => {
    if (!file.type.startsWith('image/')) return
    setImage({ kind: 'image', name: file instanceof File ? file.name : name, blob: file, previewUrl: URL.createObjectURL(file) })
  }

  const attachAudio = async (blob: Blob, name: string) => {
    try {
      setError(null)
      const pcm = await decodeToMono16k(blob)
      setAudio({ kind: 'audio', name, blob, pcm, previewUrl: URL.createObjectURL(blob) })
    } catch (err) {
      setError(`Could not decode audio: ${err instanceof Error ? err.message : String(err)}`)
    }
  }

  const toggleRecording = async () => {
    if (recorder) {
      const blob = await recorder.stop()
      recorderRef.current = null
      setRecorder(null)
      await attachAudio(blob, 'recording')
      return
    }
    try {
      setError(null)
      const r = await startRecording()
      recorderRef.current = r
      setRecorder(r)
    } catch (err) {
      setError(`Microphone unavailable: ${err instanceof Error ? err.message : String(err)}`)
    }
  }

  const canSend = ready && !generating && !recorder && (text.trim() || image || audio)

  const submit = () => {
    if (!canSend) return
    const attachments = [image, audio].filter((a): a is Attachment => a !== null)
    void sendMessage(text.trim(), attachments)
    setText('')
    setImage(null)
    setAudio(null)
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      submit()
    }
  }

  const onPaste = (e: ClipboardEvent) => {
    if (textOnly) return
    const file = Array.from(e.clipboardData.files).find((f) => f.type.startsWith('image/'))
    if (file) {
      e.preventDefault()
      attachImage(file)
    }
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    if (textOnly) return
    for (const file of Array.from(e.dataTransfer.files)) {
      if (file.type.startsWith('image/')) attachImage(file)
      else if (file.type.startsWith('audio/')) void attachAudio(file, file.name)
    }
  }

  return (
    <div className="border-t border-zinc-200 bg-white p-3 dark:border-zinc-800 dark:bg-zinc-900" onDragOver={(e) => e.preventDefault()} onDrop={onDrop}>
      {(image || audio || error) && (
        <div className="mb-2 flex flex-wrap items-center gap-2">
          {image && (
            <span className="flex items-center gap-2 rounded-lg bg-zinc-100 p-1 pr-2 text-xs dark:bg-zinc-800">
              <img src={image.previewUrl} alt="" className="size-10 rounded object-cover" />
              {image.name}
              <button className="text-zinc-500 hover:text-zinc-900 dark:hover:text-white" onClick={() => setImage(null)} aria-label="Remove image">
                ✕
              </button>
            </span>
          )}
          {audio && (
            <span className="flex items-center gap-2 rounded-lg bg-zinc-100 px-2 py-1 text-xs dark:bg-zinc-800">
              🎤 {audio.name} ({audio.pcm ? (audio.pcm.length / 16000).toFixed(1) : '?'} s)
              <button className="text-zinc-500 hover:text-zinc-900 dark:hover:text-white" onClick={() => setAudio(null)} aria-label="Remove audio">
                ✕
              </button>
            </span>
          )}
          {error && <span className="text-xs text-red-600">{error}</span>}
        </div>
      )}

      <div className="flex items-end gap-2">
        <div className="flex gap-1">
          <button className="btn-secondary px-2" title={mediaHint ?? 'Attach image'} disabled={!mediaReady} onClick={() => imageInput.current?.click()}>
            🖼️
          </button>
          <button className="btn-secondary px-2" title={mediaHint ?? 'Attach audio file'} disabled={!mediaReady} onClick={() => audioInput.current?.click()}>
            🎵
          </button>
          <button
            className={recorder ? 'btn-danger px-2' : 'btn-secondary px-2'}
            title={recorder ? 'Stop recording' : (mediaHint ?? 'Record from microphone')}
            disabled={!mediaReady && !recorder}
            onClick={toggleRecording}
          >
            {recorder ? '⏹' : '🎤'}
          </button>
        </div>

        <textarea
          className="input max-h-48 min-h-10 flex-1 resize-none"
          rows={1}
          placeholder={ready ? 'Message the model… (Enter to send, Shift+Enter for newline)' : 'Load a model to start chatting'}
          value={text}
          disabled={!ready}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          onPaste={onPaste}
        />

        {generating && !benchmarking ? (
          <button className="btn-danger" onClick={stopGeneration}>
            Stop
          </button>
        ) : (
          <button className="btn-primary" disabled={!canSend} onClick={submit}>
            Send
          </button>
        )}
      </div>

      <input
        ref={imageInput}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) attachImage(f)
          e.target.value = ''
        }}
      />
      <input
        ref={audioInput}
        type="file"
        accept="audio/*"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) void attachAudio(f, f.name)
          e.target.value = ''
        }}
      />
    </div>
  )
}
